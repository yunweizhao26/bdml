import math
import torch
import deepspeed
import os
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from deepspeed.pipe import PipelineModule
from tqdm import tqdm
import logging
import argparse
import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_stages", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--ds_config", type=str, default="ds_config.json")
    parser.add_argument("--output_dir", type=str, default="./llama-pipeline-ckpt")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

class FinalLayerNormPipeLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = torch.nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, ipt):
        hidden_states, *_rest, labels = ipt
        hidden_states = self.norm(hidden_states)
        return hidden_states, labels

class EmbeddingPipeLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.padding_idx = config.pad_token_id or 0
        
    def forward(self, ipt):
        if isinstance(ipt, tuple) and len(ipt) >= 2:
            input_ids, labels = ipt[0], ipt[1]
        else:
            input_ids = ipt
            labels = input_ids.clone() if isinstance(input_ids, torch.Tensor) else None
        
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"Expected input_ids to be a tensor, got {type(input_ids)}")
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if labels is not None and isinstance(labels, torch.Tensor) and labels.dim() == 1:
                labels = labels.unsqueeze(0)
            
        hidden_states = self.embed_tokens(input_ids)
        
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        padding_idx = getattr(self.embed_tokens, 'padding_idx', self.padding_idx)
        if padding_idx is None:
            padding_idx = 0
            
        attention_mask = (input_ids != padding_idx).long()
        
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        return hidden_states, position_ids, attention_mask, position_embeddings, labels

class LlamaLayerPipeLayer(torch.nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        self.layer = LlamaDecoderLayer(config, layer_idx)

    @staticmethod
    def _expand_attention(attention_mask, seq_len, dtype):
        if attention_mask is None:
            padding = 0
        else:
            padding = (1 - attention_mask)[:, None, None, :].to(dtype)
        min_val = torch.finfo(dtype).min
        padding = padding * min_val

        causal = torch.triu(
            torch.full((seq_len, seq_len),
                       min_val, dtype=dtype, device=padding.device),
            diagonal=1
        )[None, None, :, :]

        return padding + causal

    def forward(self, ipt):
        hidden_states, position_ids, attention_mask, position_embeddings, labels = ipt

        attn_mask = self._expand_attention(
             attention_mask,
             seq_len=hidden_states.size(1),
             dtype=hidden_states.dtype,
        )
        if hidden_states.size(-1) != self.layer.self_attn.o_proj.weight.size(0):
            raise RuntimeError(
                f"hidden {hidden_states.shape}  vs  o_proj.weight {self.layer.self_attn.o_proj.weight.shape}"
            )

        outputs = self.layer(
            hidden_states=hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        hidden_states = outputs[0]
        return hidden_states, position_ids, attention_mask, position_embeddings, labels

class LMHeadPipeLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, ipt):
        hidden_states, labels = ipt
        logits = self.lm_head(hidden_states)
        return logits, labels

class LossPipeLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        
    def forward(self, ipt):
        logits, labels = ipt
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        )
        return loss

def load_and_prepare_dataset():
    logger.info("Loading and preparing dataset...")
    data_files = {"train": "train/*.txt", "test": "test/*.txt"}
    raw_datasets = load_dataset("text", data_files=data_files)
    model_path = Path("./models/Llama3.2-3B")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    block_size = 512

    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=block_size)
        return tokens

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        cache_file_names={"train": "./cache/train_cache.arrow", "test": "./cache/test_cache.arrow"}
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        
        if total_length < block_size:
            for k in concatenated:
                concatenated[k] = concatenated[k] + [tokenizer.pad_token_id if k == "input_ids" else 0] * (block_size - total_length)
            total_length = block_size
            
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        result = {
            k: [torch.tensor(concatenated[k][i : i + block_size], dtype=torch.long) 
                for i in range(0, total_length, block_size)]
            for k in concatenated.keys()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(group_texts, batched=True)
    return tokenized_datasets, tokenizer

def create_pipeline_model(model_path, num_stages):
    logger.info("Creating pipeline model...")
    
    config = AutoConfig.from_pretrained(model_path)
    config.pretraining_tp = 1
    
    model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float16)
    
    layers = models.llama_ds_mp_wrap.get_layers_from_config(config)

    logger.info(f"Created {len(layers)} layers for pipeline")
    
    del model
    torch.cuda.empty_cache()
    
    model_pipe = PipelineModule(layers=layers,
                            num_stages=num_stages,
                            loss_fn=models.llama_ds_mp_wrap.loss_fn,
                            activation_checkpoint_interval=0)

    return model_pipe, config

class CustomDataIterator:
    def __init__(self, dataloader, device, num_microbatches):
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader)
        self.device = device
        self.num_microbatches = num_microbatches
        self.current_batch = 0

    def __iter__(self):
        self.current_batch = 0
        self.dataloader_iter = iter(self.dataloader)
        return self

    def __next__(self):
        if self.current_batch >= self.num_microbatches:
            raise StopIteration
        
        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)
        
        self.current_batch += 1
        
        if isinstance(batch, dict) and "input_ids" in batch:
            if isinstance(batch["input_ids"], list):
                input_ids = batch["input_ids"][0] if batch["input_ids"] else torch.zeros((1, 1), dtype=torch.long)
            else:
                input_ids = batch["input_ids"]
        elif isinstance(batch, list) and batch and isinstance(batch[0], dict) and "input_ids" in batch[0]:
            input_ids = batch[0]["input_ids"]
        else:
            input_ids = batch
            
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            
        input_ids = input_ids.to(self.device)
            
        if input_ids.dim() == 0:
            input_ids = input_ids.unsqueeze(0).unsqueeze(0)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        labels = input_ids.clone()
        
        return (input_ids, labels)

def main():
    args = parse_args()
    
    deepspeed.init_distributed()
    
    dataset, tokenizer = load_and_prepare_dataset()
    
    model, config = create_pipeline_model("./models/Llama3.2-3B", args.num_stages)
    
    if os.path.exists(args.ds_config):
        with open(args.ds_config, 'r') as f:
            ds_config = json.load(f)
        logger.info(f"Loaded DeepSpeed config from {args.ds_config}")
        
        if "zero_optimization" in ds_config:
            if "reduce_bucket_size" in ds_config["zero_optimization"]:
                bucket_size = ds_config["zero_optimization"]["reduce_bucket_size"]
                if isinstance(bucket_size, str) and bucket_size.endswith("MB"):
                    size_mb = int(bucket_size.replace("MB", ""))
                    ds_config["zero_optimization"]["reduce_bucket_size"] = size_mb * 1024 * 1024
                    
            if "allgather_bucket_size" in ds_config["zero_optimization"]:
                bucket_size = ds_config["zero_optimization"]["allgather_bucket_size"]
                if isinstance(bucket_size, str) and bucket_size.endswith("MB"):
                    size_mb = int(bucket_size.replace("MB", ""))
                    ds_config["zero_optimization"]["allgather_bucket_size"] = size_mb * 1024 * 1024
    else:
        logger.warning(f"Config file {args.ds_config} not found. Using default config.")
        ds_config = {
            "train_batch_size": 16,
            "train_micro_batch_size_per_gpu": args.micro_batch_size,
            "steps_per_print": 10,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 2e-4,
                    "weight_decay": 0.001,
                    "betas": [0.9, 0.99],
                    "eps": 1e-6
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 2e-4,
                    "warmup_num_steps": 50,
                    "warmup_type": "linear"
                }
            },
            "gradient_clipping": 5.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 12,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": 1,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 52428800,
                "allgather_partitions": True,
                "allgather_bucket_size": 52428800
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "contiguous_memory_optimization": True,
                "cpu_checkpointing": False
            },
            "wall_clock_breakdown": False,
            "pipeline": {
                "activation_checkpoint_interval": 1
            }
        }
    
    ds_config["train_micro_batch_size_per_gpu"] = args.micro_batch_size
    num_microbatches = ds_config.get("gradient_accumulation_steps", 1)

    train_batch_size = ds_config["train_batch_size"]
    num_updates_per_epoch = len(dataset["train"]) // train_batch_size
    total_steps = num_updates_per_epoch * args.epochs
    
    if "scheduler" in ds_config and "params" in ds_config["scheduler"]:
        ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
    
    logger.info(f"Training for {total_steps} total steps")
    
    def collate_fn(batch):
        if all(isinstance(item, dict) for item in batch):
            input_ids = [item["input_ids"] for item in batch]
            if isinstance(input_ids[0], list):
                max_len = max(len(ids) for ids in input_ids)
                input_ids = [torch.tensor(ids + [tokenizer.pad_token_id] * (max_len - len(ids)), dtype=torch.long) 
                           for ids in input_ids]
            return {"input_ids": torch.stack(input_ids) if isinstance(input_ids[0], torch.Tensor) else input_ids}
        else:
            return {"input_ids": torch.stack(batch) if isinstance(batch[0], torch.Tensor) else batch}
    
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        args=args,
        model_parameters=model.parameters()
    )
    
    train_dataset = dataset["train"]
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    sample = next(iter(train_dataloader))
    ids = sample["input_ids"].to(engine.device)
    mask = (ids != tokenizer.pad_token_id).long()
    expanded = LlamaLayerPipeLayer._expand_attention(mask, ids.size(1), torch.float16)
    print(mask.shape, expanded.shape)
    
    logger.info(f"Starting training with pipeline parallelism using {args.num_stages} stages")
    logger.info(f"Training batch size: {train_batch_size}, Micro batch size: {args.micro_batch_size}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        engine.train()
        progress_bar = tqdm(range(num_updates_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step in progress_bar:
            batch_iterator = CustomDataIterator(train_dataloader, engine.device, num_microbatches)
            loss = engine.train_batch(batch_iterator)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            if step % ds_config["steps_per_print"] == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Step: {step}, Loss: {loss.item():.4f}")
                
        if torch.distributed.get_rank() == 0:
            output_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
            logger.info(f"Saving model checkpoint to {output_dir}")
            engine.save_checkpoint(output_dir)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
