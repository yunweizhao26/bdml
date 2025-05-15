import os
import math
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader
from tqdm import tqdm
import logging
import argparse
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="./llama-climate-deepspeed")
    parser.add_argument("--model_path", type=str, default="./models/Llama3.2-3B")
    parser.add_argument("--num_stages", type=int, default=1)
    parser.add_argument("--zero_stage", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for training")
    parser.add_argument("--config", type=str, default="ds_config.json", help="DeepSpeed configuration file")
    
    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()
    return args

def load_and_prepare_dataset():
    logger.info("Loading and preparing dataset...")
    data_files = {"train": "train/*.txt", "test": "test/*.txt"}
    raw_datasets = load_dataset("text", data_files=data_files)
    model_path = Path("./models/Llama3.2-3B")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.pad_token = tokenizer.eos_token

    block_size = 512

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        cache_file_names={"train": "./cache/train_cache.arrow", "test": "./cache/test_cache.arrow"}
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [concatenated[k][i : i + block_size] for i in range(0, total_length, block_size)]
            for k in concatenated.keys()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(group_texts, batched=True)
    return tokenized_datasets, tokenizer

class LlamaModelStage(torch.nn.Module):
    def __init__(self, model_path, start_layer_idx, end_layer_idx, is_first=False, is_last=False):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path)
        
        self.is_first = is_first
        self.is_last = is_last
        
        if is_first:
            self.embed_tokens = torch.nn.Embedding(
                config.vocab_size, 
                config.hidden_size, 
                padding_idx=config.pad_token_id
            )
            
        self.layers = torch.nn.ModuleList()
        for i in range(start_layer_idx, end_layer_idx):
            self.layers.append(self._create_layer_from_config(config, i))
            
        if is_last:
            self.norm = torch.nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.config = config
            
    def _create_layer_from_config(self, config, layer_idx):
        try:
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            return LlamaDecoderLayer(config, layer_idx=layer_idx)
        except ImportError:
            logger.warning("Could not import LlamaDecoderLayer directly. Using a basic implementation.")
            return torch.nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                activation="gelu",
                batch_first=True
            )
    
    def forward(self, hidden_states=None, input_ids=None, labels=None, attention_mask=None):
        if self.is_first and input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            if attention_mask is not None:
                position_ids = (attention_mask.long().cumsum(-1) - 1)
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                batch_size, seq_length = hidden_states.shape[:2]
                position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                
            outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = outputs[0]
        
        if self.is_last:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            
            loss = None
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1)
                )
                return loss
            
            return logits
        
        return hidden_states

def create_pipeline_model(args, model_path):
    config = AutoConfig.from_pretrained(model_path)
    num_layers = config.num_hidden_layers
    layers_per_stage = num_layers // args.num_stages
    
    logger.info(f"Creating pipeline model with {args.num_stages} stages")
    logger.info(f"Model has {num_layers} layers, each stage will have ~{layers_per_stage} layers")
    
    pipeline_specs = []
    start_idx = 0
    
    for stage_idx in range(args.num_stages):
        is_first = (stage_idx == 0)
        is_last = (stage_idx == args.num_stages - 1)
        
        if stage_idx < args.num_stages - 1:
            end_idx = start_idx + layers_per_stage
        else:
            end_idx = num_layers
            
        logger.info(f"Stage {stage_idx}: Layers {start_idx} to {end_idx-1}, First: {is_first}, Last: {is_last}")
            
        stage = LlamaModelStage(
            model_path=model_path,
            start_layer_idx=start_idx,
            end_layer_idx=end_idx,
            is_first=is_first,
            is_last=is_last
        )
        
        start_idx = end_idx
        
        pipeline_specs.append(stage)
    
    model = PipelineModule(
        layers=pipeline_specs,
        loss_fn=None,
        num_stages=args.num_stages,
        partition_method='uniform'
    )
    
    return model

def load_model_checkpoint(model, model_path):
    logger.info(f"Loading pretrained weights from {model_path}")
    
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    
    for stage_idx, stage in enumerate(model.pipeline):
        if hasattr(stage, 'is_first') and stage.is_first:
            stage.embed_tokens.load_state_dict(pretrained_model.model.embed_tokens.state_dict())
            
        if hasattr(stage, 'layers'):
            if stage_idx == 0:
                start_layer = 0
            else:
                start_layer = sum(len(prev_stage.layers) for prev_stage in model.pipeline[:stage_idx])
                
            for i, layer in enumerate(stage.layers):
                pretrained_layer = pretrained_model.model.layers[start_layer + i]
                layer.load_state_dict(pretrained_layer.state_dict())
        
        if hasattr(stage, 'is_last') and stage.is_last:
            stage.norm.load_state_dict(pretrained_model.model.norm.state_dict())
            stage.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())
    
    del pretrained_model
    torch.cuda.empty_cache()
    
    return model

def create_deepspeed_config(args):
    config = {
        "train_batch_size": args.batch_size * torch.cuda.device_count(),
        "gradient_accumulation_steps": 1,
        "fp16": {
            "enabled": args.fp16,
        },
        "zero_optimization": {
            "stage": args.zero_stage,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
            }
        },
        "pipeline": {
            "stages": args.num_stages,
            "activation_checkpoint_interval": 1
        }
    }
    return config

def main():
    args = parse_args()
    
    deepspeed.init_distributed()
    
    datasets, tokenizer = load_and_prepare_dataset()
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    train_dataloader = torch.utils.data.DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        pin_memory=True,
    )
    
    eval_dataloader = torch.utils.data.DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False,
        pin_memory=True,
    )
    
    model = create_pipeline_model(args, args.model_path)
    
    model = load_model_checkpoint(model, args.model_path)
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            ds_config = json.load(f)
        logger.info(f"Loaded DeepSpeed config from {args.config}")
    else:
        ds_config = create_deepspeed_config(args)
        logger.info("Using programmatically created DeepSpeed config")
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    logger.info("Starting training with DeepSpeed Pipeline Parallelism!")
    
    train_dataloader = RepeatingLoader(train_dataloader)
    
    num_training_steps = len(datasets["train"]) // (args.batch_size * torch.cuda.device_count()) * args.epochs
    
    for epoch in range(args.epochs):
        model_engine.train()
        train_loss = 0.0
        
        progress_bar = tqdm(range(len(datasets["train"]) // (args.batch_size * torch.cuda.device_count())))
        for step in progress_bar:
            loss = model_engine.train_batch(iter(train_dataloader))
            train_loss += loss.item()
            
            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}")
            
            if step % 10 == 0 and args.local_rank == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Step {step}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / (step + 1)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_train_loss:.4f}")
        
        if args.local_rank == 0:
            output_dir = f"{args.output_dir}/epoch-{epoch+1}"
            os.makedirs(output_dir, exist_ok=True)
            model_engine.save_checkpoint(output_dir)
            logger.info(f"Saved checkpoint to {output_dir}")
        
        if args.local_rank == 0:
            model_engine.eval()
            eval_loss = 0.0
            eval_steps = 0
            
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(model_engine.device) for k, v in batch.items()}
                
                with torch.no_grad():
                    loss = model_engine(batch["input_ids"], labels=batch["labels"])
                    eval_loss += loss.item()
                    eval_steps += 1
            
            avg_eval_loss = eval_loss / eval_steps
            perplexity = math.exp(avg_eval_loss)
            logger.info(f"Evaluation - Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    main()