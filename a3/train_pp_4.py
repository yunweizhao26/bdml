import math
import torch
import torch.nn.functional as F
from pathlib import Path
import os
import deepspeed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig
)
from deepspeed.pipe import PipelineModule, LayerSpec

class EmbeddingPipe(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
    
    def forward(self, input_ids, *rest):
        embeddings = self.embed(input_ids)
        embeddings.requires_grad_(True)
        return (embeddings, *rest)

class LlamaDecoderLayerPipe(torch.nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        self.layer = LlamaDecoderLayer(config, layer_idx=layer_idx)
    
    def forward(self, hidden_states, *rest):
        print("LlamaDecoderLayerPipe")
        if len(hidden_states.shape) == 2:
            seq_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.unsqueeze(0)
        
        batch_size, seq_length, _ = hidden_states.shape
        
        attention_mask = torch.ones(batch_size, seq_length, device=hidden_states.device)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        
        outputs = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        return (outputs[0], *rest)

class LlamaRMSNormPipe(torch.nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        self.norm = LlamaRMSNorm(hidden_size, eps=eps)
    
    def forward(self, args):
        hidden_states = args
        print("LlamaRMSNormPipe")
        normalized = self.norm(hidden_states)
        return (normalized,)

class LlamaLMHeadPipe(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, hidden_states, *rest):
        print("LlamaLMHeadPipe")
        logits = self.lm_head(hidden_states)
        return (logits, *rest)

def loss_fn(outputs, labels):
    logits = outputs
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return loss

def create_pipeline_model(model_path, world_size, pipe_parallel_size=1):
    config = LlamaConfig.from_pretrained(model_path)
    
    dp_size = world_size // pipe_parallel_size
    
    layers = []
    
    layers.append(LayerSpec(EmbeddingPipe, config.vocab_size, config.hidden_size))
    
    if pipe_parallel_size <= 2:
        layers_per_stage = config.num_hidden_layers // pipe_parallel_size
    else:
        layers_per_stage = config.num_hidden_layers // (pipe_parallel_size - 2)
    
    for i in range(config.num_hidden_layers):
        layers.append(LayerSpec(LlamaDecoderLayerPipe, config, i))
    
    layers.append(LayerSpec(LlamaRMSNormPipe, config.hidden_size, config.rms_norm_eps))
    layers.append(LayerSpec(LlamaLMHeadPipe, config.hidden_size, config.vocab_size))
    
    pipe_model = PipelineModule(
        layers=layers,
        loss_fn=loss_fn,
        num_stages=pipe_parallel_size,
        partition_method='uniform'
    )
    return pipe_model

def setup_deepspeed(model, args):
    config = {
        "train_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "steps_per_print": 10,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps,
                "total_num_steps": args.total_steps
            }
        },
        "fp16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 0
        },
        "pipeline": {
            "activation_checkpoint_interval": 1
        }
    }
    
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=config,
    )
    
    return engine

def train_pipeline_model():
    deepspeed.init_distributed()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    class Args:
        def __init__(self):
            self.batch_size = 1
            self.gradient_accumulation_steps = 1
            self.learning_rate = 2e-5
            self.weight_decay = 0.01
            self.warmup_steps = 100
            self.total_steps = 1000
    
    args = Args()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    model_path = Path("./models/Llama3.2-3B")
    pipe_model = create_pipeline_model(model_path, world_size, pipe_parallel_size=2)

    engine = setup_deepspeed(pipe_model, args)
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.pad_token = tokenizer.eos_token
    
    from datasets import load_dataset
    data_files = {
        "train": "train/*.txt",
        "test": "test/*.txt"
    }
    raw_datasets = load_dataset("text", data_files=data_files)
    
    block_size = 512
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        cache_file_names={
            "train": "./cache/train_cache.arrow",
            "test": "./cache/test_cache.arrow"
        }
    )
    
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [concatenated[k][i : i + block_size] 
                for i in range(0, total_length, block_size)]
            for k in concatenated.keys()
        }
        return result
    
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )
    
    train_dataset = tokenized_datasets["train"]
    
    from torch.utils.data import DataLoader
    
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            item = self.dataset[idx]
            input_ids = torch.tensor(item["input_ids"])
            labels = input_ids.clone()
            return input_ids, labels
    
    train_dataset = TextDataset(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    train_loader = deepspeed.utils.RepeatingLoader(train_dataloader)
    train_iter = iter(train_loader)

    engine.train()
    for epoch in range(5):
        for step, batch in enumerate(train_dataloader):
            loss = engine.train_batch(data_iter=train_iter)
            if global_rank == 0 and step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
    
    if global_rank == 0:
        engine.save_checkpoint("./output")
        print("Training completed and model saved.")

if __name__ == "__main__":
    train_pipeline_model()