import math
import torch
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    TrainingArguments, 
    Trainer, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling
)
import concurrent.futures
import os
from tqdm import tqdm
import torch.distributed as dist
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel
)
from torch.distributed.device_mesh import DeviceMesh

def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size

local_rank, global_rank, world_size = init_distributed()

assert world_size == 2, f"This script requires exactly 2 GPUs, but got {world_size}"
mesh_shape = torch.arange(world_size)
tp_mesh = DeviceMesh("cuda", mesh_shape)

def load_and_prepare_dataset():
    if global_rank == 0:
        print("Loading and preparing dataset...")
    
    data_files = {
        "train": "train/*.txt",
        "test": "test/*.txt"
    }
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
    
    return tokenized_datasets, tokenizer

tokenized_datasets, tokenizer = load_and_prepare_dataset()

def load_model_with_tp():
    if global_rank == 0:
        print("Loading model with tensor parallelism...")
    
    model_path = Path("./models/Llama3.2-3B")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float16,
        use_cache=False,
    )
    
    tp_plan = {
        "q_proj": ColwiseParallel(),
        "k_proj": ColwiseParallel(),
        "v_proj": ColwiseParallel(),
        "o_proj": RowwiseParallel(),
        "gate_proj": ColwiseParallel(),
        "up_proj": ColwiseParallel(),
        "down_proj": RowwiseParallel(),
    }
    
    for layer in model.model.layers:
        parallelize_module(
            layer,
            tp_mesh,
            tp_plan
        )
    
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    return model.to(f"cuda:{local_rank}")

model = load_model_with_tp()

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

training_args = TrainingArguments(
    output_dir="./llama-climate-tp",
    per_device_train_batch_size=41,
    gradient_accumulation_steps=4,
    fp16=True,
    optim="adamw_torch",
    dataloader_num_workers=8,
    dataloader_prefetch_factor=2,
    dataloader_drop_last=True,
    lr_scheduler_type="cosine",
    dataloader_pin_memory=True,
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=500,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1, 
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if global_rank == 0:
    print("Starting tensor parallelism training!")

try:
    trainer.train()
    
    if global_rank == 0:
        model_name = "3b-3.2-tp"
        batch_size = 20
        precision = "fp16"
        trainer.save_model(f"{model_name}_bs{batch_size}_tp{world_size}_{precision}")
        
        def calculate_perplexity_chunked(model, tokenizer, text, max_length=512):
            tokens = tokenizer.encode(text)
            total_loss = 0.0
            total_tokens = 0
            for start_idx in range(0, len(tokens), max_length):
                end_idx = start_idx + max_length
                input_ids = tokens[start_idx:end_idx]
                if not input_ids:
                    continue
                input_ids = torch.tensor([input_ids], device=model.device)
                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs.loss
                chunk_size = (end_idx - start_idx)
                total_loss += loss.item() * chunk_size
                total_tokens += chunk_size
            if total_tokens == 0:
                return None

            avg_loss = total_loss / total_tokens
            return math.exp(avg_loss)

        def parallel_calculate_perplexity(model, tokenizer, test_files, max_workers=None):
            def calculate_single_perplexity(file_path):
                try:
                    with open(file_path, 'r') as f:
                        text = f.read()
                    ppl = calculate_perplexity_chunked(model, tokenizer, text, max_length=512)
                    return ppl
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    return None
                    
            test_file_paths = [os.path.join("test", f) for f in test_files if f.endswith('.txt')]
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                perplexities = list(tqdm(
                    executor.map(calculate_single_perplexity, test_file_paths), 
                    total=len(test_file_paths),
                    desc="Calculating Perplexity"
                ))
                
            perplexities = [p for p in perplexities if p is not None]
            avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float('nan')
            return {
                'perplexities': perplexities,
                'average_perplexity': avg_perplexity,
                'total_files': len(test_file_paths),
                'processed_files': len(perplexities)
            }

        print("Evaluating model perplexity...")
        test_files = os.listdir("test/")
        result = parallel_calculate_perplexity(
            model, 
            tokenizer, 
            test_files, 
            max_workers=int(os.cpu_count() * 0.8)
        )
        
        print(f"Average perplexity: {result['average_perplexity']:.2f}")
        print(f"Total files: {result['total_files']}")
        print(f"Processed files: {result['processed_files']}")
except Exception as e:
    print(f"Training failed with error: {e}")
finally:
    dist.destroy_process_group()