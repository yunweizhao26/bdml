import math
import torch
import copy
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
from fairscale.nn.pipe.balance import balance_by_time
import concurrent.futures
import os
from tqdm import tqdm
import logging
import torch.nn as nn
from fairscale.nn import Pipe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

tokenized_datasets, tokenizer = load_and_prepare_dataset()

def load_model_with_pipeline_parallel():
    logger.info("Loading model with pipeline parallelism...")
    model_path = Path("./models/Llama3.2-3B")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    
    num_layers = len(model.model.layers)
    num_gpus = torch.cuda.device_count()
    
    layers_per_gpu = num_layers // num_gpus
    
    sequential_layers = nn.Sequential()
    
    class EmbeddingLayer(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.embed_tokens = model.model.embed_tokens
            
        def forward(self, input_ids):
            return self.embed_tokens(input_ids)
            
    sequential_layers.add_module("embed", EmbeddingLayer(model))
    
    for i in range(num_layers):
        class TransformerLayerWrapper(nn.Module):
            def __init__(self, layer_idx):
                super().__init__()
                self.layer = copy.deepcopy(model.model.layers[layer_idx])
                
            def forward(self, hidden_states):
                seq_length = hidden_states.shape[1]
                position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
                hidden_states = self.layer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=None,
                )[0]
                return hidden_states
                
        sequential_layers.add_module(f"layer{i}", TransformerLayerWrapper(i))
    
    class OutputLayer(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.norm = copy.deepcopy(model.model.norm)
            self.lm_head = copy.deepcopy(model.lm_head)
            
        def forward(self, hidden_states):
            hidden_states = self.norm(hidden_states)
            return self.lm_head(hidden_states)
            
    sequential_layers.add_module("output", OutputLayer(model))
    
    balance = []
    remaining_layers = num_layers + 2
    for i in range(num_gpus):
        if i < remaining_layers % num_gpus:
            balance.append(remaining_layers // num_gpus + 1)
        else:
            balance.append(remaining_layers // num_gpus)
    
    devices = [i for i in range(num_gpus)]
    pipeline_model = Pipe(
        sequential_layers,
        balance=balance,
        chunks=4,
        devices=devices,
        checkpoint="never",
    )
    
    class PipelineParallelModel(nn.Module):
        def __init__(self, pipe_model, config):
            super().__init__()
            self.pipe_model = pipe_model
            self.config = config
            self.device = torch.device("cuda:0")
            
        def forward(self, input_ids, labels=None):
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError(f"Expected input_ids to be a Tensor, got {type(input_ids)}")
            logits = self.pipe_model(input_ids)
            loss = None
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1)
                )
            return (loss, logits) if loss is not None else logits
    
    wrapped_model = PipelineParallelModel(pipeline_model, model.config)
    
    del model
    torch.cuda.empty_cache()
    
    return wrapped_model


model = load_model_with_pipeline_parallel()

lora_config = LoraConfig(r=16, lora_alpha=64, target_modules=["q_proj", "k_proj"], lora_dropout=0.1)
try:
    model = get_peft_model(model, lora_config)
    logger.info("LoRA applied successfully")
except Exception as e:
    logger.warning(f"Failed to apply LoRA to pipeline model: {e}")
    logger.info("Continuing without LoRA...")

training_args = TrainingArguments(
    output_dir="./llama-climate-pp",
    per_device_train_batch_size=10,
    gradient_accumulation_steps=4,
    fp16=True,
    optim="adamw_torch",
    dataloader_num_workers=8,
    dataloader_prefetch_factor=2,
    dataloader_drop_last=True,
    lr_scheduler_type="cosine",
    dataloader_pin_memory=True,
    num_train_epochs=5,
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

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

class PipelineTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        
        input_ids = inputs["input_ids"]
        
        outputs = model(input_ids, labels=labels)
        
        if isinstance(outputs, tuple):
            loss, logits = outputs[0], outputs[1]
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
        
        return (loss, logits) if return_outputs else loss

trainer = PipelineTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger.info("Starting pipeline parallelism training!")
try:
    trainer.train()
    model_name = "3b-3.2-pp"
    batch_size = 10
    precision = "fp16"
    trainer.save_model(f"{model_name}_bs{batch_size}_2gpus_{precision}")

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
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
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
                with open(file_path, "r") as f:
                    text = f.read()
                return calculate_perplexity_chunked(model, tokenizer, text)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                return None
        test_file_paths = [os.path.join("test", f) for f in test_files if f.endswith(".txt")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            perplexities = list(tqdm(
                executor.map(calculate_single_perplexity, test_file_paths),
                total=len(test_file_paths),
                desc="Calculating Perplexity"
            ))
        perplexities = [p for p in perplexities if p is not None]
        avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float("nan")
        return {"perplexities": perplexities, "average_perplexity": avg_perplexity,
                "total_files": len(test_file_paths), "processed_files": len(perplexities)}

    logger.info("Evaluating model perplexity...")
    test_files = os.listdir("test/")
    result = parallel_calculate_perplexity(model, tokenizer, test_files, max_workers=os.cpu_count())
    logger.info(f"Average perplexity: {result['average_perplexity']:.2f}")
    logger.info(f"Total files: {result['total_files']}")
    logger.info(f"Processed files: {result['processed_files']}")
except Exception as e:
    logger.error(f"Training failed: {e}", exc_info=True)
