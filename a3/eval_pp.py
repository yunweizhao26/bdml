import os
import glob
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time


class PerplexityCalculator:
    def __init__(
        self,
        model_path="3b-3.2_bs41_4bit_shuffling",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        print(f"Loading model from {model_path} for perplexity calculation...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Model loaded successfully for perplexity calculation")
    
    def calculate_perplexity(self, text, max_length=512):
        if not text or len(text.strip()) == 0:
            print(f"Warning: Empty text provided for perplexity calculation")
            return None
            
        try:
            tokens = self.tokenizer.encode(text)
            total_loss = 0.0
            total_tokens = 0
            
            for start_idx in range(0, len(tokens), max_length):
                end_idx = start_idx + max_length
                input_ids = tokens[start_idx:end_idx]
                if not input_ids:
                    continue
                    
                input_ids = torch.tensor([input_ids], device=self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    
                chunk_size = (end_idx - start_idx)
                total_loss += loss.item() * chunk_size
                total_tokens += chunk_size
                
            if total_tokens == 0:
                return None

            avg_loss = total_loss / total_tokens
            return torch.exp(torch.tensor(avg_loss)).item()
        except Exception as e:
            print(f"Error calculating perplexity: {str(e)}")
            return None


def extract_answers_from_json_files(results_dir):
    answers_data = {
        "rag": {},
        "fine_tuned": {}
    }
    
    all_files = glob.glob(os.path.join(results_dir, "*_detailed.json"))
    print(f"Found {len(all_files)} JSON files to analyze")
    
    for file_path in all_files:
        try:
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                print(f"File doesn't exist or is empty: {file_path}")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()
                if not file_content:
                    print(f"File has no content: {file_path}")
                    continue
                    
                results = json.loads(file_content)
                
            is_ft = "fine_tuned" in os.path.basename(file_path)
            
            detailed_metrics = results.get("detailed_metrics", [])
            if not detailed_metrics:
                print(f"No detailed metrics found in {file_path}")
                continue
                
            print(f"Found {len(detailed_metrics)} metrics in {file_path}")
            
            for metric in detailed_metrics:
                query = metric.get("query")
                answer = metric.get("answer")
                
                if not query or not answer:
                    continue
                    
                if metric.get("run", 1) != 1:
                    continue
                    
                if is_ft:
                    answers_data["fine_tuned"][query] = answer
                else:
                    embedding_model = results.get("overall_metrics", {}).get("embedding_model", "unknown")
                    index_type = results.get("overall_metrics", {}).get("index_type", "unknown")
                    key = f"{embedding_model}_{index_type}"
                    
                    if key not in answers_data["rag"]:
                        answers_data["rag"][key] = {}
                        
                    answers_data["rag"][key][query] = answer
                    
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    return answers_data


def analyze_perplexity(answers_data, model_path, output_dir):
    calc = PerplexityCalculator(model_path=model_path)
    
    perplexity_results = {
        "fine_tuned": {},
        "rag": {}
    }
    
    if answers_data["fine_tuned"]:
        print(f"\nCalculating perplexity for {len(answers_data['fine_tuned'])} fine-tuned answers...")
        for query, answer in tqdm(answers_data["fine_tuned"].items()):
            perplexity = calc.calculate_perplexity(answer)
            query_perplexity = calc.calculate_perplexity(query)
            
            perplexity_results["fine_tuned"][query] = {
                "answer_perplexity": perplexity,
                "query_perplexity": query_perplexity
            }
    
    if answers_data["rag"]:
        total_rag_answers = sum(len(config_answers) for config_answers in answers_data["rag"].values())
        print(f"\nCalculating perplexity for {total_rag_answers} RAG answers across {len(answers_data['rag'])} configurations...")
        
        for config, queries in answers_data["rag"].items():
            if config not in perplexity_results["rag"]:
                perplexity_results["rag"][config] = {}
                
            for query, answer in tqdm(queries.items(), desc=f"Config {config}"):
                perplexity = calc.calculate_perplexity(answer)
                query_perplexity = perplexity_results["fine_tuned"].get(query, {}).get("query_perplexity")
                
                perplexity_results["rag"][config][query] = {
                    "answer_perplexity": perplexity,
                    "query_perplexity": query_perplexity
                }
    
    results_dfs = {
        "fine_tuned": pd.DataFrame(),
        "rag": {}
    }
    
    if perplexity_results["fine_tuned"]:
        ft_data = []
        for query, metrics in perplexity_results["fine_tuned"].items():
            ft_data.append({
                "query": query,
                "answer_perplexity": metrics["answer_perplexity"],
                "query_perplexity": metrics["query_perplexity"]
            })
        results_dfs["fine_tuned"] = pd.DataFrame(ft_data)
    
    for config, queries in perplexity_results["rag"].items():
        rag_data = []
        for query, metrics in queries.items():
            rag_data.append({
                "query": query,
                "answer_perplexity": metrics["answer_perplexity"],
                "query_perplexity": metrics["query_perplexity"]
            })
        results_dfs["rag"][config] = pd.DataFrame(rag_data)
    
    print("\n=== Perplexity Analysis Results ===")
    
    if not results_dfs["fine_tuned"].empty:
        ft_df = results_dfs["fine_tuned"]
        ft_df = ft_df.dropna(subset=["answer_perplexity"])
        
        if not ft_df.empty:
            avg_perplexity = ft_df["answer_perplexity"].mean()
            std_perplexity = ft_df["answer_perplexity"].std()
            
            print(f"\nFine-tuned Model:")
            print(f"Average answer perplexity: {avg_perplexity:.2f} ± {std_perplexity:.2f}")
            print(f"Valid perplexity calculations: {len(ft_df)} out of {len(results_dfs['fine_tuned'])} answers")
            
            query_df = ft_df.dropna(subset=["query_perplexity"])
            if not query_df.empty:
                avg_query_perplexity = query_df["query_perplexity"].mean()
                std_query_perplexity = query_df["query_perplexity"].std()
                print(f"Average query perplexity: {avg_query_perplexity:.2f} ± {std_query_perplexity:.2f}")
    
    if results_dfs["rag"]:
        rag_summary = []
        
        for config, df in results_dfs["rag"].items():
            valid_df = df.dropna(subset=["answer_perplexity"])
            
            if not valid_df.empty:
                avg_perplexity = valid_df["answer_perplexity"].mean()
                std_perplexity = valid_df["answer_perplexity"].std()
                
                rag_summary.append({
                    "config": config,
                    "avg_perplexity": avg_perplexity,
                    "std_perplexity": std_perplexity,
                    "valid_count": len(valid_df),
                    "total_count": len(df)
                })
        
        if rag_summary:
            rag_summary_df = pd.DataFrame(rag_summary)
            rag_summary_df = rag_summary_df.sort_values(by="avg_perplexity")
            
            print("\nRAG Systems:")
            for _, row in rag_summary_df.iterrows():
                print(f"{row['config']}: {row['avg_perplexity']:.2f} ± {row['std_perplexity']:.2f} " +
                     f"({row['valid_count']}/{row['total_count']} valid)")
    
    if rag_summary and not results_dfs["fine_tuned"].empty:
        best_rag = min(rag_summary, key=lambda x: x["avg_perplexity"])
        ft_perplexity = ft_df["answer_perplexity"].mean() if not ft_df.empty else float('inf')
        
        print("\nComparison - Best RAG vs Fine-tuned:")
        print(f"Best RAG ({best_rag['config']}): {best_rag['avg_perplexity']:.2f}")
        print(f"Fine-tuned: {ft_perplexity:.2f}")
        
        ratio = best_rag["avg_perplexity"] / ft_perplexity if ft_perplexity > 0 else float('inf')
        print(f"Perplexity Ratio (RAG/Fine-tuned): {ratio:.2f}x")
        
        if ratio > 1:
            print(f"Fine-tuned model has {ratio:.2f}x lower perplexity (better)")
        else:
            print(f"Best RAG system has {1/ratio:.2f}x lower perplexity (better)")
    
    create_perplexity_visualizations(results_dfs, output_dir)
    save_perplexity_results(perplexity_results, output_dir)
    
    return perplexity_results


def create_perplexity_visualizations(results_dfs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    ft_df = results_dfs["fine_tuned"]
    has_ft = not ft_df.empty
    has_rag = bool(results_dfs["rag"])
    
    if has_ft or has_rag:
        labels = []
        perplexity_values = []
        std_devs = []
        
        if has_ft:
            ft_df = ft_df.dropna(subset=["answer_perplexity"])
            if not ft_df.empty:
                labels.append("Fine-tuned")
                perplexity_values.append(ft_df["answer_perplexity"].mean())
                std_devs.append(ft_df["answer_perplexity"].std())
        
        if has_rag:
            for config, df in results_dfs["rag"].items():
                df = df.dropna(subset=["answer_perplexity"])
                if not df.empty:
                    config_short = config.split('_')[-1]
                    labels.append(f"RAG ({config_short})")
                    perplexity_values.append(df["answer_perplexity"].mean())
                    std_devs.append(df["answer_perplexity"].std())
        
        plt.figure(figsize=(14, 8))
        plt.bar(labels, perplexity_values, yerr=std_devs, capsize=5)
        plt.ylabel('Perplexity (lower is better)')
        plt.title('Answer Perplexity Comparison')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'perplexity_comparison.png'))
        print(f"Saved perplexity comparison to {os.path.join(output_dir, 'perplexity_comparison.png')}")

        if has_ft:
            ft_df = results_dfs["fine_tuned"].dropna(subset=["answer_perplexity", "query_perplexity"])
            if not ft_df.empty:
                plt.figure(figsize=(10, 6))
                plt.scatter(ft_df["query_perplexity"], ft_df["answer_perplexity"], alpha=0.7)
                plt.xlabel('Query Perplexity')
                plt.ylabel('Answer Perplexity')
                plt.title('Query vs Answer Perplexity (Fine-tuned Model)')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                min_val = min(ft_df["query_perplexity"].min(), ft_df["answer_perplexity"].min())
                max_val = max(ft_df["query_perplexity"].max(), ft_df["answer_perplexity"].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'query_vs_answer_perplexity.png'))
                print(f"Saved query vs answer perplexity plot to {os.path.join(output_dir, 'query_vs_answer_perplexity.png')}")
                
                plt.figure(figsize=(12, 6))
                plt.hist(ft_df["answer_perplexity"], bins=15, alpha=0.7, label="Answer")
                plt.hist(ft_df["query_perplexity"], bins=15, alpha=0.7, label="Query")
                plt.xlabel('Perplexity')
                plt.ylabel('Frequency')
                plt.title('Perplexity Distribution (Fine-tuned Model)')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'perplexity_distribution.png'))
                print(f"Saved perplexity distribution to {os.path.join(output_dir, 'perplexity_distribution.png')}")


def save_perplexity_results(perplexity_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    if perplexity_results["fine_tuned"]:
        ft_data = []
        for query, metrics in perplexity_results["fine_tuned"].items():
            ft_data.append({
                "query": query,
                "answer_perplexity": metrics["answer_perplexity"],
                "query_perplexity": metrics["query_perplexity"]
            })
        
        ft_df = pd.DataFrame(ft_data)
        ft_path = os.path.join(output_dir, "fine_tuned_perplexity.csv")
        ft_df.to_csv(ft_path, index=False)
        print(f"Saved fine-tuned perplexity results to {ft_path}")
    
    if perplexity_results["rag"]:
        all_rag_data = []
        
        for config, queries in perplexity_results["rag"].items():
            for query, metrics in queries.items():
                all_rag_data.append({
                    "config": config,
                    "query": query,
                    "answer_perplexity": metrics["answer_perplexity"],
                    "query_perplexity": metrics["query_perplexity"]
                })
        
        rag_df = pd.DataFrame(all_rag_data)
        rag_path = os.path.join(output_dir, "rag_perplexity.csv")
        rag_df.to_csv(rag_path, index=False)
        print(f"Saved RAG perplexity results to {rag_path}")
    
    summary_data = []
    
    if perplexity_results["fine_tuned"]:
        ft_perplexities = [m["answer_perplexity"] for m in perplexity_results["fine_tuned"].values() 
                          if m["answer_perplexity"] is not None]
        
        if ft_perplexities:
            summary_data.append({
                "model_type": "fine-tuned",
                "config": "fine-tuned",
                "avg_perplexity": np.mean(ft_perplexities),
                "std_perplexity": np.std(ft_perplexities),
                "min_perplexity": np.min(ft_perplexities),
                "max_perplexity": np.max(ft_perplexities),
                "valid_count": len(ft_perplexities),
                "total_count": len(perplexity_results["fine_tuned"])
            })
    
    for config, queries in perplexity_results["rag"].items():
        rag_perplexities = [m["answer_perplexity"] for m in queries.values() 
                           if m["answer_perplexity"] is not None]
        
        if rag_perplexities:
            summary_data.append({
                "model_type": "rag",
                "config": config,
                "avg_perplexity": np.mean(rag_perplexities),
                "std_perplexity": np.std(rag_perplexities),
                "min_perplexity": np.min(rag_perplexities),
                "max_perplexity": np.max(rag_perplexities),
                "valid_count": len(rag_perplexities),
                "total_count": len(queries)
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "perplexity_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved perplexity summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze perplexity of stored model outputs")
    parser.add_argument("--results_dir", type=str, default="./results", 
                        help="Directory containing the detailed results files")
    parser.add_argument("--model_path", type=str, default="3b-3.2_bs41_4bit_shuffling",
                        help="Path to the model to use for perplexity calculation")
    parser.add_argument("--output_dir", type=str, default="./perplexity_results",
                        help="Directory to save perplexity analysis results")
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory {args.results_dir} does not exist.")
        return
    
    print(f"Extracting answers from JSON files in {args.results_dir}...")
    answers_data = extract_answers_from_json_files(args.results_dir)
    
    ft_count = len(answers_data["fine_tuned"])
    rag_configs = len(answers_data["rag"])
    rag_total = sum(len(answers) for answers in answers_data["rag"].values())
    
    print(f"\nExtracted {ft_count} fine-tuned answers and {rag_total} RAG answers across {rag_configs} configurations")
    
    if ft_count == 0 and rag_total == 0:
        print("No answers found to analyze. Please check the results directory.")
        return
    
    perplexity_results = analyze_perplexity(answers_data, args.model_path, args.output_dir)
    
    print("\nPerplexity analysis complete!")


if __name__ == "__main__":
    main()