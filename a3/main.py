import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
from rag import RAGSystem
from typing import List, Dict, Any
from tqdm import tqdm
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


class FineTunedModelEvaluator:
    def __init__(
        self, 
        model_path="3b-3.2_bs41_4bit_shuffling",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        print(f"Loading fine-tuned model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Fine-tuned model loaded successfully")
    
    def generate(self, query, max_new_tokens=4096):
        start_time = time.time()
        
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if response.startswith(query):
            response = response[len(query):].strip()
        
        generation_time = time.time() - start_time
        
        return response, generation_time
    

def generate_standard_queries(documents, num_queries=10, seed=42):
    np.random.seed(seed)
    
    query_templates = [
        "What is {topic}?",
        "Explain the concept of {topic}.",
        "How does {topic} work?",
        "What are the main components of {topic}?",
        "What are the advantages of {topic}?",
        "What are the limitations of {topic}?",
        "How does {topic} compare to {alt_topic}?",
        "What are the applications of {topic}?",
        "Summarize the current state of research on {topic}.",
        "What are the future directions for {topic}?",
        "What challenges exist in implementing {topic}?",
        "Who are the key researchers in {topic}?",
        "When was {topic} first developed?",
        "What tools are used for {topic}?",
        "How is {topic} evaluated?",
    ]
    
    topics = []
    for doc in documents[:min(5, len(documents))]:
        paragraphs = doc.split('\n\n')
        for para in paragraphs[:3]:
            words = para.split()
            potential_topics = [w for w in words if len(w) >= 5 and w.lower() not in 
                              ['about', 'above', 'across', 'after', 'again', 'against', 'there', 'these', 'those', 'their']]
            if potential_topics:
                topics.extend(potential_topics[:5])
    
    if not topics or len(topics) < 10:
        topics = ["machine learning", "data science", "natural language processing", 
                 "artificial intelligence", "neural networks", "computer vision",
                 "deep learning", "reinforcement learning", "climate change", 
                 "data analysis", "big data", "quantum computing", "algorithms"]
    
    topics = list(set(topics))
    queries = []
    alt_topics = topics.copy()
    np.random.shuffle(alt_topics)
    np.random.shuffle(query_templates)
    
    for i in range(min(num_queries, len(query_templates))):
        topic = topics[i % len(topics)]
        alt_topic = alt_topics[i % len(alt_topics)]
        query = query_templates[i].format(topic=topic, alt_topic=alt_topic)
        queries.append(query)
    
    return queries[:num_queries]


def evaluate_rag_system(rag_system, queries, embedding_model_name, index_type, top_k=5, num_runs=3):
    all_metrics = []
    
    print(f"Evaluating RAG with {embedding_model_name} and {index_type} index...")
    
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        run_metrics = []
        
        for query in tqdm(queries, desc=f"Evaluating queries (run {run+1})"):
            retrieval_start = time.time()
            retrieved_results = rag_system.retrieve(query, top_k)
            retrieval_time = time.time() - retrieval_start
            context = "\n\n".join([chunk for chunk, _ in retrieved_results])
            generation_start = time.time()
            answer = rag_system.generate(query, context)
            generation_time = time.time() - generation_start
            total_time = retrieval_time + generation_time
            context_size = len(context)
            answer_length = len(answer)
            metrics = {
                "query": query,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "context_size": context_size,
                "answer_length": answer_length,
                "run": run + 1,
                "answer": answer
            }
            
            run_metrics.append(metrics)
        
        all_metrics.extend(run_metrics)
    
    df = pd.DataFrame(all_metrics)
    
    agg_df = df.groupby('query').agg({
        'retrieval_time': ['mean', 'std'],
        'generation_time': ['mean', 'std'],
        'total_time': ['mean', 'std'],
        'context_size': ['mean', 'std'],
        'answer_length': ['mean', 'std']
    }).reset_index()
    
    overall_metrics = {
        "embedding_model": embedding_model_name,
        "index_type": index_type,
        "avg_retrieval_time": df['retrieval_time'].mean(),
        "std_retrieval_time": df['retrieval_time'].std(),
        "avg_generation_time": df['generation_time'].mean(),
        "std_generation_time": df['generation_time'].std(),
        "avg_total_time": df['total_time'].mean(),
        "std_total_time": df['total_time'].std(),
        "avg_context_size": df['context_size'].mean(),
        "std_context_size": df['context_size'].std(),
        "avg_answer_length": df['answer_length'].mean(),
        "std_answer_length": df['answer_length'].std(),
    }
    
    return {
        "detailed_metrics": df.to_dict('records'),
        "overall_metrics": overall_metrics
    }


def evaluate_fine_tuned_model(ft_evaluator, queries, num_runs=3):
    all_metrics = []
    
    print(f"Evaluating fine-tuned model...")
    
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        run_metrics = []
        
        for query in tqdm(queries, desc=f"Evaluating queries (run {run+1})"):
            answer, generation_time = ft_evaluator.generate(query)
            answer_length = len(answer)
            
            metrics = {
                "query": query,
                "generation_time": generation_time,
                "total_time": generation_time,
                "answer_length": answer_length,
                "run": run + 1,
                "answer": answer
            }
            
            run_metrics.append(metrics)
        
        all_metrics.extend(run_metrics)
    
    df = pd.DataFrame(all_metrics)
    
    agg_df = df.groupby('query').agg({
        'generation_time': ['mean', 'std'],
        'total_time': ['mean', 'std'],
        'answer_length': ['mean', 'std']
    }).reset_index()
    
    overall_metrics = {
        "model_type": "fine-tuned",
        "avg_generation_time": df['generation_time'].mean(),
        "std_generation_time": df['generation_time'].std(),
        "avg_total_time": df['total_time'].mean(),
        "std_total_time": df['total_time'].std(),
        "avg_answer_length": df['answer_length'].mean(),
        "std_answer_length": df['answer_length'].std(),
    }
    
    return {
        "detailed_metrics": df.to_dict('records'),
        "overall_metrics": overall_metrics
    }


def compare_models(documents, queries, models_to_eval, index_types, ft_model_path, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    comparison_results = []
    
    for model_name in models_to_eval:
        for index_type in index_types:
            print(f"\n=== Evaluating RAG with {model_name} and {index_type} index ===\n")
            openai_api_key = os.environ.get("OPENAI_API_KEY", None)
            is_openai_model = model_name.startswith(("text-embedding", "ada"))
            
            if is_openai_model and not openai_api_key:
                print(f"Skipping {model_name} as OpenAI API key is not provided")
                continue
            
            rag = RAGSystem(
                embedding_model_name=model_name,
                index_type=index_type,
                openai_api_key=openai_api_key,
                cache_dir="./cache"
            )
            
            rag.index_documents(documents)
            results = evaluate_rag_system(rag, queries, model_name, index_type)
            
            results["overall_metrics"]["model_type"] = "rag"
            
            comparison_results.append(results["overall_metrics"])
            model_short_name = model_name.replace("/", "_")
            detailed_results_path = os.path.join(output_dir, f"{model_short_name}_{index_type}_detailed.json")
            with open(detailed_results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved detailed results to {detailed_results_path}")
    
    print(f"\n=== Evaluating fine-tuned model from {ft_model_path} ===\n")
    try:
        ft_evaluator = FineTunedModelEvaluator(ft_model_path)
        ft_results = evaluate_fine_tuned_model(ft_evaluator, queries)
        comparison_results.append(ft_results["overall_metrics"])
        
        ft_detailed_results_path = os.path.join(output_dir, "fine_tuned_detailed.json")
        with open(ft_detailed_results_path, 'w') as f:
            json.dump(ft_results, f, indent=2)
        print(f"Saved detailed results to {ft_detailed_results_path}")
        
    except Exception as e:
        print(f"Error evaluating fine-tuned model: {str(e)}")
    
    if comparison_results:
        comp_df = pd.DataFrame(comparison_results)
        comparison_path = os.path.join(output_dir, "model_comparison.csv")
        comp_df.to_csv(comparison_path, index=False)
        
        create_comparison_visualizations(comp_df, output_dir)
        
        print(f"\nComparison results saved to {comparison_path}")
        return comp_df
    else:
        print("No successful evaluations to compare")
        return None


def create_comparison_visualizations(comp_df, output_dir):
    rag_df = comp_df[comp_df['model_type'] == 'rag']
    ft_df = comp_df[comp_df['model_type'] == 'fine-tuned']
    
    has_rag = not rag_df.empty
    has_ft = not ft_df.empty
    
    if has_rag:
        models = rag_df['embedding_model'].unique()
        index_types = rag_df['index_type'].unique()

        plt.figure(figsize=(14, 8))
        for i, model in enumerate(models):
            model_data = rag_df[rag_df['embedding_model'] == model]
            pos = np.arange(len(index_types)) + i * 0.2
            plt.bar(pos, model_data['avg_retrieval_time'], width=0.2, 
                    yerr=model_data['std_retrieval_time'], capsize=5,
                    label=model)
            
        plt.xlabel('Index Type')
        plt.ylabel('Retrieval Time (s)')
        plt.title('Average Retrieval Time by Model and Index Type')
        plt.xticks(np.arange(len(index_types)) + 0.2, index_types)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'retrieval_time_comparison.png'))
        
        plt.figure(figsize=(14, 8))
        for i, model in enumerate(models):
            model_data = rag_df[rag_df['embedding_model'] == model]
            pos = np.arange(len(index_types)) + i * 0.2
            plt.bar(pos, model_data['avg_generation_time'], width=0.2, 
                    yerr=model_data['std_generation_time'], capsize=5,
                    label=model)
            
        plt.xlabel('Index Type')
        plt.ylabel('Generation Time (s)')
        plt.title('Average Generation Time by Model and Index Type')
        plt.xticks(np.arange(len(index_types)) + 0.2, index_types)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'generation_time_comparison.png'))
        
        plt.figure(figsize=(14, 8))
        for i, model in enumerate(models):
            model_data = rag_df[rag_df['embedding_model'] == model]
            pos = np.arange(len(index_types)) + i * 0.2
            plt.bar(pos, model_data['avg_total_time'], width=0.2, 
                    yerr=model_data['std_total_time'], capsize=5,
                    label=model)
            
        plt.xlabel('Index Type')
        plt.ylabel('Total Time (s)')
        plt.title('Average Total Inference Time by Model and Index Type')
        plt.xticks(np.arange(len(index_types)) + 0.2, index_types)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'total_time_comparison.png'))
    
    if has_rag and has_ft:
        best_rag = rag_df.loc[rag_df['avg_total_time'].idxmin()]
        best_rag_name = f"{best_rag['embedding_model']} + {best_rag['index_type']}"
        
        ft_result = ft_df.iloc[0]
        
        plt.figure(figsize=(10, 6))
        models = [best_rag_name, "Fine-tuned Model"]
        times = [best_rag['avg_total_time'], ft_result['avg_total_time']]
        std_devs = [best_rag['std_total_time'], ft_result['std_total_time']]
        
        plt.bar(models, times, yerr=std_devs, capsize=5)
        plt.ylabel('Total Time (s)')
        plt.title('Inference Time: Best RAG vs. Fine-tuned Model')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rag_vs_finetuned_time.png'))
        
        plt.figure(figsize=(10, 6))
        lengths = [best_rag['avg_answer_length'], ft_result['avg_answer_length']]
        
        plt.bar(models, lengths)
        plt.ylabel('Answer Length (characters)')
        plt.title('Answer Length: Best RAG vs. Fine-tuned Model')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rag_vs_finetuned_length.png'))
        
        plt.figure(figsize=(12, 6))
        labels = ['Retrieval', 'Generation', 'Fine-tuned Total']
        times = [best_rag['avg_retrieval_time'], best_rag['avg_generation_time'], ft_result['avg_total_time']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        plt.bar(labels, times, color=colors)
        plt.axhline(y=best_rag['avg_total_time'], color='r', linestyle='--', 
                   label=f'RAG Total: {best_rag["avg_total_time"]:.3f}s')
        plt.ylabel('Time (s)')
        plt.title('Time Breakdown: RAG Components vs. Fine-tuned Model')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_breakdown_comparison.png'))


def analyze_answers(output_dir):
    all_files = glob.glob(os.path.join(output_dir, "*_detailed.json"))
    
    rag_answers = {}
    ft_answers = {}
    
    for file_path in all_files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {file_path}. Skipping this file.")
            continue
        
        is_ft = "fine_tuned" in os.path.basename(file_path)
        detailed_metrics = results.get("detailed_metrics", [])
        
        for metric in detailed_metrics:
            query = metric.get("query")
            answer = metric.get("answer")
            
            if not query or not answer:
                continue
                
            if metric.get("run", 1) != 1:
                continue
                
            if is_ft:
                ft_answers[query] = answer
            else:
                embedding_model = results.get("overall_metrics", {}).get("embedding_model", "unknown")
                index_type = results.get("overall_metrics", {}).get("index_type", "unknown")
                key = f"{embedding_model}_{index_type}"
                
                if key not in rag_answers:
                    rag_answers[key] = {}
                    
                rag_answers[key][query] = answer
    
    if ft_answers and rag_answers:
        comparison_rows = []
        
        for query in ft_answers.keys():
            row = {"query": query, "fine_tuned_answer": ft_answers[query]}
            
            for config, answers in rag_answers.items():
                if query in answers:
                    row[f"rag_{config}_answer"] = answers[query]
            
            comparison_rows.append(row)
        
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(os.path.join(output_dir, "answer_comparison.csv"), index=False)
        
        print(f"Saved answer comparison to {os.path.join(output_dir, 'answer_comparison.csv')}")
    else:
        print("No answers found to compare.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and Compare RAG Systems with Fine-tuned Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing documents")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["all-MiniLM-L6-v2", "BAAI/bge-large-en"], 
                        help="Embedding models to evaluate")
    parser.add_argument("--index_types", type=str, nargs="+", 
                        default=["flat", "hnsw"], 
                        help="Index types to evaluate")
    parser.add_argument("--ft_model_path", type=str, 
                        default="3b-3.2_bs41_4bit_shuffling",
                        help="Path to fine-tuned model")
    parser.add_argument("--num_queries", type=int, default=10,
                        help="Number of standard queries to generate")
    parser.add_argument("--openai_key_file", type=str, 
                        help="File containing OpenAI API key (for OpenAI models)")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs for each evaluation for statistical significance")
    args = parser.parse_args()
    
    if args.openai_key_file and os.path.exists(args.openai_key_file):
        with open(args.openai_key_file, 'r') as f:
            api_key = f.read().strip()
            os.environ["OPENAI_API_KEY"] = api_key
    
    documents = []
    for ext in ["*.txt", "*.md", "*.csv"]:
        for file_path in glob.glob(os.path.join(args.data_dir, ext)):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(content)
    
    print(f"Loaded {len(documents)} documents")
    queries = generate_standard_queries(documents, num_queries=args.num_queries)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "standard_queries.json"), "w") as f:
        json.dump(queries, f, indent=2)
    
    print(f"Generated {len(queries)} standard queries:")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")
    
    comparison_results = compare_models(
        documents=documents,
        queries=queries,
        models_to_eval=args.models,
        index_types=args.index_types,
        ft_model_path=args.ft_model_path,
        output_dir=args.output_dir
    )
    
    analyze_answers(args.output_dir)
    
    if comparison_results is not None:
        print("\nSummary of Results:")
        
        rag_results = comparison_results[comparison_results['model_type'] == 'rag']
        if not rag_results.empty:
            rag_sorted = rag_results.sort_values(by='avg_total_time')
            print("\nRAG Systems:")
            print(rag_sorted[['embedding_model', 'index_type', 
                             'avg_retrieval_time', 'std_retrieval_time',
                             'avg_generation_time', 'std_generation_time',
                             'avg_total_time', 'std_total_time']])
        
        ft_results = comparison_results[comparison_results['model_type'] == 'fine-tuned']
        if not ft_results.empty:
            print("\nFine-tuned Model:")
            print(ft_results[['avg_generation_time', 'std_generation_time',
                             'avg_total_time', 'std_total_time']])
            
            if not rag_results.empty:
                best_rag = rag_sorted.iloc[0]
                print("\nComparison - Best RAG vs Fine-tuned:")
                speed_ratio = best_rag['avg_total_time'] / ft_results['avg_total_time'].values[0]
                print(f"Speed Ratio (RAG/Fine-tuned): {speed_ratio:.2f}x")
                if speed_ratio > 1:
                    print(f"Fine-tuned model is {speed_ratio:.2f}x faster than the best RAG system")
                else:
                    print(f"Best RAG system is {1/speed_ratio:.2f}x faster than the fine-tuned model")


if __name__ == "__main__":
    main()