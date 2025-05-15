import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from rag_system import RAGSystem

def evaluate_rag_system(test_data_path, rag_system):
    """Evaluate the RAG system on a test dataset."""
    # Load test data (queries and ground truth answers)
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    results = []
    
    for item in test_data:
        query = item['query']
        ground_truth = item['ground_truth']
        
        # Run query through RAG system
        response = rag_system.query(query)
        
        # Metrics to track
        result = {
            'query': query,
            'ground_truth': ground_truth,
            'predicted': response['answer'],
            'retrieval_time': None,
            'generation_time': None,
            'total_time': None,
        }
        
        results.append(result)
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Calculate average times
    avg_times = {
        'avg_retrieval_time': df['retrieval_time'].mean(),
        'avg_generation_time': df['generation_time'].mean(),
        'avg_total_time': df['total_time'].mean(),
    }
    
    return df, avg_times

def compare_fine_tuned_vs_rag(test_data_path, fine_tuned_model_path, rag_system):
    """Compare performance of fine-tuned LLM vs RAG system."""
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Initialize fine-tuned model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
    model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)
    
    results = []
    
    for item in test_data:
        query = item['query']
        ground_truth = item['ground_truth']
        
        # Get RAG response
        rag_start_time = time.time()
        rag_response = rag_system.query(query)
        rag_time = time.time() - rag_start_time
        
        # Get fine-tuned model response
        ft_start_time = time.time()
        inputs = tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_new_tokens=256)
        ft_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ft_time = time.time() - ft_start_time
        
        result = {
            'query': query,
            'ground_truth': ground_truth,
            'rag_response': rag_response['answer'],
            'ft_response': ft_response,
            'rag_time': rag_time,
            'ft_time': ft_time,
        }
        
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate average times
    avg_times = {
        'avg_rag_time': df['rag_time'].mean(),
        'avg_ft_time': df['ft_time'].mean(),
        'speedup_factor': df['ft_time'].mean() / df['rag_time'].mean() if df['rag_time'].mean() > 0 else float('inf')
    }
    
    # Plot time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(['Fine-tuned LLM', 'RAG System'], 
            [avg_times['avg_ft_time'], avg_times['avg_rag_time']])
    plt.ylabel('Average Inference Time (seconds)')
    plt.title('Inference Time Comparison')
    plt.savefig('time_comparison.png')
    
    return df, avg_times

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG System")
    parser.add_argument("--test_data", type=str, required=True, 
                      help="Path to test data JSON file")
    parser.add_argument("--data_dir", type=str, required=True, 
                      help="Directory containing documents for RAG")
    parser.add_argument("--llm_path", type=str, default="./models/Llama3.2-3B", 
                      help="Path to LLaMA model")
    parser.add_argument("--fine_tuned_model", type=str, 
                      help="Path to fine-tuned model for comparison")
    parser.add_argument("--index_type", type=str, default="flat", 
                      choices=["flat", "pq", "ivfpq", "hnsw"], 
                      help="Type of vector index to use")
    args = parser.parse_args()
    
    # Load documents
    documents = []
    for file_path in glob.glob(os.path.join(args.data_dir, "*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(content)
    
    # Initialize RAG system
    rag = RAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model_path=args.llm_path,
        index_type=args.index_type
    )
    
    # Index documents
    rag.index_documents(documents)
    
    # Standard evaluation
    df, avg_times = evaluate_rag_system(args.test_data, rag)
    
    print("\nEvaluation Results:")
    print(f"Average Retrieval Time: {avg_times['avg_retrieval_time']:.4f} seconds")
    print(f"Average Generation Time: {avg_times['avg_generation_time']:.4f} seconds")
    print(f"Average Total Time: {avg_times['avg_total_time']:.4f} seconds")
    
    # Compare with fine-tuned model if provided
    if args.fine_tuned_model:
        comp_df, comp_times = compare_fine_tuned_vs_rag(
            args.test_data, args.fine_tuned_model, rag
        )
        
        print("\nComparison with Fine-tuned Model:")
        print(f"Average RAG Time: {comp_times['avg_rag_time']:.4f} seconds")
        print(f"Average Fine-tuned Time: {comp_times['avg_ft_time']:.4f} seconds")
        print(f"Speedup Factor: {comp_times['speedup_factor']:.2f}x")
        print("\nTime comparison chart saved as 'time_comparison.png'")

if __name__ == "__main__":
    main()