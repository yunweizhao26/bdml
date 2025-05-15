import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_path: str = "./models/Llama3.2-3B",
        index_type: str = "flat",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        openai_api_key: Optional[str] = None,
        cache_dir: str = "./cache",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = cache_dir
        self.embedding_model_name = embedding_model_name
        
        os.makedirs(cache_dir, exist_ok=True)
        
        self.openai_api_key = openai_api_key
        if openai_api_key:
            import openai
            openai.api_key = openai_api_key
        
        self.is_openai_model = embedding_model_name.startswith(("text-embedding", "ada"))
        
        logger.info(f"Loading embedding model: {embedding_model_name}")
        
        if not self.is_openai_model:
            if embedding_model_name == "BAAI/bge-large-en" or embedding_model_name.startswith("BAAI/bge-"):
                self.is_bge_model = True
                logger.info("Using BGE model with instruction prepending")
            else:
                self.is_bge_model = False
                
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()
        else:
            embedding_dimensions = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            
            self.embedding_size = embedding_dimensions.get(embedding_model_name, 1536)
            self.is_bge_model = False
            
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embedding models")
            
            try:
                self.get_embedding("Test", batch=False)
                logger.info(f"Successfully connected to OpenAI API with model {embedding_model_name}")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to OpenAI API: {str(e)}")
        
        logger.info(f"Loading LLM from: {llm_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        
        self.index_type = index_type
        self.index = None
        self.documents = []
        self.document_chunks = []
        self.init_index(index_type)
    
    def prepare_bge_text(self, text_or_texts, for_query=False):
        if for_query:
            instruction = "Represent this question for retrieving supporting documents: "
        else:
            instruction = "Represent this document for retrieval: "
            
        if isinstance(text_or_texts, list):
            return [instruction + text for text in text_or_texts]
        else:
            return instruction + text_or_texts
    
    def get_embedding(self, text_or_texts, batch=True, for_query=False):
        if not self.is_openai_model:
            if self.is_bge_model:
                text_or_texts = self.prepare_bge_text(text_or_texts, for_query=for_query)
            
            if batch:
                return self.embedding_model.encode(text_or_texts, convert_to_numpy=True)
            else:
                return self.embedding_model.encode([text_or_texts], convert_to_numpy=True)[0]
        else:
            import openai
            if not batch:
                response = openai.Embedding.create(
                    input=[text_or_texts],
                    model=self.embedding_model_name
                )
                return np.array(response["data"][0]["embedding"])
            else:
                if isinstance(text_or_texts, list):
                    batch_size = 1000
                    embeddings = []
                    
                    for i in range(0, len(text_or_texts), batch_size):
                        batch = text_or_texts[i:i+batch_size]
                        logger.info(f"Processing batch {i//batch_size + 1}/{(len(text_or_texts)-1)//batch_size + 1}")
                        
                        response = openai.Embedding.create(
                            input=batch,
                            model=self.embedding_model_name
                        )
                        
                        batch_embeddings = [np.array(item["embedding"]) for item in response["data"]]
                        embeddings.extend(batch_embeddings)
                    
                    return np.array(embeddings)
                else:
                    response = openai.Embedding.create(
                        input=[text_or_texts],
                        model=self.embedding_model_name
                    )
                    return np.array([response["data"][0]["embedding"]])
    
    def init_index(self, index_type: str):
        d = self.embedding_size
        
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(d)
        
        elif index_type == "pq":
            M = 8
            nbits = 8
            self.index = faiss.IndexPQ(d, M, nbits)
            self.needs_training = True
        
        elif index_type == "ivf":
            nlist = 100
            quantizer = faiss.IndexFlatL2(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
            self.needs_training = True
        
        elif index_type == "vq":
            nlist = 256
            quantizer = faiss.IndexFlatL2(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
            self.index.nprobe = 1  
            self.needs_training = True
        
        elif index_type == "hnsw":
            M = 32
            self.index = faiss.IndexHNSWFlat(d, M)
            self.index.hnsw.efConstruction = 128
            self.index.hnsw.efSearch = 64
        
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        logger.info(f"Initialized {index_type} index with dimension {d}")

    
    def get_cache_path(self, prefix=""):
        model_name = self.embedding_model_name.replace("/", "_")
        base_name = f"{prefix}{model_name}_{self.index_type}_{self.chunk_size}_{self.chunk_overlap}"
        
        chunks_path = os.path.join(self.cache_dir, f"{base_name}_chunks.pkl")
        documents_path = os.path.join(self.cache_dir, f"{base_name}_documents.pkl")
        index_path = os.path.join(self.cache_dir, f"{base_name}_index.faiss")
        
        return chunks_path, documents_path, index_path
    
    def save_to_cache(self):
        chunks_path, documents_path, index_path = self.get_cache_path()
        
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.document_chunks, f)
        
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        faiss.write_index(self.index, index_path)
        
        logger.info(f"Saved index and data to cache: {self.cache_dir}")
        
    def load_from_cache(self):
        chunks_path, documents_path, index_path = self.get_cache_path()
        
        if not (os.path.exists(chunks_path) and 
                os.path.exists(documents_path) and 
                os.path.exists(index_path)):
            logger.info("Cache not found or incomplete. Will need to process documents.")
            return False
        
        try:
            with open(chunks_path, 'rb') as f:
                self.document_chunks = pickle.load(f)
            
            with open(documents_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            self.index = faiss.read_index(index_path)
            
            logger.info(f"Successfully loaded index and data from cache ({len(self.document_chunks)} chunks).")
            return True
        except Exception as e:
            logger.error(f"Error loading from cache: {str(e)}")
            return False
    
    def chunk_documents(self, documents: List[str]) -> List[str]:
        chunks = []
        for doc in documents:
            if len(doc) <= self.chunk_size:
                chunks.append(doc)
                continue
                
            for i in range(0, len(doc), self.chunk_size - self.chunk_overlap):
                chunk = doc[i:i + self.chunk_size]
                if len(chunk) >= 50:
                    chunks.append(chunk)
        
        return chunks
    
    def index_documents(self, documents: List[str], force_reindex=False):
        start_time = time.time()
        
        if not force_reindex and self.load_from_cache():
            return
        
        self.documents = documents
        self.document_chunks = self.chunk_documents(documents)
        
        logger.info(f"Creating embeddings for {len(self.document_chunks)} chunks")
        
        if len(self.document_chunks) > 10000:
            batch_size = 10000
            all_embeddings = []
            
            for i in range(0, len(self.document_chunks), batch_size):
                end_idx = min(i + batch_size, len(self.document_chunks))
                logger.info(f"Processing chunks {i} to {end_idx}")
                
                batch = self.document_chunks[i:end_idx]
                batch_embeddings = self.get_embedding(batch, batch=True, for_query=False)
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
        else:
            embeddings = self.get_embedding(self.document_chunks, batch=True, for_query=False)
        
        if hasattr(self, 'needs_training') and self.needs_training:
            logger.info(f"Training the {self.index_type} index")
            self.index.train(embeddings)
        
        logger.info(f"Adding {len(embeddings)} vectors to index")
        self.index.add(embeddings)
        
        self.save_to_cache()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Indexing completed in {elapsed_time:.2f} seconds")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        start_time = time.time()
        
        query_vector = self.get_embedding(query, batch=False, for_query=True)
        query_vector = query_vector.reshape(1, -1)
        
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:
                results.append((self.document_chunks[idx], float(distance)))
        
        elapsed_time = time.time() - start_time
        logger.info(f"Retrieval completed in {elapsed_time:.4f} seconds")
        
        return results
    
    def generate(self, query: str, context: str, max_new_tokens: int = 256) -> str:
        start_time = time.time()
        prompt = f"""You are answering questions based SOLELY on the retrieved context below.
            If the context doesn't contain enough information to answer the question directly, say so.
            Do not fabricate information or expand beyond what's explicitly mentioned in the context.

            Context:
            {context}

            Question: {query}

            Answer:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "\nAnswer:" in response:
            response = response.split("\nAnswer:")[1].strip()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generation completed in {elapsed_time:.2f} seconds")
        
        return response
    
    def query(self, query: str, top_k: int = 5) -> Dict:
        retrieved_results = self.retrieve(query, top_k)
        
        context = "\n\n".join([chunk for chunk, _ in retrieved_results])
        
        answer = self.generate(query, context)
        
        return {
            "query": query,
            "retrieved_chunks": retrieved_results,
            "answer": answer
        }
    
    def benchmark(self, queries: List[str], top_k: int = 5) -> Dict:
        results = []
        
        total_retrieval_time = 0
        total_generation_time = 0
        
        for query in tqdm(queries, desc="Benchmarking"):
            retrieval_start = time.time()
            retrieved_results = self.retrieve(query, top_k)
            retrieval_time = time.time() - retrieval_start
            
            context = "\n\n".join([chunk for chunk, _ in retrieved_results])
            
            generation_start = time.time()
            answer = self.generate(query, context)
            generation_time = time.time() - generation_start
            
            total_time = retrieval_time + generation_time
            
            results.append({
                "query": query,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "answer": answer
            })
            
            total_retrieval_time += retrieval_time
            total_generation_time += generation_time
        
        avg_retrieval_time = total_retrieval_time / len(queries)
        avg_generation_time = total_generation_time / len(queries)
        avg_total_time = (total_retrieval_time + total_generation_time) / len(queries)
        
        return {
            "individual_results": results,
            "summary": {
                "avg_retrieval_time": avg_retrieval_time,
                "avg_generation_time": avg_generation_time,
                "avg_total_time": avg_total_time,
                "num_queries": len(queries)
            }
        }
    
    def compare_index_types(self, documents: List[str], queries: List[str], index_types: List[str]):
        results = {}
        
        for index_type in index_types:
            logger.info(f"Testing index type: {index_type}")
            
            self.init_index(index_type)
            
            self.index_documents(documents)
            
            benchmark = self.benchmark(queries)
            results[index_type] = benchmark["summary"]
        
        df = pd.DataFrame(results).T
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df[["avg_retrieval_time", "avg_generation_time"]].plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Time (seconds)")
        ax.set_title("RAG Performance by Index Type")
        plt.tight_layout()
        plt.savefig("rag_index_comparison.png")
        
        return df, fig