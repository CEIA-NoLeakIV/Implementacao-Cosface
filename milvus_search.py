from pymilvus import connections, Collection
import time
import numpy as np

# Conexão
connections.connect("default", host="172.17.0.1", port="19540")
collection = Collection("lfw_cosface_tta")
collection.load() # OBRIGATÓRIO: Carregar a coleção na RAM para busca

def benchmark_search(num_queries=10):
    # Carregando os mesmos embeddings para usar alguns como 'alvo' de teste
    embeddings = np.load("assets/lfw_resnet_embeddings.npy")
    
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    
    print(f"--- Iniciando Benchmark de Busca ({num_queries} queries) ---")
    start_time = time.time()
    
    # Simula 10 buscas de rostos aleatórios
    results = collection.search(
        data=embeddings[:num_queries].tolist(), 
        anns_field="embedding", 
        param=search_params, 
        limit=5, # Top 5 resultados
        output_fields=["label"]
    )
    
    end_time = time.time()
    avg_latency = (end_time - start_time) / num_queries
    
    print(f"Tempo total para {num_queries} buscas: {end_time - start_time:.4f}s")
    print(f"Latência média por busca: {avg_latency * 1000:.2f} ms")
    
    # Exemplo de um resultado
    print(f"\nExemplo do Top 1 encontrado para a primeira busca: {results[0][0].entity.get('label')}")

if __name__ == "__main__":
    benchmark_search(100) # Testando com 100 buscas para ter uma média sólida
