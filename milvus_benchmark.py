import time
import psutil
import json
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# --- CONFIGURAÇÕES ---
MILVUS_HOST = "172.17.0.1" 
MILVUS_PORT = "19540"
COLLECTION_NAME = "lfw_cosface_tta"

# Caminhos gerados pelo seu script de inferência
EMBEDDINGS_PATH = "assets/lfw_resnet_embeddings.npy"
NAMES_PATH = "assets/lfw_resnet_names.json"

def run_milvus_task():
    # 1. CONEXÃO
    print(f"-> Conectando ao Milvus em {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("✓ Conectado!")
    except Exception as e:
        print(f"✗ Erro ao conectar: {e}")
        return

    # 2. CARREGAR DADOS
    print("-> Carregando embeddings e nomes...")
    embeddings = np.load(EMBEDDINGS_PATH).astype('float32')
    with open(NAMES_PATH, 'r') as f:
        labels = json.load(f)
    
    num_vectors = len(embeddings)
    dim = embeddings.shape[1] 
    print(f"✓ {num_vectors} vetores de dimensão {dim} carregados.")

    # 3. SCHEMA E LIMPEZA (Correção do erro anterior aqui)
    if utility.has_collection(COLLECTION_NAME):
        print(f"-> Removendo coleção existente: {COLLECTION_NAME}")
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, "Benchmark LFW CosFace TTA")
    collection = Collection(COLLECTION_NAME, schema)
    print(f"✓ Coleção {COLLECTION_NAME} criada.")

    # 4. MÉTODOS DE INSERÇÃO (Métricas de tempo)
    print("\n>>> COMPARANDO MÉTODOS DE INGESTÃO (TASK 3)")
    
    # Teste A: Row-by-row (100 amostras para medir latência)
    print("MÉTODO A: Inserindo 100 vetores individualmente (Síncrono)...")
    t_start = time.time()
    for i in range(100):
        # Milvus espera uma lista de listas para cada campo
        collection.insert([[labels[i]], [embeddings[i].tolist()]])
    
    # É necessário dar flush para garantir que os dados saíram da memória do cliente
    collection.flush() 
    print(f"MÉTODO A: Tempo para 100 vetores: {time.time() - t_start:.4f}s")

    # Teste B: Bulk (O restante do dataset de uma vez)
    print(f"\nMÉTODO B: Inserindo restante ({num_vectors-100}) em lote (Batch Size 1000)...")
    batch_size = 1000
    t_start = time.time()
    for i in range(100, num_vectors, batch_size):
        end = min(i + batch_size, num_vectors)
        batch_labels = labels[i:end]
        batch_vectors = embeddings[i:end].tolist()
        collection.insert([batch_labels, batch_vectors])
    
    collection.flush()
    print(f"MÉTODO B: Tempo para o lote: {time.time() - t_start:.4f}s")

    # 5. ANÁLISE DE RECURSOS (Indexação HNSW - TASK 4)
    print("\n>>> ANALISANDO INDEXAÇÃO (RECURSOS DGX)")
    process = psutil.Process()
    
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 256}
    }

    mem_before = process.memory_info().rss / 1024 / 1024
    t_start = time.time()
    
    # Aqui o Milvus realmente usa a CPU da DGX para construir o grafo de busca
    collection.create_index(field_name="embedding", index_params=index_params)
    
    t_end = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024
    cpu_usage = psutil.cpu_percent(interval=1) # Mede a CPU durante 1 segundo

    print("-" * 40)
    print("RESULTADOS FINAIS PARA O RELATÓRIO:")
    print(f"1. Tempo de Indexação: {t_end - t_start:.2f} segundos")
    print(f"2. Consumo de RAM Adicional (Script): {mem_after - mem_before:.2f} MB")
    print(f"3. Carga de CPU na DGX: {cpu_usage}%")
    print(f"4. Total de vetores no Milvus: {collection.num_entities}")
    print("-" * 40)
    print("Task cumprida. O banco está pronto para buscas de alta performance.")

if __name__ == "__main__":
    run_milvus_task()
