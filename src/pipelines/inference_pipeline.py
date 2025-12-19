# Local: src/pipelines/inference_pipeline.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.datasets import mnist

from face_embeddings.src.models.Cosface.src.models.heads import CosFace

def run_cosface_inference(cosface_config, inference_config):
    """
    Executa a inferência apenas no modelo CosFace para extrair e visualizar embeddings.
    """
    print("--- Iniciando Pipeline de Inferência do Modelo CosFace ---") 
    
    # 1. Carregar Dados
    (_, _), (X_test, y_test) = mnist.load_data()
    sample_size = inference_config.SAMPLE_SIZE
    X_test_keras = X_test[:sample_size, :, :, np.newaxis].astype('float32') / 255
    y_test_sampled = y_test[:sample_size]

    # 2. Processar Features do CosFace
    print("Processando modelo CosFace...")
    cosface_model = load_model(cosface_config.CHECKPOINT_PATH, custom_objects={'CosFace': CosFace})
    feature_extractor = Model(inputs=cosface_model.input[0], outputs=cosface_model.layers[-3].output)
    cosface_features = feature_extractor.predict(X_test_keras, verbose=0)
    cosface_features /= np.linalg.norm(cosface_features, axis=1, keepdims=True)

    # 3. Salvar Embeddings em JSON (Como você se lembrava)
    print("Salvando amostra de embeddings em formato JSON...")
    embeddings_data = [
        {'id': i, 'label': int(y_test_sampled[i]), 'embedding': cosface_features[i].tolist()} 
        for i in range(sample_size)
    ]
    json_output_path = 'deliverables/cosface_embeddings.json'
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    with open(json_output_path, 'w') as f:
        json.dump(embeddings_data, f, indent=4)
    print(f"Embeddings salvos em: {json_output_path}")

    # 4. Gerar Gráfico de Visualização 3D
    print("Gerando visualização 3D...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    for i in range(10):
        idx = y_test_sampled == i
        ax.scatter(cosface_features[idx, 0], cosface_features[idx, 1], cosface_features[idx, 2], label=f'Dígito {i}', alpha=0.4)
    ax.set_title('Visualização 3D dos Embeddings do CosFace')
    ax.legend()
    
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig(inference_config.OUTPUT_FIGURE_PATH)
    print(f"\n--- Gráfico salvo em: {inference_config.OUTPUT_FIGURE_PATH} ---")