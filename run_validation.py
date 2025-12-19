import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Importa a camada customizada
from src.models.heads import CosFace

def load_and_preprocess_image(path, image_size=(112, 112)):
    """Carrega e normaliza a imagem conforme o padrão ResNet50."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
        
    img = Image.open(path).convert('RGB')
    img = img.resize(image_size)
    img_array = np.array(img).astype(np.float32)
    # Pré-processamento crucial
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def extract_embeddings(extractor_model, img_path):
    """Extrai embeddings usando o modelo extrator já carregado."""
    img = load_and_preprocess_image(img_path)
    embedding = extractor_model.predict(img, verbose=0)
    return tf.nn.l2_normalize(embedding, axis=1).numpy()

def compute_metrics(y_true, similarities, threshold=0.35):
    fpr, tpr, thresholds = roc_curve(y_true, similarities)
    roc_auc = auc(fpr, tpr)
    
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    
    metrics = {
        'auc': roc_auc,
        'eer': eer,
        'accuracy': accuracy_score(y_true, similarities > threshold),
        'precision': precision_score(y_true, similarities > threshold, zero_division=0),
        'recall': recall_score(y_true, similarities > threshold, zero_division=0),
        'f1': f1_score(y_true, similarities > threshold, zero_division=0)
    }
    
    for far_target in [0.001, 0.01]:
        idx = np.where(fpr <= far_target)[0]
        metrics[f'TAR@FAR={far_target}'] = tpr[idx[-1]] if len(idx) > 0 else 0
        
    return metrics, fpr, tpr

def run_evaluation(model_path, lfw_root, pairs_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Carregando modelo completo: {model_path}")
    full_model = tf.keras.models.load_model(model_path, custom_objects={'CosFace': CosFace})
    
    # --- CRIAÇÃO DO EXTRATOR BASEADO NO SUMMARY ---
    print("Criando modelo extrator...")
    try:
        # Pelo seu summary, a camada 'resnet50_backbone' já cospe (None, 512)
        # Então ela é o nosso feature extractor!
        target_layer = full_model.get_layer("resnet50_backbone")
        
        extractor_model = tf.keras.Model(
            inputs=full_model.input[0], # A entrada de imagem do modelo principal
            outputs=target_layer.output # A saída de 512 dimensões
        )
        print(f"Extrator criado com sucesso. Output shape: {extractor_model.output_shape}")
        
    except Exception as e:
        print(f"ERRO CRÍTICO AO CRIAR EXTRATOR: {e}")
        print("Tentando estratégia alternativa (busca por output shape)...")
        # Fallback: Procura qualquer camada que saia (None, 512)
        found = False
        for layer in full_model.layers:
            if layer.output.shape[-1] == 512:
                print(f"Camada alternativa encontrada: {layer.name}")
                extractor_model = tf.keras.Model(inputs=full_model.input[0], outputs=layer.output)
                found = True
                break
        if not found:
            print("Não foi possível encontrar a camada de embedding.")
            return

    y_true = []
    similarities = []
    
    with open(pairs_path, 'r') as f:
        pairs = f.readlines()[1:]
        
    print(f"Processando {len(pairs)} pares...")
    
    success_count = 0
    error_count = 0
    
    for i, line in enumerate(pairs):
        if i % 50 == 0: print(f"Processado: {i}/{len(pairs)}", end='\r')
            
        p = line.strip().split()
        try:
            if len(p) == 3: # Positivo
                path1 = os.path.join(lfw_root, p[0], f"{p[0]}_{int(p[1]):04d}.jpg")
                path2 = os.path.join(lfw_root, p[0], f"{p[0]}_{int(p[2]):04d}.jpg")
                label = 1
            elif len(p) == 4: # Negativo
                path1 = os.path.join(lfw_root, p[0], f"{p[0]}_{int(p[1]):04d}.jpg")
                path2 = os.path.join(lfw_root, p[2], f"{p[2]}_{int(p[3]):04d}.jpg")
                label = 0
            else: continue

            emb1 = extract_embeddings(extractor_model, path1)
            emb2 = extract_embeddings(extractor_model, path2)
            
            sim = np.dot(emb1, emb2.T)[0][0]
            similarities.append(sim)
            y_true.append(label)
            success_count += 1
            
        except Exception as e:
            error_count += 1
            if error_count == 1: 
                print(f"\nERRO (Exemplo): {e}")
            continue

    print(f"\nConcluído. Sucessos: {success_count}")

    if not similarities:
        print("Nenhum resultado gerado.")
        return

    metrics, fpr, tpr = compute_metrics(np.array(y_true), np.array(similarities))
    
    # Plot ROC
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, np.array(similarities) > 0.35)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    print("\n--- RESULTADOS FINAIS ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--lfw_root', type=str, required=True)
    parser.add_argument('--pairs', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='eval_results')
    args = parser.parse_args()
    
    run_evaluation(args.model, args.lfw_root, args.pairs, args.save_dir)