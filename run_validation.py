import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model, load_model

# Adicionar o diretório atual ao path para importar módulos locais
sys.path.append(os.getcwd())

from src.losses.margin_losses import CosFace

# --- Configuração de Logging ---
def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'validation_metrics.log')
    
    # Configurar logger
    logger = logging.getLogger('ValidationLogger')
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Adicionar handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# --- Funções Auxiliares ---

def preprocess_image(image_path, image_size):
    '''Carrega e pré-processa uma imagem individualmente.'''
    if not os.path.exists(image_path):
        return None
    
    # Carregar imagem
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    
    # Resize
    img = tf.image.resize(img, (image_size, image_size))
    
    # Pré-processamento específico do ResNet50 (caffe style: BGR, zero-centered)
    # Importante: Usar o mesmo preprocessamento do treino!
    img = tf.keras.applications.resnet50.preprocess_input(img)
    
    return img

def extract_deep_features_tta(feature_extractor, image_path, image_size):
    '''
    Extrai features profundas usando TTA (Test Time Augmentation).
    Combina a imagem original e a versão flipada horizontalmente.
    Adaptado do código de referência PyTorch.
    '''
    img = preprocess_image(image_path, image_size)
    if img is None:
        return None

    # Criar versão flipada
    img_flipped = tf.image.flip_left_right(img)
    
    # Batch com as duas imagens: [Original, Flipped]
    batch = tf.stack([img, img_flipped])
    
    # Inferência
    features = feature_extractor.predict(batch, verbose=0)
    
    # features shape: (2, embedding_dim)
    original_feat = features[0]
    flipped_feat = features[1]
    
    # Concatenar features (como no script de referência)
    # O script PyTorch faz torch.cat([f1, f2], dim=1), resultando num vetor 2x maior
    combined_features = np.concatenate([original_feat, flipped_feat], axis=0)
    
    return combined_features

def parse_pairs(pairs_path, lfw_dir):
    '''
    Lê o arquivo de pares (formato LFW) e gera caminhos de arquivos e labels.
    Suporta formato de 3 colunas (mesma pessoa) e 4 colunas (pessoas diferentes).
    '''
    pairs = []
    skipped = 0
    
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Arquivo de pares não encontrado: {pairs_path}")

    with open(pairs_path, 'r') as f:
        lines = f.readlines()
        
        # Pular cabeçalho se existir (geralmente a primeira linha tem o numero de splits)
        if len(lines[0].strip().split()) == 1:
            lines = lines[1:]

        for line in lines:
            parts = line.strip().split()
            
            if len(parts) == 3: # Mesma pessoa
                name = parts[0]
                img1 = f"{name}_{int(parts[1]):04d}.jpg"
                img2 = f"{name}_{int(parts[2]):04d}.jpg"
                path1 = os.path.join(lfw_dir, name, img1)
                path2 = os.path.join(lfw_dir, name, img2)
                is_same = 1
                
            elif len(parts) == 4: # Pessoas diferentes
                name1 = parts[0]
                img1 = f"{name1}_{int(parts[1]):04d}.jpg"
                name2 = parts[2]
                img2 = f"{name2}_{int(parts[3]):04d}.jpg"
                path1 = os.path.join(lfw_dir, name1, img1)
                path2 = os.path.join(lfw_dir, name2, img2)
                is_same = 0
            else:
                continue
                
            pairs.append((path1, path2, is_same))
            
    return pairs

def compute_cosine_distance(feat1, feat2):
    '''
    Calcula distância de cosseno.
    Fórmula adaptada do script: f1.dot(f2) / (norm(f1) * norm(f2))
    '''
    dot_product = np.dot(feat1, feat2)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    
    # Adicionado epsilon para evitar divisão por zero
    similarity = dot_product / (norm1 * norm2 + 1e-5)
    return similarity

def find_best_threshold(y_true, y_scores, thresholds):
    best_acc = 0
    best_thresh = 0
    
    for thresh in thresholds:
        # Predição: 1 se similaridade > threshold, 0 caso contrário
        y_pred = (y_scores > thresh).astype(int)
        acc = np.mean(y_pred == y_true)
        
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    return best_thresh, best_acc

# --- Função Principal de Avaliação ---

def evaluate_lfw(model, pairs_path, lfw_dir, image_size, output_dir, logger):
    logger.info(f"Iniciando avaliação LFW...")
    logger.info(f"Pairs: {pairs_path}")
    logger.info(f"Images: {lfw_dir}")
    
    # Preparar Feature Extractor
    # O modelo carregado tem a camada CosFace (Input -> Backbone -> Embedding -> Head)
    # Precisamos extrair do Backbone/Embedding.
    # Assumindo estrutura do CosFace onde layers[-3] é o embedding antes do head,
    # ou buscando pelo nome se possível.
    
    try:
        # Tenta pegar output da camada BatchNormalization final ou Dense de embedding
        # Ajuste baseado na estrutura do seu create_resnet50_cosface
        # Geralmente é a entrada da camada "cosface_loss" ou a saída da camada anterior
        cosface_layer = model.get_layer("cosface_loss")
        embedding_output = cosface_layer.input[0] # Input 0 é o embedding, Input 1 é o label
        feature_extractor = Model(inputs=model.input[0], outputs=embedding_output)
        logger.info("Feature Extractor criado com sucesso a partir da camada de entrada do CosFace.")
    except Exception as e:
        logger.warning(f"Não foi possível isolar automaticamente via 'cosface_loss'. Usando layers[-3] como fallback. Erro: {e}")
        # Fallback baseado no script de inferência original
        feature_extractor = Model(inputs=model.input[0], outputs=model.layers[-3].output)

    # Parse pares
    pairs = parse_pairs(pairs_path, lfw_dir)
    logger.info(f"Total de pares encontrados: {len(pairs)}")
    
    y_true = []
    y_scores = [] # Similaridades
    pairs_processed = 0
    pairs_skipped = 0
    
    for path1, path2, label in pairs:
        # Extrair features com TTA
        feat1 = extract_deep_features_tta(feature_extractor, path1, image_size)
        feat2 = extract_deep_features_tta(feature_extractor, path2, image_size)
        
        if feat1 is None or feat2 is None:
            logger.warning(f"Imagem não encontrada, pulando par: {path1} ou {path2}")
            pairs_skipped += 1
            continue
            
        # Calcular similaridade
        sim = compute_cosine_distance(feat1, feat2)
        
        y_true.append(label)
        y_scores.append(sim)
        pairs_processed += 1
        
        if pairs_processed % 100 == 0:
            print(f"Processados {pairs_processed}/{len(pairs)} pares...", end='\r')
            
    print(f"Processamento concluído.              ")
    logger.info(f"Pares processados: {pairs_processed}")
    logger.info(f"Pares pulados: {pairs_skipped}")
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # --- Cálculo de Métricas ---
    
    # 1. Curva ROC e AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    logger.info(f"AUC (Area Under ROC): {roc_auc:.4f}")
    
    # Plot ROC
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    logger.info(f"Gráfico ROC salvo em: {roc_path}")
    
    # 2. Melhor Threshold e Acurácia
    best_thresh, best_acc = find_best_threshold(y_true, y_scores, thresholds)
    logger.info(f"Best Threshold: {best_thresh:.4f}")
    logger.info(f"Accuracy at Best Threshold: {best_acc:.4f}")
    
    # 3. Métricas Detalhadas (Precision, Recall, F1, Confusion Matrix)
    y_pred = (y_scores > best_thresh).astype(int)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    logger.info("-" * 30)
    logger.info("MÉTRICAS ADICIONAIS")
    logger.info("-" * 30)
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Diferente', 'Mesma'], 
                yticklabels=['Diferente', 'Mesma'])
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão (Threshold={best_thresh:.3f})')
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Matriz de Confusão salva em: {cm_path}")
    
    return {
        'auc': roc_auc,
        'accuracy': best_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': best_thresh
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validação Avançada com TTA e Métricas Completas")
    parser.add_argument('--model_path', type=str, required=True, help='Caminho para o modelo treinado (.keras)')
    parser.add_argument('--lfw_path', type=str, required=True, help='Diretório raiz das imagens do LFW')
    parser.add_argument('--lfw_pairs', type=str, required=True, help='Caminho para o arquivo pairs.txt')
    parser.add_argument('--output_dir', type=str, default='validation_results', help='Diretório para salvar logs e gráficos')
    parser.add_argument('--image_size', type=int, default=112, help='Tamanho das imagens (padrão 112)')
    
    args = parser.parse_args()

    # Setup
    logger = setup_logger(args.output_dir)
    logger.info("Iniciando script de validação avançada.")
    logger.info(f"Configuração: {vars(args)}")

    # Otimização GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU detectada e configurada: {len(gpus)} dispositivo(s)")
        except RuntimeError as e:
            logger.error(f"Erro na configuração da GPU: {e}")

    try:
        # Carregar Modelo
        logger.info(f"Carregando modelo de: {args.model_path}")
        model = load_model(args.model_path, custom_objects={'CosFace': CosFace})
        
        # Executar Validação
        metrics = evaluate_lfw(
            model=model,
            pairs_path=args.lfw_pairs,
            lfw_dir=args.lfw_path,
            image_size=args.image_size,
            output_dir=args.output_dir,
            logger=logger
        )
        
        logger.info("Validação concluída com sucesso.")
        
    except Exception as e:
        logger.error(f"Ocorreu um erro crítico durante a execução: {e}", exc_info=True)
        sys.exit(1)
