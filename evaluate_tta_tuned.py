# [FINAL] evaluate_tta_tuned.py
import os
import csv
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict

# --- Imports Personalizados ---
from utils.face_validation import FaceValidator, validate_lfw_pairs, validate_audit_log_pairs, print_validation_summary

# ==========================================
# 1. Lógica de Carregamento de Datasets
# ==========================================
def load_audit_log_pairs(audit_log_path, val_root):
    pairs = []
    if not os.path.exists(audit_log_path): return []
    
    def find_image_path(filename, cpf=None):
        search_paths = [
            os.path.join(val_root, filename),
            os.path.join(val_root, 'train', filename),
            os.path.join(val_root, 'val', filename),
        ]
        if cpf: search_paths.append(os.path.join(val_root, cpf, filename))
        for path in search_paths:
            if os.path.exists(path): return path
        return None

    with open(audit_log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_file, m_file = row.get('query_filename', '').strip(), row.get('match_filename', '').strip()
            q_cpf, m_cpf = row.get('query_cpf', '').strip(), row.get('match_cpf', '').strip()
            if not q_file or not m_file: continue
            
            is_same = '1' if q_cpf == m_cpf else '0'
            p1, p2 = find_image_path(q_file, q_cpf), find_image_path(m_file, m_cpf)
            if p1 and p2: pairs.append((p1, p2, is_same))
    return pairs

def load_mapping_val_pairs(mapping_val_path, max_pairs=None, negative_ratio=1.0, seed=42):
    random.seed(seed)
    images_by_cpf = defaultdict(list)
    if not os.path.exists(mapping_val_path): return []

    with open(mapping_val_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpf, img_path = row.get('cpf', '').strip(), row.get('caminho_imagem', '').strip()
            if cpf and img_path and os.path.exists(img_path):
                images_by_cpf[cpf].append(img_path)

    positive_pairs = []
    for cpf, imgs in images_by_cpf.items():
        if len(imgs) < 2: continue
        curr_pairs = [(imgs[i], imgs[j], '1') for i in range(len(imgs)) for j in range(i+1, len(imgs))]
        if max_pairs and len(curr_pairs) > max_pairs: curr_pairs = random.sample(curr_pairs, max_pairs)
        positive_pairs.extend(curr_pairs)

    num_neg = int(len(positive_pairs) * negative_ratio)
    cpfs = list(images_by_cpf.keys())
    negative_pairs = set()
    while len(negative_pairs) < num_neg:
        c1, c2 = random.sample(cpfs, 2)
        i1, i2 = random.choice(images_by_cpf[c1]), random.choice(images_by_cpf[c2])
        pair = tuple(sorted((i1, i2)))
        if pair not in negative_pairs:
            negative_pairs.add(pair)
    
    pairs = positive_pairs + [(p[0], p[1], '0') for p in negative_pairs]
    random.shuffle(pairs)
    return pairs

def load_custom_pairs(ann_file, root):
    pairs = []
    with open(ann_file, 'r') as f: lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            pairs.append((os.path.join(root, parts[0], parts[1]), os.path.join(root, parts[0], parts[2]), '1'))
        elif len(parts) == 4:
            pairs.append((os.path.join(root, parts[0], parts[1]), os.path.join(root, parts[2], parts[3]), '0'))
    return pairs

# ==========================================
# 2. Extração de Características (TTA)
# ==========================================
def extract_deep_features(model, image, device):
    """
    Mantém a lógica de TTA (Original + Flip) para performance máxima.
    """
    original_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    flipped_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img_orig = original_transform(image).unsqueeze(0).to(device)
    img_flip = flipped_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat_orig = model(img_orig)
        feat_flip = model(img_flip)

    return torch.cat([feat_orig, feat_flip], dim=1).squeeze().cpu()

# ==========================================
# 3. Métricas Avançadas
# ==========================================
def compute_advanced_metrics(y_true, similarities, threshold=0.35):
    fpr, tpr, thresholds_roc = roc_curve(y_true, similarities)
    roc_auc = auc(fpr, tpr)

    # EER
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    eer_thresh = thresholds_roc[eer_idx]

    # TAR @ FAR
    metrics = {'auc': roc_auc, 'eer': eer, 'eer_threshold': eer_thresh}
    for far_target in [0.001, 0.01, 0.1]:
        idx = np.where(fpr <= far_target)[0]
        metrics[f'TAR@FAR={far_target}'] = tpr[idx[-1]] if len(idx) > 0 else 0

    # Classification Metrics at Threshold
    y_pred = (similarities > threshold).astype(int)
    metrics.update({
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'mean_similarity': np.mean(similarities),
        'std_similarity': np.std(similarities)
    })
    
    # Best Threshold Search
    best_acc = 0
    best_thr = 0
    for thr in np.linspace(similarities.min(), similarities.max(), 100):
        acc = accuracy_score(y_true, (similarities > thr).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    metrics['best_threshold'] = best_thr
    metrics['best_accuracy'] = best_acc

    return metrics

def load_celeba_pairs(ann_file, root):
    pairs = []
    if not os.path.exists(ann_file):
        print(f"Error: Annotation file not found: {ann_file}")
        return []
        
    with open(ann_file, 'r') as f:
        lines = f.readlines()
        # Pula cabeçalho se existir (geralmente linha que começa com 'identity' ou '#')
        if len(lines) > 0 and (lines[0].startswith('#') or 'identity' in lines[0]):
            lines = lines[1:]

    for line in lines:
        parts = line.strip().split()
        # Formato esperado: filename1 filename2 is_same(0/1)
        # OU se o seu arquivo tiver apenas pares positivos: filename1 filename2
        
        if len(parts) == 3:
            f1, f2, label = parts
            path1 = os.path.join(root, 'img_align_celeba', 'img_align_celeba', f1)
            path2 = os.path.join(root, 'img_align_celeba', 'img_align_celeba', f2)
            pairs.append((path1, path2, label))
        elif len(parts) == 2:
            # Assumindo pares positivos se não tiver label explícito (comum em alguns arquivos do celeba)
            f1, f2 = parts
            path1 = os.path.join(root, 'img_align_celeba', 'img_align_celeba', f1)
            path2 = os.path.join(root, 'img_align_celeba', 'img_align_celeba', f2)
            pairs.append((path1, path2, '1')) # Assume positivo
            
    return pairs

# ==========================================
# 4. Função Principal de Avaliação (CORRIGIDA)
# ==========================================
def eval(model, device=None, val_dataset='lfw', val_root='data/val', 
         threshold=0.35, save_metrics_path=None, 
         face_validator=None, no_face_policy='exclude'):
    
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # 1. Carregar Pares
    pairs = []
    
    # --- LFW ---
    if val_dataset == 'lfw':
        ann_file = os.path.join(val_root, 'lfw_ann.txt')
        if face_validator:
            # Se tiver validador, usa a função específica dele
            pairs, _, face_stats = validate_lfw_pairs(face_validator, val_root, ann_file, no_face_policy)
        else:
            # Fallback manual APENAS para LFW (formato: nome num1 num2)
            if os.path.exists(ann_file):
                with open(ann_file) as f:
                    lines = f.readlines()[1:]
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        p, i1, i2 = parts
                        pairs.append((os.path.join(val_root, p, f'{p}_{int(i1):04d}.jpg'),
                                      os.path.join(val_root, p, f'{p}_{int(i2):04d}.jpg'), '1'))
                    elif len(parts) == 4:
                        p1, i1, p2, i2 = parts
                        pairs.append((os.path.join(val_root, p1, f'{p1}_{int(i1):04d}.jpg'),
                                      os.path.join(val_root, p2, f'{p2}_{int(i2):04d}.jpg'), '0'))

    # --- CELEBA (CORREÇÃO AQUI) ---
    elif val_dataset == 'celeba':
        ann_file = os.path.join(val_root, 'celeba_pairs.txt')
        # Sempre usa a função dedicada load_celeba_pairs
        raw_pairs = load_celeba_pairs(ann_file, val_root)
        
        if face_validator:
            # Para CelebA, usamos validate_audit_log_pairs pois raw_pairs já é uma lista de caminhos completos
            pairs, _, face_stats = validate_audit_log_pairs(face_validator, raw_pairs, no_face_policy)
        else:
            pairs = raw_pairs

    # --- AUDIT LOG ---
    elif val_dataset == 'audit_log':
        raw_pairs = load_audit_log_pairs(os.path.join(val_root, 'audit_log.csv'), val_root)
        if face_validator:
            pairs, _, face_stats = validate_audit_log_pairs(face_validator, raw_pairs, no_face_policy)
        else:
            pairs = raw_pairs
            
    # --- MAPPING VAL ---
    elif val_dataset == 'mapping_val':
        raw_pairs = load_mapping_val_pairs(os.path.join(val_root, 'mapping_val.csv'))
        if face_validator:
             pairs, _, face_stats = validate_audit_log_pairs(face_validator, raw_pairs, no_face_policy)
        else:
             pairs = raw_pairs
        
    # --- CUSTOM ---
    elif val_dataset == 'custom':
        pairs = load_custom_pairs(os.path.join(val_root, 'custom.txt'), val_root)
    
    # Reportar estatísticas de validação de face se existirem
    if face_validator and 'face_stats' in locals():
        print_validation_summary(face_validator)

    if not pairs:
        print(f"Warning: No pairs found for evaluation on {val_dataset}.")
        return 0.0, np.array([]), {}

    # 2. Processar Pares (Inferência)
    predicts = []
    print(f"Evaluating {len(pairs)} pairs on {val_dataset} with TTA...")
    
    for path1, path2, is_same in pairs:
        try:
            img1 = Image.open(path1).convert('RGB')
            img2 = Image.open(path2).convert('RGB')
            
            f1 = extract_deep_features(model, img1, device)
            f2 = extract_deep_features(model, img2, device)
            
            # Similaridade Cosseno
            sim = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append([path1, path2, sim.item(), is_same])
        except Exception as e:
            # print(f"Error processing pair: {e}") # Descomente para debug se necessário
            continue

    if not predicts: 
        return 0.0, np.array([]), {}

    # 3. Calcular Métricas
    predicts = np.array(predicts, dtype=object)
    metrics = compute_advanced_metrics(
        predicts[:, 3].astype(int), 
        predicts[:, 2].astype(float), 
        threshold
    )
    
    if face_validator and 'face_stats' in locals():
        metrics['face_validation_stats'] = face_stats

    # 4. Salvar Gráficos
    if save_metrics_path:
        os.makedirs(save_metrics_path, exist_ok=True)
        
        # --- A. Plot ROC Curve ---
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(predicts[:, 3].astype(int), predicts[:, 2].astype(float))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {metrics["auc"]:.4f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {val_dataset}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_metrics_path, f'{val_dataset}_roc.png'), dpi=300)
        plt.close()

        # --- B. Plot Confusion Matrix ---
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Different', 'Same'], 
                        yticklabels=['Different', 'Same'])
            plt.title(f'Confusion Matrix - {val_dataset}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(save_metrics_path, f'{val_dataset}_confusion_matrix.png'), dpi=300)
            plt.close()
            
    return metrics['mean_similarity'], predicts, metrics

# ==========================================
# 5. Execução Principal (Standalone)
# ==========================================
if __name__ == '__main__':
    from models import create_resnet50, sphere20, MobileNetV1
    
    parser = argparse.ArgumentParser(description="Standalone Evaluation Script")
    parser.add_argument('--model-path', type=str, required=True, help='Path to .ckpt file')
    parser.add_argument('--network', type=str, default='resnet50', choices=['resnet50', 'sphere20', 'mobilenetv1'])
    parser.add_argument('--val-dataset', type=str, default='lfw', choices=['lfw', 'celeba', 'audit_log', 'custom', 'mapping_val'])
    parser.add_argument('--val-root', type=str, default='data/val', help='Root directory of validation images')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    # Argumento novo para salvar resultados
    parser.add_argument('--save-dir', type=str, default='evaluation_results', help='Directory to save plots and metrics')
    
    parser.add_argument('--use-face-validation', action='store_true', help="Enable RetinaFace cleaning")
    parser.add_argument('--no-face-policy', type=str, default='exclude', choices=['exclude', 'include'])
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Carregar Modelo
    if args.network == 'resnet50':
        model = create_resnet50(embedding_dim=512, pretrained=False)
    elif args.network == 'sphere20':
        model = sphere20(embedding_dim=512)
    else:
        model = MobileNetV1(embedding_dim=512)
        
    model.to(device)

    if os.path.isfile(args.model_path):
        print(f"Loading checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Remove prefixo 'module.' se necessário
        new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state)
    else:
        print(f"Checkpoint not found: {args.model_path}")
        exit(1)

    # Inicializar Validador
    face_validator = None
    if args.use_face_validation:
        face_validator = FaceValidator()

    # Executar Avaliação (AGORA COM save_metrics_path)
    print(f"Starting evaluation on {args.val_dataset}...")
    save_path = os.path.join(args.save_dir, args.val_dataset)
    
    mean_sim, predictions, metrics = eval(
        model, 
        device=device,
        val_dataset=args.val_dataset,
        val_root=args.val_root,
        threshold=0.35, 
        save_metrics_path=save_path,
        face_validator=face_validator,
        no_face_policy=args.no_face_policy
    )

    print("\n" + "="*50)
    print(f"FINAL RESULTS: {args.val_dataset.upper()}")
    print("="*50)
    print(f"Best Accuracy:  {metrics.get('best_accuracy', 0.0):.4f}")
    print(f"Best Threshold: {metrics.get('best_threshold', 0.0):.4f}")
    print(f"AUC:            {metrics.get('auc', 0.0):.4f}")
    print(f"EER:            {metrics.get('eer', 0.0):.4f}")
    print("-" * 30)
    print(f"Plots saved to: {save_path}")
    print("="*50 + "\n")
