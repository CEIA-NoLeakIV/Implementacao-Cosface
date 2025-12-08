#Versão para avaliação do modelo com TTA (Test Time Augmentation) no LFW
 
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from PIL import Image
import cv2
from insightface.model_zoo.scrfd import SCRFD
from insightface.utils.storage import ensure_available

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

from models import (
    sphere20,
    sphere36,
    sphere64,
    MobileNetV1,
    MobileNetV2,
    mobilenet_v3_small,
    mobilenet_v3_large,
    create_resnet50
)


def extract_deep_features(model, image, device):
    """
    Extracts deep features for an image using the model, including both the original and flipped versions.

    Args:
        model (torch.nn.Module): The pre-trained deep learning model used for feature extraction.
        image (PIL.Image): The input image to extract features from.
        device (torch.device): The device (CPU or GPU) on which the computation will be performed.

    Returns:
        torch.Tensor: Combined feature vector of original and flipped images.
    """

    # Define transforms
    original_transform = transforms.Compose([
	transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    flipped_transform = transforms.Compose([
	transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=1.0),  # Always flip
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Apply transforms
    original_image_tensor = original_transform(image).unsqueeze(0).to(device)
    flipped_image_tensor = flipped_transform(image).unsqueeze(0).to(device)

    # Extract features
    original_features = model(original_image_tensor)
    flipped_features = model(flipped_image_tensor)

    # Combine and return features
    combined_features = torch.cat([original_features, flipped_features], dim=1).squeeze()
    return combined_features


def k_fold_split(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    fold_size = n // n_folds

    for idx in range(n_folds):
        test = base[idx * fold_size:(idx + 1) * fold_size]
        train = base[:idx * fold_size] + base[(idx + 1) * fold_size:]
        folds.append([train, test])

    return folds


def eval_accuracy(predictions, threshold):
    y_true = []
    y_pred = []

    for _, _, distance, gt in predictions:
        y_true.append(int(gt))
        pred = 1 if float(distance) > threshold else 0
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)
    return accuracy


def find_best_threshold(predictions, thresholds):
    best_accuracy = 0
    best_threshold = 0

    for threshold in thresholds:
        accuracy = eval_accuracy(predictions, threshold)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold


def has_face(detector, img_np):
    """Verifica se há rosto na imagem usando múltiplos thresholds e tamanhos, igual ao cropping_vgg.py. Adiciona debug."""
    if img_np is None:
        print("[DEBUG] Imagem é None!")
        return False
    print(f"[DEBUG] Shape da imagem: {img_np.shape}")
    tries = [
        ((640, 640), 0.6),
        ((800, 800), 0.55),
        ((1024, 1024), 0.50),
        ((1024, 1024), 0.4)  # Threshold ainda mais baixo
    ]
    for det_size, thr in tries:
        detector.conf_threshold = thr
        bboxes, _ = detector.detect(img_np, input_size=det_size, max_num=1)
        print(f"[DEBUG] Tamanho: {det_size}, Threshold: {thr}, BBoxes: {bboxes}")
        if bboxes is not None and len(bboxes) > 0:
            print(f"[DEBUG] Face detectada com threshold {thr} e tamanho {det_size}")
            return True
    print("[DEBUG] Nenhuma face detectada nesta imagem.")
    return False


def eval(model, model_path=None, device=None, lfw_root='data/val', results_dir="evaluation_results", verify_faces=False, detector=None):
    # Cria o diretório de resultados, se não existir
    os.makedirs(results_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device).eval()

    root = lfw_root
    try:
        with open(os.path.join(root, 'lfw_ann.txt')) as f:
            pair_lines = f.readlines()
    except FileNotFoundError:
        print(f"ERRO: O arquivo de anotacao 'lfw_ann.txt' nao foi encontrado em '{root}'. Verifique o caminho.")
        return 0.0, np.array([])


    predicts = []
    pairs_skipped = 0
    face_cache = dict()  # Cache de detecção por imagem
    with torch.no_grad():
        for line in pair_lines:
            parts = line.strip().split()
            # ...existing code...
            if len(parts) == 3:
                person_name, img_num1, img_num2 = parts[0], parts[1], parts[2]
                filename1 = f'{person_name}_{int(img_num1):04d}.jpg'
                filename2 = f'{person_name}_{int(img_num2):04d}.jpg'
                path1 = os.path.join(root, person_name, filename1)
                path2 = os.path.join(root, person_name, filename2)
                is_same = '1'
            elif len(parts) == 4:
                person1, img_num1, person2, img_num2 = parts[0], parts[1], parts[2], parts[3]
                filename1 = f'{person1}_{int(img_num1):04d}.jpg'
                filename2 = f'{person2}_{int(img_num2):04d}.jpg'
                path1 = os.path.join(root, person1, filename1)
                path2 = os.path.join(root, person2, filename2)
                is_same = '0'
            else:
                continue
            try:
                img1_pil = Image.open(path1).convert('RGB')
                img2_pil = Image.open(path2).convert('RGB')
            except FileNotFoundError:
                print(f"Alerta: Imagem nao encontrada, pulando o par: {path1} ou {path2}")
                continue
            # --- Verificação Opcional de Face com cache ---
            if verify_faces:
                if detector is None:
                    raise ValueError("O detector de faces deve ser fornecido quando 'verify_faces' é True.")
                # img1
                if path1 not in face_cache:
                    img1_np = cv2.cvtColor(np.array(img1_pil), cv2.COLOR_RGB2BGR)
                    face_cache[path1] = has_face(detector, img1_np)
                # img2
                if path2 not in face_cache:
                    img2_np = cv2.cvtColor(np.array(img2_pil), cv2.COLOR_RGB2BGR)
                    face_cache[path2] = has_face(detector, img2_np)
                if not face_cache[path1] or not face_cache[path2]:
                    print(f"Alerta: Nenhuma face detectada em {path1} ou {path2}. Pulando par.")
                    pairs_skipped += 1
                    continue
            # --- Fim da Verificação ---
            f1 = extract_deep_features(model, img1_pil, device)
            f2 = extract_deep_features(model, img2_pil, device)
            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append([path1, path2, distance.item(), is_same])
    
    if len(predicts) == 0:
        print("Alerta: Nenhum par valido foi processado na avaliacao.")
        if verify_faces:
            print(f"Total de pares pulados por falta de face: {pairs_skipped}")
        return 0.0, np.array([])

    predicts = np.array(predicts)
    similarities = predicts[:, 2].astype(float)
    y_true = predicts[:, 3].astype(int)
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    fpr, tpr, thresholds = roc_curve(y_true, similarities)
    tar_array = tpr
    far_array = fpr
    frr_array = 1 - tar_array
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'Curva ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FAR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TAR)')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    # Salva a figura em um arquivo
    plot_filename = os.path.join(results_dir, "roc_curve.png")
    plt.savefig(plot_filename)
    print(f"Gráfico da Curva ROC salvo em: {plot_filename}")

    best_threshold = find_best_threshold(predicts, thresholds)

    y_pred = [1 if float(distance) > best_threshold else 0 for _, _, distance, _ in predicts]

    # Calcula as métricas
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'\n--- Métricas Adicionais (no melhor threshold) ---')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'Matriz de Confusão:\n{cm}')

    # Visualização da Matriz de Confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Diferente', 'Mesma'], yticklabels=['Diferente', 'Mesma'])
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    confusion_matrix_filename = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_filename)
    print(f"Gráfico da Matriz de Confusão salvo em: {confusion_matrix_filename}")

    print(f'--- Validation Metrics (Verification) ---')
    print(f'LFW - Pairs Processed: {len(y_true)}')
    if verify_faces:
        print(f'LFW - Pairs Skipped (No Face): {pairs_skipped}')
    print(f'AUC (Area Under ROC): {roc_auc:.4f}')
    print(f'Best Threshold (for Accuracy): {best_threshold:.4f}')

    accuracy_at_best_thresh = eval_accuracy(predicts, best_threshold)
    print(f'Accuracy (at Best Threshold): {accuracy_at_best_thresh:.4f}')

    return accuracy_at_best_thresh, predicts

if __name__ == '__main__':
    # --- Configurações da Avaliação ---
    # 1. O caminho para o seu dataset LFW que você forneceu:
    LFW_DATASET_PATH = "/home/ubuntu/noleak/face_embeddings/data/raw/lfw"
    
    # 2. Caminho para o checkpoint do modelo treinado
    CHECKPOINT_PATH = "/home/ubuntu/noleak/face_embeddings/src/models/Cosface_Refactor/weights/ResNet50_CosFace_VGGFace2/resnet50_COS_last.ckpt"
    
    # 3. (NOVO) Diretório para salvar os resultados (gráficos, etc.)
    RESULTS_DIR = "evaluation_results"

    # 4. (NOVO) Ativar a verificação de faces no dataset de validação
    VERIFY_FACES_IN_LFW = False
    # -----------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- Inicialização do Detector (se necessário) ---
    face_detector = None
    if VERIFY_FACES_IN_LFW:
        print("Inicializando o detector de faces SCRFD...")
        scrfd_path = ensure_available('models', 'scrfd_10g_bnkps.onnx')
        face_detector = SCRFD(model_file=scrfd_path)
        face_detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1) # ctx_id=-1 para CPU
        print("Detector de faces pronto.")
    # ---------------------------------------------

    # Carregar o backbone ResNet50 (sem pesos pré-treinados da ImageNet)
    model = create_resnet50(embedding_dim=512, pretrained=False)
    
    print(f"Carregando pesos do checkpoint: {CHECKPOINT_PATH}")
    
    # O checkpoint salva um dicionário ('model', 'optimizer', etc.)
    # Nós carregamos apenas o 'model'
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])

    print(f"Iniciando avaliação no LFW em: {LFW_DATASET_PATH}")
    
    # Chamar a função 'eval' com os novos parâmetros
    accuracy, predictions = eval(
        model=model, 
        model_path=None, # Os pesos já estão carregados
        device=device, 
        lfw_root=LFW_DATASET_PATH,
        results_dir=RESULTS_DIR,
        verify_faces=VERIFY_FACES_IN_LFW,
        detector=face_detector
    )
    
    print("\n--- Resultado Final da Avaliação (LFW) ---")
    print(f"(Métricas de Verificação: AUC, FAR/FRR, etc. estão nos logs acima)")
    print(f"Acurácia (no melhor threshold): {accuracy:.4f}")
    print("---------------------------------------------")

'''
    _, result = eval(sphere20(512).to('cuda'), model_path='weights/sphere20_mcp.pth')
    _, result = eval(sphere36(512).to('cuda'), model_path='weights/sphere36_mcp.pth')
    _, result = eval(MobileNetV1(512).to('cuda'), model_path='weights/mobilenetv1_mcp.pth')
    _, result = eval(MobileNetV2(512).to('cuda'), model_path='weights/mobilenetv2_mcp.pth')
    _, result = eval(mobilenet_v3_small(512).to('cuda'), model_path='weights/mobilenetv3_small_mcp.pth')
    _, result = eval(mobilenet_v3_large(512).to('cuda'), model_path='weights/mobilenetv3_large_mcp.pth')
    '''