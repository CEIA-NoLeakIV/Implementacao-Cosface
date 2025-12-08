'''
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from PIL import Image

import torch
from torchvision import transforms

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


def eval(model, model_path=None, device=None, lfw_root='data/val'):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device).eval()

    root = lfw_root
    try:
        with open(os.path.join(root, 'lfw_ann.txt')) as f:
            pair_lines = f.readlines()[1:]
    except FileNotFoundError:
        print(f"ERRO: O arquivo de anotacao 'lfw_ann.txt' nao foi encontrado em '{root}'. Verifique o caminho.")
        return 0.0, np.array([])


    predicts = []
    with torch.no_grad():
        for line in pair_lines:
            parts = line.strip().split()

            if len(parts) == 3:
                person_name, img_num1, img_num2 = parts[0], parts[1], parts[2]
                
                # Formata o nome do arquivo como: "Nome_Pessoa_0001.jpg"
                filename1 = f'{person_name}_{int(img_num1):04d}.jpg'
                filename2 = f'{person_name}_{int(img_num2):04d}.jpg'
                
                # Monta o caminho completo
                path1 = os.path.join(root, person_name, filename1)
                path2 = os.path.join(root, person_name, filename2)
                is_same = '1'
            else:
                # Ignora linhas que nao tenham 3 colunas
                continue

            try:
                img1 = Image.open(path1).convert('RGB')
                img2 = Image.open(path2).convert('RGB')
            except FileNotFoundError:
                print(f"Alerta: Imagem nao encontrada, pulando o par: {path1} ou {path2}")
                continue

            f1 = extract_deep_features(model, img1, device)
            f2 = extract_deep_features(model, img2, device)

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append([path1, path2, distance.item(), is_same])
    
    if len(predicts) == 0:
        print("Alerta: Nenhum par valido foi processado na avaliacao.")
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
    best_threshold = find_best_threshold(predicts, thresholds)

    # Visualização da Curva ROC
    try:
        import matplotlib.pyplot as plt
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
        plt.savefig("roc_curve.png")
        print("Gráfico da Curva ROC salvo em: roc_curve.png")
    except Exception as e:
        print(f"[AVISO] Não foi possível salvar o gráfico ROC: {e}")

    # Métricas extras
    try:
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
        import seaborn as sns
        y_pred = [1 if float(distance) > best_threshold else 0 for _, _, distance, _ in predicts]
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
        plt.savefig("confusion_matrix.png")
        print("Gráfico da Matriz de Confusão salvo em: confusion_matrix.png")
    except Exception as e:
        print(f"[AVISO] Não foi possível calcular/salvar métricas extras: {e}")

    print(f'--- Validation Metrics (Verification) ---')
    print(f'LFW - Pairs Processed: {len(y_true)}')
    print(f'AUC (Area Under ROC): {roc_auc:.4f}')
    print(f'Best Threshold (for Accuracy): {best_threshold:.4f}')

    accuracy_at_best_thresh = eval_accuracy(predicts, best_threshold)
    print(f'Accuracy (at Best Threshold): {accuracy_at_best_thresh:.4f}')

    return accuracy_at_best_thresh, predicts

if __name__ == '__main__':
    _, result = eval(sphere20(512).to('cuda'), model_path='weights/sphere20_mcp.pth')
    _, result = eval(sphere36(512).to('cuda'), model_path='weights/sphere36_mcp.pth')
    _, result = eval(MobileNetV1(512).to('cuda'), model_path='weights/mobilenetv1_mcp.pth')
    _, result = eval(MobileNetV2(512).to('cuda'), model_path='weights/mobilenetv2_mcp.pth')
    _, result = eval(mobilenet_v3_small(512).to('cuda'), model_path='weights/mobilenetv3_small_mcp.pth')
    _, result = eval(mobilenet_v3_large(512).to('cuda'), model_path='weights/mobilenetv3_large_mcp.pth')

'''
