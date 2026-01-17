import numpy as np
from PIL import Image
import os
import json
from pathlib import Path
import torch
from torchvision import transforms

from models import create_resnet50 

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    model = create_resnet50(embedding_dim=512, pretrained=False)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    
    # Extrai state_dict da chave 'model' e remove prefixo 'module.' de treinos DDP
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.to(device).eval()
    return model

def get_tta_transforms():
    base_transform = [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    t_orig = transforms.Compose(base_transform)
    t_flip = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return t_orig, t_flip

def extract_features(model, device, img_path: str) -> np.ndarray:
    """
    Extrai embeddings usando TTA (512 orig + 512 flip = 1024 dimensões).
    """
    t_orig, t_flip = get_tta_transforms()
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Erro ao abrir {img_path}: {e}")

    img_orig = t_orig(img).unsqueeze(0).to(device)
    img_flip = t_flip(img).unsqueeze(0).to(device)

    with torch.no_grad():
        f_orig = model(img_orig)
        f_flip = model(img_flip)
        # Concatenação TTA para performance máxima
        features = torch.cat([f_orig, f_flip], dim=1).squeeze().cpu().numpy()
    return features

def extract_batch_embeddings(model, device, image_folder: str, output_prefix: str = "assets/lfw_resnet"):
    """
    Extrai embeddings e salva em formato binário .npy (matriz) e .json (nomes).
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_folder).rglob(f"*{ext}"))
    
    all_embeddings = []
    all_names = []
    
    print(f"Extraindo de {len(image_files)} imagens (ResNet50 TTA)...")
    
    for img_path in image_files:
        try:
            embedding = extract_features(model, device, str(img_path))
            all_embeddings.append(embedding)
            # Salva o nome ou caminho relativo para identificar depois
            all_names.append(str(img_path.relative_to(image_folder)))
        except Exception as e:
            print(f"✗ Erro em {img_path.name}: {e}")
    
    # Converte a lista para uma matriz NumPy (N, 1024)
    embeddings_matrix = np.array(all_embeddings).astype('float32')
    
    # Salva a matriz de embeddings (O arquivo .npy que você citou)
    np.save(f"{output_prefix}_embeddings.npy", embeddings_matrix)
    
    # Salva os nomes em um JSON separado para manter a ordem
    with open(f"{output_prefix}_names.json", 'w') as f:
        json.dump(all_names, f, indent=2)
    
    print(f"\n[SUCESSO]")
    print(f"Matriz de Embeddings: {output_prefix}_embeddings.npy ({embeddings_matrix.shape})")
    print(f"Lista de Nomes: {output_prefix}_names.json")
    
    return embeddings_matrix, all_names

def compare_faces(model, device, img1_path: str, img2_path: str, threshold: float = 0.35) -> tuple[float, bool]:
    """
    Compara dois rostos usando Similaridade de Cosseno (escala CosFace).
    """
    feat1 = extract_features(model, device, img1_path)
    feat2 = extract_features(model, device, img2_path)

    f1, f2 = torch.from_numpy(feat1), torch.from_numpy(feat2)
    similarity = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
    
    score = similarity.item()
    return score, score > threshold

if __name__ == "__main__":
    WEIGHTS = "/workspace/cosface/weights/resnet50_MCP_best.ckpt"
    LFW_PATH = "/workspace/dataset/lfw"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(WEIGHTS, DEVICE)

    # ESTA LINHA ABAIXO É A QUE GERA O ARQUIVO:
    # Ela vai varrer o /workspace/dataset/lfw e criar os arquivos na pasta assets/
    print("\nIniciando extração em lote para Milvus...")
    extract_batch_embeddings(model, DEVICE, LFW_PATH, "assets/lfw_resnet")
