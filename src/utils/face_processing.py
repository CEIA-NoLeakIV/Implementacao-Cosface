# Arquivo: src/utils/face_processing.py
import cv2
import numpy as np
from skimage.transform import SimilarityTransform
from uniface import RetinaFace  # Importação do UniFace

# --- Configuração Global ---
# Matriz de referência (ArcFace/CosFace standard 112x112)
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963], # Olho esquerdo
    [73.5318, 51.5014], # Olho direito
    [56.0252, 71.7366], # Nariz
    [41.5493, 92.3655], # Canto boca esq
    [70.7299, 92.2041]  # Canto boca dir
], dtype=np.float32)

# Variável global para manter o modelo carregado na memória (evita re-inicializar a cada imagem)
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        # Inicializa o RetinaFace via ONNX (muito rápido)
        # O modelo será baixado automaticamente na primeira execução
        _detector = RetinaFace() 
    return _detector

def estimate_norm(landmark, image_size=112):
    """Calcula a matriz de transformação para alinhar os olhos."""
    assert landmark.shape == (5, 2)
    
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    dst = REFERENCE_LANDMARKS * ratio
    dst[:, 0] += diff_x

    tform = SimilarityTransform()
    tform.estimate(landmark, dst)
    return tform.params[0:2, :]

def align_face(image, landmark, image_size=112):
    """Aplica a transformação geométrica (Warp)."""
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)
    return warped

def process_image_pipeline(image_np):
    """
    Função principal chamada pelo TensorFlow.
    1. Converte RGB -> BGR
    2. Detecta Face (UniFace)
    3. Alinha
    4. Retorna
    """
    # Converter para uint8 (formato de imagem padrão)
    if image_np.dtype != np.uint8:
        image_uint8 = image_np.astype(np.uint8)
    else:
        image_uint8 = image_np

    # O TensorFlow carrega em RGB, mas o UniFace/OpenCV espera BGR
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

    detector = get_detector()
    
    try:
        # Detecta faces
        # O UniFace retorna uma lista de dicionários: [{'bbox':..., 'landmarks':...}]
        faces = detector.detect(image_bgr)
        
        if len(faces) > 0:
            # Pega a face com maior confiança (geralmente a primeira da lista)
            best_face = faces[0]
            landmarks = np.array(best_face['landmarks'], dtype=np.float32)
            
            # Alinha usando a imagem original (pode ser em RGB ou BGR, vamos manter RGB para o TF)
            # Como 'align_face' usa warpAffine, não importa a cor, apenas a geometria
            aligned = align_face(image_uint8, landmarks, image_size=112)
            
            return aligned.astype(np.float32)
            
    except Exception as e:
        # Em caso de erro, imprime mas não quebra o treino
        print(f"[Aviso] Erro no UniFace: {e}")

    # Fallback: Se não detectou nada, retorna array de zeros (será filtrado no loader)
    return np.zeros((112, 112, 3), dtype=np.float32)


def process_image_pipeline_with_detection_flag(image_np):
    """
    Função para validação que retorna a imagem alinhada e um flag indicando se detectou face.
    Usado para excluir amostras sem detecção durante a validação.
    
    Returns:
        tuple: (aligned_image, face_detected_flag)
        - aligned_image: imagem alinhada (112, 112, 3) ou zeros se não detectou
        - face_detected_flag: 1.0 se detectou face, 0.0 caso contrário
    """
    # Converter para uint8 (formato de imagem padrão)
    if image_np.dtype != np.uint8:
        image_uint8 = image_np.astype(np.uint8)
    else:
        image_uint8 = image_np

    # O TensorFlow carrega em RGB, mas o UniFace/OpenCV espera BGR
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

    detector = get_detector()
    
    try:
        # Detecta faces
        faces = detector.detect(image_bgr)
        
        if len(faces) > 0:
            # Pega a face com maior confiança (geralmente a primeira da lista)
            best_face = faces[0]
            landmarks = np.array(best_face['landmarks'], dtype=np.float32)
            
            # Alinha usando a imagem original
            aligned = align_face(image_uint8, landmarks, image_size=112)
            
            return aligned.astype(np.float32), np.float32(1.0)  # Face detectada
            
    except Exception as e:
        # Em caso de erro, considera como não detectado
        print(f"[Aviso] Erro no RetinaFace durante validação: {e}")

    # Não detectou face: retorna zeros e flag 0.0
    return np.zeros((112, 112, 3), dtype=np.float32), np.float32(0.0)
