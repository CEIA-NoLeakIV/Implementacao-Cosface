# Arquivo: src/utils/face_processing.py
import cv2
import numpy as np
from skimage.transform import SimilarityTransform
from mtcnn import MTCNN

# Matriz de referência padrão (ArcFace/CosFace)
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963], # Olho esquerdo
    [73.5318, 51.5014], # Olho direito
    [56.0252, 71.7366], # Nariz
    [41.5493, 92.3655], # Canto boca esq
    [70.7299, 92.2041]  # Canto boca dir
], dtype=np.float32)

def estimate_norm(landmark, image_size=112):
    """Calcula a matriz de transformação para alinhar os olhos."""
    assert landmark.shape == (5, 2)
    
    # Ajuste de escala baseado no tamanho da imagem
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
    """Aplica a transformação afim na imagem."""
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)
    return warped

# --- Detetor Global (para não recriar a cada imagem) ---
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = MTCNN(min_face_size=20)
    return _detector

def process_image_pipeline(image_np):
    """
    Recebe imagem (numpy), detecta rosto, alinha e retorna.
    Se falhar, faz resize simples (fallback).
    """
    # Garantir uint8
    if image_np.dtype != np.uint8:
        image_uint8 = image_np.astype(np.uint8)
    else:
        image_uint8 = image_np

    detector = get_detector()
    
    try:
        # Detecção
        results = detector.detect_faces(image_uint8)
        
        if results:
            # Pega o rosto com maior confiança
            best_face = results[0]
            keypoints = best_face['keypoints']
            
            # Formata landmarks na ordem correta
            landmarks = np.array([
                keypoints['left_eye'],
                keypoints['right_eye'],
                keypoints['nose'],
                keypoints['mouth_left'],
                keypoints['mouth_right']
            ], dtype=np.float32)
            
            # Alinha
            aligned = align_face(image_uint8, landmarks, image_size=112)
            return aligned.astype(np.float32)
            
    except Exception as e:
        print(f"Erro no alinhamento: {e}")

    # Fallback: Resize simples se não achar rosto
    return cv2.resize(image_uint8, (112, 112)).astype(np.float32)
