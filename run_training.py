import sys
import os
import tensorflow as tf
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Otimização de memória (Memory Growth) ativada para {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

sys.path.append(os.getcwd())

from config.face_recognition_config import FaceRecognitionConfig
from src.pipelines.training_pipeline import run_face_training

if __name__ == '__main__':
    # Defina como True se quiser continuar um treino existente
    RESUME_TRAINING = False 

    parser = argparse.ArgumentParser(description="Treinamento de Modelo de Reconhecimento Facial")
    parser.add_argument('--dataset_path', type=str, required=True, help='Caminho para o diretório do dataset de treino.')
    parser.add_argument('--align_faces', action='store_true', help='Ativa o alinhamento facial (UniFace/MTCNN).')
    
    args = parser.parse_args()

    config = FaceRecognitionConfig()
    
    print(f"Iniciando treinamento com dataset: {args.dataset_path}")
    print(f"Alinhamento facial ativo: {args.align_faces}")

    run_face_training(
        config,
        args.dataset_path,
        resume_training=RESUME_TRAINING, 
        align_faces=args.align_faces
    )
