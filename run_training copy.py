import sys
import os
import tensorflow as tf
import argparse
# Agora podemos importar nossos módulos com segurança
from config.face_recognition_config import FaceRecognitionConfig, VGGFACE2_TRAIN_PATH
from src.pipelines.training_pipeline import run_face_training

if __name__ == '__main__':
    # Defina como True para pular o aquecimento e continuar o fine-tuning
    RESUME_TRAINING = True

    parser = argparse.ArgumentParser(description="Treinamento de Modelo de Reconhecimento Facial")
    parser.add_argument('--dataset_path', type=str, required=True, help='Caminho para o diretório do dataset de treino.')
    args = parser.parse_args()

    config = FaceRecognitionConfig()
    run_face_training(
        config,
        args.dataset_path,
        resume_training=RESUME_TRAINING
    )