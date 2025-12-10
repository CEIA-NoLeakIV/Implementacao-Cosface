import sys
import os
import tensorflow as tf
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configura o TensorFlow para alocar memória dinamicamente
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Otimização de memória (Memory Growth) ativada para {len(gpus)} GPU(s).")
    except RuntimeError as e:
        # Memory growth deve ser configurado no início
        print(e)

sys.path.append(os.getcwd())

# Agora podemos importar nossos módulos com segurança
from config.face_recognition_config import FaceRecognitionConfig, VGGFACE2_TRAIN_PATH
from src.pipelines.training_pipeline import run_face_training

if __name__ == '__main__':
    # Defina como True para pular o aquecimento e continuar o fine-tuning
    RESUME_TRAINING = True

    parser = argparse.ArgumentParser(description="Treinamento de Modelo de Reconhecimento Facial")
    parser.add_argument('--dataset_path', type=str, required=True, help='Caminho para o diretório do dataset de treino.')
    parser.add_argument('--align_faces', action='store_true', help='Ativa o alinhamento facial (MTCNN) em tempo real. Lento na primeira época.')
    args = parser.parse_args()
    args = parser.parse_args()

    config = FaceRecognitionConfig()
    run_face_training(
        config,
        args.dataset_path,
        resume_training=RESUME_TRAINING
        align_faces=args.align_faces
    )
