import sys
import os
import tensorflow as tf

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
from config.face_recognition_config import FaceRecognitionConfig, VGGFACE2_TRAIN_PATH, LFW_PATH, LFW_PAIRS_PATH
from src.pipelines.training_pipeline import run_face_training

if __name__ == '__main__':
    config = FaceRecognitionConfig()
    run_face_training(config, VGGFACE2_TRAIN_PATH, LFW_PATH, LFW_PAIRS_PATH)