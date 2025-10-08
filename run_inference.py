# Local: run_inference.py

import sys
import os
import tensorflow as tf

# --- OTIMIZAÇÃO DE MEMÓRIA DA GPU ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Otimização de memória (Memory Growth) ativada para {len(gpus)} GPU(s).")
  except RuntimeError as e:
    print(e)
# --- FIM DA OTIMIZAÇÃO ---

sys.path.append(os.getcwd())

from config.training_config import CosFaceConfig, InferenceConfig
# A importação do pipeline permanece a mesma
from src.pipelines.inference_pipeline import run_cosface_inference

if __name__ == '__main__':
    cosface_cfg = CosFaceConfig()
    inference_cfg = InferenceConfig()
    
    # Chamamos a nova função de pipeline, passando apenas as configs necessárias
    run_cosface_inference(cosface_cfg, inference_cfg)