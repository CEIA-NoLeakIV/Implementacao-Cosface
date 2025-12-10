import os
import sys
import argparse
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
sys.path.append(os.getcwd())

from config.face_recognition_config import FaceRecognitionConfig
from src.models.heads import CosFace
from src.components.evaluator import LFWEvaluator

# Configuração de Logs
def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'validation.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("Validation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validação Avançada (LFW, ROC, Confusion Matrix)")
    
    parser.add_argument('--model_path', type=str, required=True, help='Caminho para o modelo treinado (.keras ou .h5)')
    parser.add_argument('--lfw_path', type=str, required=True, help='Diretório raiz das imagens LFW')
    parser.add_argument('--lfw_pairs', type=str, required=True, help='Caminho para o arquivo pairs.txt')
    parser.add_argument('--output_dir', type=str, default='validation_results', help='Pasta para salvar gráficos e logs')
    
    args = parser.parse_args()

    # 1. Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    logger = setup_logger(args.output_dir)
    logger.info("Iniciando processo de validação...")

    try:
        # 2. Carregar Config e Modelo
        config = FaceRecognitionConfig()
        
        logger.info(f"Carregando modelo de: {args.model_path}")
        # Custom objects são necessários para carregar a camada CosFace
        model = load_model(args.model_path, custom_objects={'CosFace': CosFace})
        
        # 3. Inicializar Avaliador
        # A classe LFWEvaluator (criada anteriormente) gerencia TTA e extração de features
        evaluator = LFWEvaluator(model, logger)
        
        # 4. Executar Avaliação
        logger.info(f"Avaliando no dataset LFW: {args.lfw_path}")
        results = evaluator.evaluate(
            pairs_path=args.lfw_pairs,
            lfw_dir=args.lfw_path,
            output_dir=args.output_dir,
            image_size=112
        )
        
        logger.info("Validação concluída com sucesso.")
        print("\n--- Resumo dos Resultados ---")
        print(f"Acurácia: {results['accuracy']:.4f}")
        print(f"AUC:      {results['auc']:.4f}")
        print(f"Threshold:{results['threshold']:.4f}")
        print(f"Logs salvos em: {args.output_dir}")

    except Exception as e:
        logger.error(f"Erro crítico na validação: {e}", exc_info=True)
        sys.exit(1)
