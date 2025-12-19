# Local: evaluate.py
import argparse
import logging
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Importar configurações e componentes locais
from config.face_recognition_config import FaceRecognitionConfig
from face_embeddings.src.models.Cosface.src.models.heads import CosFace
from src.components.evaluator import LFWEvaluator

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("MainEvaluation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação Profissional de Reconhecimento Facial (TensorFlow)")
    
    parser.add_argument('--model_path', type=str, required=True, help='Caminho para o modelo .keras treinado')
    parser.add_argument('--lfw_path', type=str, required=True, help='Diretório raiz das imagens LFW')
    parser.add_argument('--lfw_pairs', type=str, required=True, help='Caminho para o arquivo pairs.txt')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Onde salvar logs e gráficos')
    parser.add_argument('--gpu_memory_growth', action='store_true', help='Ativar alocação dinâmica de memória GPU')
    
    args = parser.parse_args()

    # Configuração inicial
    logger = setup_logger(args.output_dir)
    
    # Setup de GPU
    if args.gpu_memory_growth:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Memória dinâmica ativada para {len(gpus)} GPU(s).")
            except RuntimeError as e:
                logger.error(f"Erro ao configurar GPU: {e}")

    try:
        # 1. Carregar Configurações Padrão
        config = FaceRecognitionConfig()
        
        # 2. Carregar Modelo
        logger.info(f"Carregando modelo: {args.model_path}")
        model = load_model(args.model_path, custom_objects={'CosFace': CosFace})
        
        # 3. Instanciar Avaliador
        evaluator = LFWEvaluator(model, logger)
        
        # 4. Executar Avaliação
        results = evaluator.evaluate(
            pairs_path=args.lfw_pairs,
            lfw_dir=args.lfw_path,
            output_dir=args.output_dir,
            image_size=config.image_size[0] if isinstance(config.image_size, tuple) else config.image_size
        )
        
        logger.info("Processo de avaliação finalizado com sucesso.")

    except Exception as e:
        logger.error(f"Falha crítica na avaliação: {e}", exc_info=True)