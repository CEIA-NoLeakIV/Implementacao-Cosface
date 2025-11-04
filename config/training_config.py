class CosFaceConfig:
    ARCH = 'vgg8_cosface'
    NUM_FEATURES = 3

    # --- Parâmetros de Treinamento ---
    SCHEDULER = 'CosineAnnealing'
    EPOCHS = 50
    BATCH_SIZE = 64
    OPTIMIZER = 'SGD'
    LEARNING_RATE = 0.1
    MIN_LEARNING_RATE = 0.001
    MOMENTUM = 0.5

    # --- Caminhos (Paths) ---
    CHECKPOINT_PATH = 'models/cosface_vgg8.keras'
    LOG_PATH = 'deliverables/cosface_training_log.csv'

# --- Parâmetros do Modelo Baseline ---
class BaselineConfig:
    EMBEDDING_SIZE = 512 
    NUM_CLASSES = 480 
    CHECKPOINT_PATH = 'models/baseline_model_epoch_20.pth'

# --- Parâmetros de Inferência ---
class InferenceConfig:
    SAMPLE_SIZE = 1000  # Amostras para a visualização
    OUTPUT_FIGURE_PATH = 'reports/figures/comparative_visualization.png'