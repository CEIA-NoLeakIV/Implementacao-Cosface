import argparse
import tensorflow as tf
from config.face_recognition_configcopy import FaceRecognitionConfig
from src.data_loader.face_datasetscopy import get_train_val_datasets
from src.data_loader.face_datasetscopy import LFWValidationCallback
#from src.data_loader.facelfw_callback import LFWValidationCallback
from src.losses.margin_lossescopy import CosFace 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validação de Modelo de Reconhecimento Facial")
    parser.add_argument('--model_path', type=str, required=True, help='Caminho para o modelo treinado.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Caminho para o diretório do dataset de validação.')
    parser.add_argument('--lfw_path', type=str, required=False, help='Caminho para o dataset LFW (opcional).')
    parser.add_argument('--lfw_pairs', type=str, required=False, help='Caminho para o arquivo de pares do LFW (opcional).')
    args = parser.parse_args()

    config = FaceRecognitionConfig()

    # Carregar o modelo treinado
    model = tf.keras.models.load_model(args.model_path, custom_objects={'CosFace': CosFace})

    # Carregar o dataset de validação
    _, val_dataset, _ = get_train_val_datasets(
        args.dataset_path,
        config.image_size,
        config.batch_size,
        validation_split=0.0  # Usar todo o dataset como validação
    )

    # Adicionar logs para verificar a criação do val_dataset
    print("[DEBUG] Verificando a criação do val_dataset...")
    if val_dataset is None:
        print("[ERRO] val_dataset é None. Verifique o pipeline de carregamento de dados.")
    else:
        print("[DEBUG] val_dataset criado com sucesso.")

    # Mapear o dataset para o formato esperado pelo modelo
    val_dataset = val_dataset.map(lambda image, label: ((image, tf.one_hot(label, depth=8632)), tf.one_hot(label, depth=8632)))

    # Inspecionar o dataset de validação antes da avaliação
    print("[DEBUG] Inspecionando o dataset de validação...")
    for batch in val_dataset.take(1):
        print("[DEBUG] Batch de validação:", batch)
        for i, example in enumerate(batch):
            if example is None:
                print(f"[ERRO] Exemplo {i} é None no batch de validação")
            elif hasattr(example, 'shape'):
                print(f"[DEBUG] Exemplo {i} shape: {example.shape}")
            else:
                print(f"[ERRO] Exemplo {i} não possui atributo shape")

    # Avaliar o modelo
    print("--- Avaliando no dataset de validação ---")
    results = model.evaluate(val_dataset, verbose=1)
    print("Resultados da avaliação:", results)

    # Validação no LFW, se fornecido
    if args.lfw_path and args.lfw_pairs:
        print("\n--- Avaliando no LFW ---")
        lfw_callback = LFWValidationCallback(
            feature_extractor=model,
            lfw_path=args.lfw_path,
            pairs_path=args.lfw_pairs,
            image_size=config.image_size
        )
        lfw_callback.on_epoch_end(epoch=0)