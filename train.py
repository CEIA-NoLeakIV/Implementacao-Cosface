import argparse
import os
from src.data_loader.face_datasets import get_train_val_datasets
from src.models.builder import build_face_model
from src.components.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path dataset')
    parser.add_argument('--lfw_path', type=str, default=None)
    parser.add_argument('--lfw_pairs', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01) # Reduzi para 0.01 para melhor convergência
    parser.add_argument('--align_faces', action='store_true', help='Filtrar faces antes do treino')

    args = parser.parse_args()

    if args.align_faces:
        print("--> Alinhamento/Filtragem de faces ativado.")

    # Configuração simples
    class Config:
        image_size = (112, 112)
        epochs = args.epochs
        num_classes = 0 

    config = Config()
    
    # Passamos a flag align_faces para o loader
    train_ds, val_ds, num_classes = get_train_val_datasets(
        args.data, 
        config.image_size, 
        args.batch_size,
        align_faces=args.align_faces
    )
    config.num_classes = num_classes

    model = build_face_model(config)
    trainer = Trainer(model, config, output_dir="experiments/clean_run")
    trainer.compile(learning_rate=args.lr)
    
    lfw_conf = {'path': args.lfw_path, 'pairs': args.lfw_pairs} if args.lfw_path else None
    
    # O fit agora gerencia os logs e o LFW automaticamente
    trainer.fit(train_ds, val_ds, lfw_config=lfw_conf)

if __name__ == "__main__":
    main()