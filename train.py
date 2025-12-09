import os
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score

import evaluate
from utils.dataset import ImageFolder
from utils.metrics import MarginCosineProduct, AngleLinear, CosFace
from utils.general import (
    setup_seed,
    reduce_tensor,
    save_on_master,
    calculate_accuracy,
    init_distributed_mode,
    AverageMeter,
    EarlyStopping,
    LOGGER,
    add_file_handler,
)

from modelos import (
    sphere20,
    sphere36,
    sphere64,
    MobileNetV1,
    MobileNetV2,
    mobilenet_v3_small,
    mobilenet_v3_large,
    create_resnet50
)

# Importação da classe de landmarks
from modelos.landmark_conditioned import LandmarkConditionedModel
from utils.validation_split import create_validation_split

def parse_arguments():
    parser = argparse.ArgumentParser(description=("Command-line arguments for training a face recognition model"))

    # Dataset and Paths
    parser.add_argument('--root', type=str, default='data/train/webface_112x112/', help='Path to dataset.')
    parser.add_argument('--database', type=str, default='WebFace', choices=['WebFace', 'VggFace2', "MS1M"], help='Database name.')
    parser.add_argument('--dataset-fraction', type=float, default=1.0, help='Fração do dataset (0.0 a 1.0).')

    # Model Settings
    parser.add_argument('--network', type=str, default='sphere20', help='Network architecture.')
    parser.add_argument('--classifier', type=str, default='MCP', choices=['ARC', 'MCP', 'AL', 'L', 'COS'], help='Classifier type.')

    # Training Hyperparameters
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
    
    # Scheduler & Optimizer
    parser.add_argument('--lr-scheduler', type=str, default='MultiStepLR', choices=['StepLR', 'MultiStepLR'])
    parser.add_argument('--step-size', type=int, default=10, help='Step size for StepLR.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for LR scheduler.')
    parser.add_argument('--milestones', type=int, nargs='+', default=[10, 20, 25], help='Milestones for MultiStepLR.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum factor for SGD.')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay for SGD.')
    
    # System
    parser.add_argument('--save-path', type=str, default='weights', help='Path to save model checkpoints.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of data loader workers.')
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to continue training.")
    parser.add_argument('--print-freq', type=int, default=100, help='Frequency for printing training progress.')
    parser.add_argument("--world-size", default=1, type=int, help="Number of distributed processes")
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="Forces deterministic algorithms.")

    # Landmarks arguments
    parser.add_argument('--use-landmarks', action='store_true', help='Enable landmark conditioning')
    parser.add_argument('--landmark-cache-dir', type=str, default='landmark_cache', help='Cache directory')
    parser.add_argument('--landmark-dim', type=int, default=128, help='Landmark embedding dim')
    parser.add_argument('--landmark-dropout', type=float, default=0.1, help='Landmark dropout')

    return parser.parse_args()

def validate_model(model, classification_head, val_loader, device, use_landmarks=False):
    model.eval()
    classification_head.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in val_loader:
            if use_landmarks:
                images, landmarks, targets = batch
                images = images.to(device)
                landmarks = landmarks.to(device)
                embeddings = model(images, landmarks)
            else:
                images, targets = batch
                images = images.to(device)
                embeddings = model(images)
            
            targets = targets.to(device)
            targets_cpu = targets.cpu().numpy()
            all_targets.extend(targets_cpu)

            if isinstance(classification_head, torch.nn.Linear):
                outputs = classification_head(embeddings)
            else:
                outputs = classification_head(embeddings, targets)

            _, predicted_idx = torch.max(outputs.data, 1)
            all_predictions.extend(predicted_idx.cpu().numpy())

    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    accuracy = np.mean(all_predictions == all_targets)
    
    # Métricas com zero_division=0 para evitar warnings no início do treino
    precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    LOGGER.info(f'--- Validation Metrics ---')
    LOGGER.info(f'Accuracy: {accuracy:.4f}')
    LOGGER.info(f'Precision: {precision:.4f}')
    LOGGER.info(f'Recall: {recall:.4f}')
    LOGGER.info(f'F1-Score: {f1:.4f}')

    model.train()
    classification_head.train()
    return accuracy

def get_classification_head(classifier, embedding_dim, num_classes):
    classifiers = {
        'MCP': MarginCosineProduct(embedding_dim, num_classes),
        'AL': AngleLinear(embedding_dim, num_classes),
        'L': torch.nn.Linear(embedding_dim, num_classes, bias=False),
        'COS': CosFace(embedding_dim, num_classes, s=30.0, m=0.35)
    }
    if classifier not in classifiers:
        raise ValueError(f"Unsupported classifier: {classifier}")
    return classifiers[classifier]

def train_one_epoch(model, classification_head, criterion, optimizer, data_loader, device, epoch, params):
    model.train()
    losses = AverageMeter("Avg Loss", ":6.3f")
    accuracy_meter = AverageMeter("Accuracy", ":4.2f")
    start_time = time.time()
    
    for batch_idx, batch in enumerate(data_loader):
        # Desempacotamento dinâmico
        if params.use_landmarks:
            images, landmarks, target = batch
            images = images.to(device)
            landmarks = landmarks.to(device)
            target = target.to(device)
        else:
            images, target = batch
            images = images.to(device)
            target = target.to(device)

        optimizer.zero_grad()

        # Forward
        if params.use_landmarks:
            embeddings = model(images, landmarks)
        else:
            embeddings = model(images)

        # Loss calculation
        if isinstance(classification_head, torch.nn.Linear):
            output = classification_head(embeddings)
        else:
            output = classification_head(embeddings, target)

        loss = criterion(output, target)
        accuracy = calculate_accuracy(output, target)

        if params.distributed:
            reduced_loss = reduce_tensor(loss, params.world_size)
            accuracy = reduce_tensor(accuracy, params.world_size)
        else:
            reduced_loss = loss

        loss.backward()
        optimizer.step()

        losses.update(reduced_loss.item(), images.size(0))
        accuracy_meter.update(accuracy.item(), images.size(0))

        if device.type == 'cuda':
            torch.cuda.synchronize()

        if batch_idx % params.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            LOGGER.info(
                f'Epoch: [{epoch}/{params.epochs}][{batch_idx}/{len(data_loader)}] '
                f'Loss: {losses.avg:.3f}, Acc: {accuracy_meter.avg:.2f}%, LR: {lr:.5f}'
            )

    LOGGER.info(f'Epoch Summary: Loss: {losses.avg:.3f}, Acc: {accuracy_meter.avg:.2f}%, Time: {time.time()-start_time:.1f}s')

def main(params):
    init_distributed_mode(params)
    setup_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- LÓGICA DE LANDMARKS ---
    landmarks_dict = None
    if params.use_landmarks:
        try:
            from utils.landmark_annotator import LandmarkAnnotator
        except ImportError:
            # Fallback seguro para evitar travamento se o arquivo não estiver perfeito
            raise ImportError("utils.landmark_annotator não encontrado")

        LOGGER.info(f"Landmark conditioning ENABLED.")
        
        annotator = LandmarkAnnotator(cache_dir=params.landmark_cache_dir)
        dataset_name = os.path.basename(params.root.rstrip('/'))
        
        # Carrega o cache (NÃO EXECUTA DETECÇÃO - assume que prepare_data.py já rodou)
        # Se não encontrar o cache, vai falhar propositalmente para forçar o uso do prepare_data.py
        landmarks_dict = annotator.annotate_dataset(
            dataset_root=params.root,
            dataset_name=dataset_name,
            limit_fraction=params.dataset_fraction
        )
        
        if len(landmarks_dict) == 0:
             raise ValueError("Cache de landmarks vazio ou não encontrado! Rode 'python prepare_data.py' primeiro.")

        LOGGER.info(f"Landmarks loaded. Valid images: {len(landmarks_dict)}")
    # ---------------------------

    # Database config
    db_config = {'WebFace': 10572, 'VggFace2': 8631, 'MS1M': 85742}
    num_classes = db_config.get(params.database)
    if not num_classes: raise ValueError("Unsupported database")

    # Backbone
    if params.network == 'sphere20': backbone = sphere20(512, in_channels=3)
    elif params.network == 'sphere36': backbone = sphere36(512, in_channels=3)
    elif params.network == 'sphere64': backbone = sphere64(512, in_channels=3)
    elif params.network == "mobilenetv1": backbone = MobileNetV1(512)
    elif params.network == "mobilenetv2": backbone = MobileNetV2(512)
    elif params.network == "mobilenetv3_small": backbone = mobilenet_v3_small(512)
    elif params.network == "mobilenetv3_large": backbone = mobilenet_v3_large(512)
    elif params.network == "resnet50": backbone = create_resnet50(512, pretrained=True)
    else: raise ValueError("Unsupported network")

    # Wrapper do Modelo
    if params.use_landmarks:
        model = LandmarkConditionedModel(
            backbone=backbone,
            embedding_dim=512,
            num_landmarks=5,
            landmark_dim=params.landmark_dim,
            dropout=params.landmark_dropout
        )
    else:
        model = backbone

    model = model.to(device)
    if params.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params.local_rank])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    os.makedirs(params.save_path, exist_ok=True)
    # Configure file logging to persist training logs
    try:
        log_filename = os.path.join(params.save_path, f'{params.network}_{params.classifier}.log')
        add_file_handler(log_filename)
        LOGGER.info(f'Logging to file: {log_filename}')
    except Exception as e:
        LOGGER.info(f'Could not initialize file logging: {e}')
    classification_head = get_classification_head(params.classifier, 512, num_classes).to(device)

    # --- DATA AUGMENTATION CORRIGIDO ---
    # Removido RandomHorizontalFlip() pois os landmarks são estáticos (cache)
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),  <-- REMOVIDO PARA EVITAR DESALINHAMENTO
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    LOGGER.info('Loading training data.')
    full_dataset = ImageFolder(
        root=params.root, 
        transform=train_transform,
        use_landmarks=params.use_landmarks,
        landmarks_dict=landmarks_dict
    )

    # Filtro manual apenas se landmarks estiverem desligados
    if not params.use_landmarks and params.dataset_fraction < 1.0:
        total = len(full_dataset)
        subset = int(total * params.dataset_fraction)
        indices = np.random.permutation(total)[:subset]
        full_dataset.samples = [full_dataset.samples[i] for i in indices]
        LOGGER.info(f"Dataset limited to {subset} images (no landmarks).")

    train_dataset, val_dataset = create_validation_split(full_dataset, val_split=0.1)
    
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, sampler=None, shuffle=True, num_workers=params.num_workers, pin_memory=True)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
        {'params': classification_head.parameters()}
    ], lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)

    if params.lr_scheduler == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)

    best_accuracy = 0.0
    early_stopping = EarlyStopping(patience=10)

    LOGGER.info(f'Starting training: {params.network} + {params.classifier}')
    
    for epoch in range(params.epochs):
        train_one_epoch(model, classification_head, criterion, optimizer, train_loader, device, epoch, params)
        lr_scheduler.step()

        checkpoint = {
            'epoch': epoch + 1,
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': params
        }
        save_on_master(checkpoint, os.path.join(params.save_path, f'{params.network}_{params.classifier}_last.ckpt'))

        if params.local_rank == 0:
            val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
            acc = validate_model(model_without_ddp, classification_head, val_loader, device, params.use_landmarks)
            
            if acc > best_accuracy:
                best_accuracy = acc
                save_on_master(checkpoint, os.path.join(params.save_path, f'{params.network}_{params.classifier}_best.ckpt'))
                LOGGER.info(f"New best accuracy: {best_accuracy:.4f}")

            if early_stopping(epoch, acc):
                break

    LOGGER.info('Training completed.')

if __name__ == '__main__':
    args = parse_arguments()
    main(args)