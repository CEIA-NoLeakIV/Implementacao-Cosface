# [FINAL] train_resnet_tuned.py
import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# --- Novos Imports ---
import evaluate_tta_tuned as evaluate  # Usa o novo script de validação
from utils.training_tracker import TrainingTracker
from utils.face_validation import FaceValidator
# ---------------------

from utils.dataset import ImageFolder
from utils.metrics import MarginCosineProduct, AngleLinear, CosFace
from utils.general import (
    setup_seed, reduce_tensor, save_on_master, calculate_accuracy, 
    init_distributed_mode, AverageMeter, EarlyStopping, LOGGER
)
from models import (
    sphere20, sphere36, sphere64, MobileNetV1, MobileNetV2, 
    mobilenet_v3_small, mobilenet_v3_large, create_resnet50
)
from utils.validation_split import create_validation_split

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training Face Recognition with ResNet/CosFace + Tracker")
    # ... (Argumentos originais mantidos: root, database, network, classifier, etc) ...
    parser.add_argument('--root', type=str, default='data/train/webface_112x112/')
    parser.add_argument('--database', type=str, default='WebFace', choices=['WebFace', 'VggFace2', "MS1M", "VggFaceHQ", "vggHQcropadoSimilaridade"])
    parser.add_argument('--network', type=str, default='resnet50', choices=['resnet50', 'sphere20', 'mobilenetv1']) # simplificado
    parser.add_argument('--classifier', type=str, default='COS', choices=['ARC', 'MCP', 'COS'])
    
    # Hyperparams padrão
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save-path', type=str, default='weights')
    parser.add_argument("--checkpoint", type=str, default=None)
    
    # --- Novos Argumentos para Tracker e Validação ---
    parser.add_argument('--val-dataset', type=str, default='lfw', choices=['lfw', 'celeba', 'audit_log', 'custom'], help="Dataset for external validation")
    parser.add_argument('--val-root', type=str, default='data/lfw/val', help="Root path for external validation dataset")
    parser.add_argument('--use-face-validation', action='store_true', help="Enable RetinaFace cleaning during validation")
    parser.add_argument('--no-face-policy', type=str, default='exclude', choices=['exclude', 'include'])
    parser.add_argument('--metrics-dir', type=str, default='metrics', help="Directory to save tracker plots and logs")
    # -------------------------------------------------

    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--use-deterministic-algorithms", action="store_true")
    
    # Add scheduler args manually inside logic or keep explicit args
    parser.add_argument('--lr-scheduler', default='MultiStepLR')
    parser.add_argument('--milestones', nargs='+', type=int, default=[10, 20, 25])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--step-size', type=int, default=10)

    return parser.parse_args()

def validate_model_internal(model, classification_head, val_loader, device):
    """Validação interna (Train Subset) - Retorna loss e accuracy"""
    model.eval()
    classification_head.eval()
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            embeddings = model(images)
            if isinstance(classification_head, torch.nn.Linear):
                outputs = classification_head(embeddings)
            else:
                outputs = classification_head(embeddings, targets)
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
            
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return accuracy

def get_classification_head(classifier, embedding_dim, num_classes):
    classifiers = {
        'MCP': MarginCosineProduct(embedding_dim, num_classes),
        'AL': AngleLinear(embedding_dim, num_classes),
        'L': torch.nn.Linear(embedding_dim, num_classes, bias=False),
        'COS': CosFace(embedding_dim, num_classes, s=30.0, m=0.35)
    }
    return classifiers[classifier]

def train_one_epoch(model, classification_head, criterion, optimizer, data_loader, device, epoch, params):
    model.train()
    losses = AverageMeter("Avg Loss", ":6.3f")
    accuracy_meter = AverageMeter("Accuracy", ":4.2f")
    batch_time = AverageMeter("Batch Time", ":4.3f")
    
    start_time = time.time()
    for batch_idx, (images, target) in enumerate(data_loader):
        images, target = images.to(device), target.to(device)
        optimizer.zero_grad()
        
        embeddings = model(images)
        output = classification_head(embeddings) if isinstance(classification_head, torch.nn.Linear) else classification_head(embeddings, target)
        
        loss = criterion(output, target)
        acc = calculate_accuracy(output, target)
        
        if params.distributed:
            loss = reduce_tensor(loss, params.world_size)
            acc = reduce_tensor(acc, params.world_size)
            
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        accuracy_meter.update(acc.item(), images.size(0))
        batch_time.update(time.time() - start_time)
        start_time = time.time()
        
        if batch_idx % params.print_freq == 0:
            LOGGER.info(f'Epoch: [{epoch}][{batch_idx}/{len(data_loader)}] Loss: {losses.avg:.4f} Acc: {accuracy_meter.avg:.2f}')
            
    return losses.avg, accuracy_meter.avg, batch_time.sum

def main(params):
    init_distributed_mode(params)
    setup_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Inicializar Tracker e Validator ---
    if params.local_rank == 0:
        tracker = TrainingTracker(save_dir=params.metrics_dir, experiment_name=f"{params.network}_{params.classifier}")
        face_validator = None
        if params.use_face_validation:
            LOGGER.info("Initializing FaceValidator...")
            face_validator = FaceValidator()
            tracker.set_face_validation_policy(params.no_face_policy)
    # ------------------------------------------

    # Configuração do Modelo e Dataset (Mantido do original)
    db_config = {'WebFace': {'num_classes': 10572}, 'VggFace2': {'num_classes': 8631}, 'MS1M': {'num_classes': 85742}, 'VggFaceHQ': {'num_classes': 9132}, 'vggHQcropadoSimilaridade': {'num_classes': 14128}}
    num_classes = db_config[params.database]['num_classes']
    
    if params.network == "resnet50":
        model = create_resnet50(embedding_dim=512, pretrained=True)
    elif params.network == "sphere20":
        model = sphere20(embedding_dim=512)
    # ... outros modelos ...
    else:
        model = create_resnet50(embedding_dim=512) # Fallback

    model = model.to(device)
    model_without_ddp = model.module if params.distributed else model
    
    classification_head = get_classification_head(params.classifier, 512, num_classes).to(device)
    
    # DataLoader e Transforms (Mantido)
    train_transform = transforms.Compose([transforms.Resize((112, 112)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    full_dataset = ImageFolder(root=params.root, transform=train_transform)
    train_dataset, val_dataset = create_validation_split(full_dataset, val_split=0.1)
    
    # Correção do transform de validação interna
    val_dataset.dataset.transform = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if params.distributed else torch.utils.data.RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, sampler=train_sampler, num_workers=params.num_workers, pin_memory=True)
    # Validação interna Loader
    val_loader_internal = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': classification_head.parameters()}], lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
    
    # Scheduler logic (simplified)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)

    # Resume Logic
    start_epoch = 0
    if params.checkpoint and os.path.exists(params.checkpoint):
        ckpt = torch.load(params.checkpoint)
        model_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        # Recuperar histórico do tracker se disponível
        if params.local_rank == 0:
            tracker.load_history_from_checkpoint(params.checkpoint)

    best_accuracy = 0.0
    early_stopping = EarlyStopping(patience=10)

    # LOOP DE TREINO
    for epoch in range(start_epoch, params.epochs):
        if params.distributed: train_sampler.set_epoch(epoch)
        
        # Treino
        loss_train, acc_train, epoch_time = train_one_epoch(model, classification_head, criterion, optimizer, train_loader, device, epoch, params)
        lr_scheduler.step()
        
        if params.local_rank == 0:
            # Validação Interna
            acc_val_internal = validate_model_internal(model_without_ddp, classification_head, val_loader_internal, device)
            
            # Validação Externa (Com Tracker e FaceValidator)
            LOGGER.info(f"Evaluating on {params.val_dataset}...")
            _, _, ext_metrics = evaluate.eval(
                model_without_ddp, 
                device=device, 
                val_dataset=params.val_dataset,
                val_root=params.val_root,
                face_validator=face_validator if params.use_face_validation else None,
                no_face_policy=params.no_face_policy
            )
            
            # --- LOG NO TRACKER ---
            curr_accuracy = ext_metrics.get('accuracy', 0.0)
            tracker.log_epoch(
                epoch=epoch+1,
                train_loss=loss_train,
                train_accuracy=acc_train,
                val_accuracy=acc_val_internal, # Validação interna do split
                external_metrics=ext_metrics,  # LFW/CelebA metrics ricas
                learning_rate=optimizer.param_groups[0]['lr'],
                epoch_time=epoch_time
            )
            
            # Salvar Checkpoint com Histórico
            checkpoint = {
                'epoch': epoch + 1,
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': params
            }
            tracker.save_checkpoint_with_history(checkpoint, os.path.join(params.save_path, f'{params.network}_{params.classifier}_last.ckpt'))

            # Salvar Best Model (Baseado na validação externa escolhida)
            if curr_accuracy > best_accuracy:
                best_accuracy = curr_accuracy
                tracker.save_checkpoint_with_history(checkpoint, os.path.join(params.save_path, f'{params.network}_{params.classifier}_best.ckpt'))
                LOGGER.info(f"New best {params.val_dataset} accuracy: {best_accuracy:.4f}")

            if early_stopping(epoch, curr_accuracy):
                break

    if params.local_rank == 0:
        tracker.generate_final_report()
        LOGGER.info("Training Finished.")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
