import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from models import (HybridModel,CNN,STCNN,OCNN,RSCNN_Parallel, RSCNN_Sequential,RST,
                    BCNN,create_resnet18,BasicCNN,SpectralCNN2D,create_improved_resnet18,ViTForOCT)
import argparse
import csv
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from pathlib import Path
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold
import pandas as pd
from untils.dataset import GrayscaleDataset
from untils.stop import EarlyStopping
from untils.loss import calculate_metrics
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='Train the classification model')
    parser.add_argument('--json_path', type=str, default=r'json\combined_dataset.json', help='Path to JSON data file')
    parser.add_argument('--output_dir', type=str, default='runs')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--initial_lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', type=str, default=r'', help='Pretrained model path')
    parser.add_argument('--model_type', type=str, default='RSCNN_Parallel',
                        choices=[ 'RSCNN_Parallel'],
                        help='Model type')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for cross-validation')  #
    return parser.parse_args()
def print_training_info(args):
    print("\n" + "="*50)
    print("Training Configuration Information:")
    print("="*50)
    print(f"Model Type: {args.model_type}")
    print(f"Data Directory: {args.json_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Image Size: {args.img_size}x{args.img_size}")
    print(f"Input Channels: {args.channels}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Device: {args.device}")
    print(f"Initial Learning Rate: {args.initial_lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Random Seed: {args.seed}")
    print(f"Cross-validation Folds: {args.n_splits}")
    print(f"Pretrained Model: {'Using ' + args.pretrained if args.pretrained else 'Not using'}")

    print("="*50 + "\n")


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epoch):
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    train_loss = 0
    train_outputs_all = []
    train_targets_all = []

    for images, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1} Training'):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        train_outputs_all.append(outputs)
        train_targets_all.append(targets)

    if scheduler is not None:
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr:.6f}')

    train_outputs_all = torch.cat(train_outputs_all)
    train_targets_all = torch.cat(train_targets_all)
    train_metrics = calculate_metrics(train_outputs_all, train_targets_all)
    train_metrics['untils'] = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    val_outputs_all = []
    val_targets_all = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f'Epoch {epoch + 1} Validation'):
            images, targets = images.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_outputs_all.append(outputs)
            val_targets_all.append(targets)

    val_outputs_all = torch.cat(val_outputs_all)
    val_targets_all = torch.cat(val_targets_all)
    val_metrics = calculate_metrics(val_outputs_all, val_targets_all)
    val_metrics['untils'] = val_loss / len(val_loader)

    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    return train_metrics, val_metrics


def plot_learning_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def get_unique_path(path):
    unique_path = path
    count = 1
    while os.path.exists(unique_path):
        unique_path = f"{path}_{count}"
        count += 1
    return unique_path

def main(args):
    print_training_info(args)
    set_seed(args.seed)




    model_output_dir = Path(args.output_dir) / f'train_{args.model_type}'
    model_output_dir = get_unique_path(str(model_output_dir))
    model_output_dir = Path(model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    for data_idx in range(1, 10):
        data_key = f'data{data_idx}'
        print(f'\nProcessing {data_key}...')


        data_output_dir = model_output_dir / data_key
        data_output_dir.mkdir(parents=True, exist_ok=True)

        dataset = GrayscaleDataset(args.json_path, args.img_size, data_key)

        # K-Fold Cross-Validation
        kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f'Fold {fold + 1}/5')


            fold_output_dir = data_output_dir / f'fold_{fold + 1}'
            fold_output_dir.mkdir(parents=True, exist_ok=True)

            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            # Define transforms
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            val_transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            train_subset.dataset.transform = train_transform
            val_subset.dataset.transform = val_transform

            train_loader = DataLoader(
                train_subset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True
            )

            # Model initialization
            if args.model_type == 'rscnn_parallel':
                model = RSCNN_Parallel(num_classes=args.num_classes).to(device)



            if args.pretrained:
                print(f"Loading pretrained model from {args.pretrained}")
                model.load_state_dict(torch.load(args.pretrained))
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.initial_lr,
                epochs=args.epochs,
                steps_per_epoch=len(train_loader)
            )

            metrics_file = fold_output_dir / 'metrics.csv'
            train_losses = []
            val_losses = []
            best_val_acc = 0

            with open(metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'train_acc', 'train_auc',
                    'train_sensitivity', 'train_specificity', 'train_f1',
                    'val_loss', 'val_acc', 'val_auc',
                    'val_sensitivity', 'val_specificity', 'val_f1'
                ])

            for epoch in range(args.epochs):
                train_metrics, val_metrics = train_and_validate(
                    model, train_loader, val_loader,
                    criterion, optimizer, scheduler, device, epoch
                )

                train_losses.append(train_metrics['untils'])
                val_losses.append(val_metrics['untils'])

                # Save metrics
                with open(metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, train_metrics['untils'], train_metrics['accuracy'],
                        train_metrics['auc'], train_metrics['sensitivity'],
                        train_metrics['specificity'], train_metrics['f1'],
                        val_metrics['untils'], val_metrics['accuracy'],
                        val_metrics['auc'], val_metrics['sensitivity'],
                        val_metrics['specificity'], val_metrics['f1']
                    ])

                # Save best model
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    torch.save(model.state_dict(), fold_output_dir / 'best.pt')

                # Save last model
                torch.save(model.state_dict(), fold_output_dir / 'last.pt')

                # Plot learning curves
                plot_learning_curves(
                    train_losses,
                    val_losses,
                    fold_output_dir / 'learning_curves.png'
                )


if __name__ == '__main__':
    args = parse_args()
    main(args)
