import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from models import (HybridModel,CNN,STCNN,OCNN,RSCNN_Parallel, RSCNN_Sequential,RST,
                    BCNN,create_resnet18,BasicCNN,SpectralCNN2D,create_improved_resnet18,ViTForOCT)
from untils.dataset import GrayscaleDataset
from untils.loss import calculate_metrics
import argparse
from PIL import Image
from tqdm import tqdm
from untils.mixmodels import LearnableEnsemble
from sklearn.metrics import roc_curve
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Test the classification model')
    parser.add_argument('--json_path', type=str, default='json/test.json', help='Path to data directory containing JSON')
    parser.add_argument('--models_dir', type=str, default='runs/train_3', help='Path to saved models directory')
    parser.add_argument('--output_dir', type=str, default='runs/detect', help='Path to save output CSV files')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--weights', type=str, default=r'',
                        help='Path to the best model weights')
    parser.add_argument('--model_type', type=str, default='RSCNN_Parallel',
                        choices=[ 'RSCNN_Parallel'],
                        help='Choose model type')
    parser.add_argument('--ensemble_method', type=str, default='',
                        choices=['attention' ],
                        help='Choose ensemble_method type')
    return parser.parse_args()



def load_ensemble_model(args, base_model):
    model_paths = []
    base_dir = Path(args.models_dir) / f'train_{args.model_type}'

    for data_idx in range(1, 10):
        model_path = base_dir / f'data{data_idx}' / 'fold_1' / 'best.pt'
        if model_path.exists():
            model_paths.append(model_path)

    ensemble = LearnableEnsemble(
        base_model_class=base_model,
        model_paths=model_paths,
        num_classes=args.num_classes,
        device=args.device,

        ensemble_method=args.ensemble_method
    )
    return ensemble


def test(model, test_loader, device, output_dir):
    model.eval()



    results_file = Path(output_dir) / 'test_results.csv'
    roc_data_file = output_dir / 'roc_data.csv'
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'true_label', 'predicted_label', 'confidence'])

    all_outputs = []
    all_targets = []
    all_img_names = []

    all_probs = []
    all_true_labels = []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc='Testing')):
            images, targets = images.to(device), targets.to(device)


            start_index = batch_idx * test_loader.batch_size
            img_names = [test_loader.dataset.data[i]['image'] for i in range(start_index, start_index + len(images))]

            with torch.cuda.amp.autocast():
                outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)
            preds = preds.cpu().numpy()
            confidences = confidences.cpu().numpy()
            all_probs.extend(probs.cpu().numpy())
            all_true_labels.extend(targets.cpu().numpy())

            all_outputs.append(outputs)
            all_targets.append(targets)
            all_img_names.extend(img_names)

            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for i in range(len(img_names)):
                    writer.writerow([
                        img_names[i],
                        targets[i].cpu().numpy(),
                        preds[i],
                        f'{confidences[i]:.4f}'
                    ])

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    metrics_results = calculate_metrics(all_outputs, all_targets)


    with open(Path(output_dir) / 'average_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for metric, value in metrics_results.items():
            writer.writerow([metric, f'{value:.4f}'])


    all_probs = np.array(all_probs)
    all_true_labels = np.array(all_true_labels)


    fpr, tpr, thresholds = roc_curve(all_true_labels, all_probs[:, 1])

    fpr_fixed = np.linspace(0, 1.0, num=6)
    tpr_fixed = np.interp(fpr_fixed, fpr, tpr)


    with open(roc_data_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['FPR', 'TPR'])
        for fpr_val, tpr_val in zip(fpr_fixed, tpr_fixed):
            writer.writerow([f'{fpr_val:.1f}', f'{tpr_val:.4f}'])

    print("\nAverage Test Metrics:")
    for metric, value in metrics_results.items():
        print(f"{metric}: {value:.4f}")

    return metrics_results
def get_unique_path(path):
    unique_path = path
    count = 1
    while os.path.exists(unique_path):
        unique_path = f"{path}_{count}"
        count += 1
    return unique_path

def main():
    args = parse_args()
    output_dir = Path(args.output_dir) / f'detect_{args.model_type}'/ f'train_{args.ensemble_method}'
    output_dir = get_unique_path(str(output_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if  args.model_type == 'rscnn_parallel':
        base_model = RSCNN_Parallel


    ensemble_model = load_ensemble_model(args, base_model).to(device)

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


    test_dataset = GrayscaleDataset(args.json_path, args.img_size, 'test', transform=test_transform)


    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers,
                             pin_memory=True)


    metrics_results = test(ensemble_model, test_loader, args.device,output_dir)


if __name__ == '__main__':
    main()
