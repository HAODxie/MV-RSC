import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from models import (HybridModel,CNN,STCNN,OCNN,RSCNN_Parallel, RSCNN_Sequential,RST,
                    BCNN,create_resnet18,BasicCNN,SpectralCNN2D,create_improved_resnet18,ViTForOCT)
import argparse
import csv
from tqdm import tqdm
from pathlib import Path
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from untils.loss import calculate_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Test the classification model')
    parser.add_argument('--data_dir', type=str, default='./data/plaque/test/dataset1')

    parser.add_argument('--output_dir', type=str, default='runs/detect_3')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--weights', type=str, default=r'', help='Path to the best model weights')
    parser.add_argument('--model_type', type=str, default='RSCNN_Parallel',
                        choices=[ 'RSCNN_Parallel'],
                        help='Model type')
    return parser.parse_args()


class TestDataset(Dataset):
    def __init__(self, root_dir, img_size):
        self.root_dir = root_dir
        self.img_size = img_size

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls_name], img_name))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, img_name = self.samples[idx]
        img = Image.open(img_path).convert('L')
        img = self.transform(img)
        return img, label, img_name


def test(model, test_loader, device, output_dir):
    model.eval()

    # Create results CSV file
    results_file = output_dir / 'test_results.csv'
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
        for images, targets, img_names in tqdm(test_loader, desc='Testing'):
            images, targets = images.to(device), targets.to(device)

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

            # Save individual results
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

    with open(output_dir / 'average_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for metric, value in metrics_results.items():
            writer.writerow([metric, f'{value:.4f}'])


    all_probs = np.array(all_probs)
    all_true_labels = np.array(all_true_labels)


    fpr, tpr, thresholds = roc_curve(all_true_labels, all_probs[:, 1])

    fpr_fixed = np.linspace(0, 1.0, num=200)
    tpr_fixed = np.interp(fpr_fixed, fpr, tpr)

    with open(roc_data_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['FPR', 'TPR'])
        for fpr_val, tpr_val in zip(fpr_fixed, tpr_fixed):
            writer.writerow([f'{fpr_val:.2f}', f'{tpr_val:.4f}'])

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
def main(args):
    # Create output directory
    output_dir = Path(args.output_dir) / f'detect_{args.model_type}'
    output_dir = get_unique_path(str(output_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Initialize model

    if args.model_type == 'rscnn_parallel':
        model = RSCNN_Parallel(num_classes=args.num_classes).to(device)


    # Load model weights
    model.load_state_dict(torch.load(args.weights))

    # Prepare test dataset and dataloader
    test_dataset = TestDataset(args.data_dir, args.img_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers,
                             pin_memory=True)

    print("\nTest Configuration:")
    print(f"Model Type: {args.model_type}")
    print(f"Test Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Model Weights: {args.weights}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Class mapping: {test_dataset.class_to_idx}\n")

    # Run testing
    test_metrics = test(model, test_loader, device, output_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)