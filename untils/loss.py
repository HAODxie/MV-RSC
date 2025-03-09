import torch
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
def calculate_metrics(outputs, targets):
    with torch.no_grad():
        outputs = outputs.detach()
        targets = targets.detach()
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        accuracy = metrics.accuracy_score(targets, preds)
        sensitivity = metrics.recall_score(targets, preds, average='macro')
        precision = metrics.precision_score(targets, preds, average='macro')
        f1 = metrics.f1_score(targets, preds, average='macro')
        auc_scores = []
        for i in range(outputs.shape[1]):
            true_binary = (targets == i).astype(int)
            auc_scores.append(metrics.roc_auc_score(true_binary, probs[:, i]))
        auc = np.mean(auc_scores)
        specificity = 0
        for i in range(outputs.shape[1]):
            true_neg = np.sum((targets != i) & (preds != i))
            total_neg = np.sum(targets != i)
            specificity += true_neg / total_neg if total_neg > 0 else 0
        specificity /= outputs.shape[1]
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'auc': auc
        }


class EnsembleLoss(nn.Module):
    def __init__(self, ensemble_method, base_criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.ensemble_method = ensemble_method
        self.base_criterion = base_criterion

    def forward(self, outputs, targets, model):
        base_loss = self.base_criterion(outputs, targets)

        if self.ensemble_method == 'diverse_ensemble':

            diversity_weight = 0.1
            diversity_loss = model.get_diversity_loss(outputs)
            return base_loss + diversity_weight * diversity_loss

        elif self.ensemble_method == 'mixture_of_experts':

            sparsity_weight = 0.01
            entropy_weight = 0.01


            expert_gates = model.expert_gates(outputs)


            sparsity_loss = -torch.mean(torch.sum(expert_gates * torch.log(expert_gates + 1e-6), dim=1))


            expert_outputs = [model.expert_value(output) for output in outputs]
            diversity_loss = 0
            for i in range(len(expert_outputs)):
                for j in range(i + 1, len(expert_outputs)):
                    diversity_loss += torch.mean(F.kl_div(
                        F.log_softmax(expert_outputs[i], dim=1),
                        F.softmax(expert_outputs[j], dim=1),
                        reduction='batchmean'
                    ))

            return base_loss + sparsity_weight * sparsity_loss + entropy_weight * diversity_loss

        elif self.ensemble_method == 'attention':

            attention_weight = 0.01
            attention_scores = model.attention(outputs)
            attention_entropy = -torch.mean(torch.sum(attention_scores * torch.log(attention_scores + 1e-6), dim=1))
            return base_loss + attention_weight * attention_entropy

        elif self.ensemble_method == 'boosting':

            if model.sample_weights is not None:
                return torch.mean(base_loss * model.sample_weights)
            return base_loss

        return base_loss