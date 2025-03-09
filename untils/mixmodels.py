import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableEnsemble(nn.Module):
    def __init__(self, base_model_class, model_paths, num_classes, device,
                 ensemble_method='attention',
                 hidden_size=128, dropout_rate=0.3):
        super().__init__()
        self.models = self.models = nn.ModuleList()
        self.device = device
        self.ensemble_method = ensemble_method
        self.num_classes = num_classes
        self.num_models = len(model_paths)
        self.warmup_steps = 1000
        self.current_step = 0
        self.residual_connection = True
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.temp_min = 0.1
        self.temp_max = 10.0
        self.dropout_rate = dropout_rate

        self.weight_scale = nn.Parameter(torch.ones(1))

        for path in model_paths:
            model = base_model_class(num_classes=num_classes).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            for param in model.parameters():

                param.requires_grad = True
            self.models.append(model)


        if ensemble_method == 'attention':
            self.attention_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(num_classes, hidden_size // (2 ** i)),
                    nn.ReLU(),
                    nn.Linear(hidden_size // (2 ** i), 1)
                ) for i in range(3)
            ])
            self.attention_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):
        if self.ensemble_method == 'attention':
            return self._attention_ensemble(x)

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def _attention_ensemble(self, x):

        outputs = []
        multi_scale_attention = []

        for model in self.models:
            output = model(x)

            scaled_output = output / (self.temperature.clamp(self.temp_min, self.temp_max))
            outputs.append(F.softmax(scaled_output, dim=1))


            scale_scores = []
            for attention_layer in self.attention_layers:
                score = attention_layer(scaled_output)
                scale_scores.append(score)

            attention_weights = F.softmax(self.attention_weights, dim=0)
            combined_score = sum(w * s for w, s in zip(attention_weights, scale_scores))
            multi_scale_attention.append(combined_score)


        attention_weights = F.softmax(torch.cat(multi_scale_attention, dim=1), dim=1)
        attention_weights = attention_weights * self.weight_scale


        ensemble_output = torch.zeros_like(outputs[0])
        if self.residual_connection:
            base_output = sum(outputs) / len(outputs)

        for i, output in enumerate(outputs):
            ensemble_output += output * attention_weights[:, i:i + 1]
            dropout_layer1 = nn.Dropout(self.dropout_rate)
            dropout_layer2 = nn.Dropout(self.dropout_rate)

            ensemble_output = dropout_layer1(ensemble_output)
            ensemble_output = dropout_layer2(ensemble_output)

        if self.residual_connection:
            ensemble_output = 0.7 * ensemble_output + 0.3 * base_output

        return ensemble_output

