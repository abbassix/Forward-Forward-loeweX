import math
from collections import defaultdict
from itertools import combinations

import torch
import torch.nn as nn

from src import utils


class FF_model(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(FF_model, self).__init__()

        self.opt = opt
        self.num_channels = [self.opt.model.hidden_dim] * self.opt.model.num_layers
        self.act_fn = ReLU_full_grad()

        # Initialize the model.
        self.model = nn.ModuleList([nn.Linear(784, self.num_channels[0])])
        for i in range(1, len(self.num_channels)):
            self.model.append(nn.Linear(self.num_channels[i - 1], self.num_channels[i]))

        # Initialize forward-forward loss.
        self.ff_loss = nn.BCEWithLogitsLoss()

        # Initialize peer normalization loss.
        self.running_means = [
            torch.zeros(self.num_channels[i], device=self.opt.device) + 0.5
            for i in range(self.opt.model.num_layers)
        ]

        # Initialize downstream classification loss.
        channels_for_classification_loss = sum(
            self.num_channels[-i] for i in range(self.opt.model.num_layers - 1)
        )
        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, 10, bias=False)
        )
        self.classification_loss = nn.CrossEntropyLoss()

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(
                    m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0])
                )
                torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    def _calc_peer_normalization_loss(self, idx, z):
        # Only calculate mean activity over positive samples.
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim=0)

        self.running_means[idx] = self.running_means[
            idx
        ].detach() * self.opt.model.momentum + mean_activity * (
            1 - self.opt.model.momentum
        )

        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1)

        logits = sum_of_squares - z.shape[1]
        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels)
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def _calc_goodness(self, z):
        """Calculate goodness (sum of squared activations) for FF-native validation."""
        return torch.sum(z ** 2, dim=-1)

    def forward(self, inputs, labels):
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization Loss": torch.zeros(1, device=self.opt.device),  # Peer Normalization
            "Binary Losses": {},
            "Binary Accuracies": {},
        }

        # Concatenate positive and negative samples and create corresponding labels.
        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device)
        posneg_labels[: self.opt.input.batch_size] = 1

        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        for idx, layer in enumerate(self.model):
            z = layer(z)
            z = self.act_fn.apply(z)

            if self.opt.model.peer_normalization > 0:
                peer_loss = self._calc_peer_normalization_loss(idx, z)
                scalar_outputs["Peer Normalization Loss"] += peer_loss
                scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss

            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            scalar_outputs["Binary Losses"][f"Layer {idx}"] = ff_loss
            scalar_outputs["Binary Accuracies"][f"Layer {idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            z = z.detach()

            z = self._layer_norm(z)

        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        return scalar_outputs

    def forward_ff_native_validation(self, inputs, labels):
        """
        FF-native validation: Try all possible labels and choose the one with highest goodness.
        Calculates FF accuracy for all 7 combinations of layers: l0, l1, l2, l0+l1, l0+l2, l1+l2, l0+l1+l2
        """
        scalar_outputs = {
            "Mode": "FF-native",
            "Accuracies": {},
        }

        batch_size = inputs["neutral_sample"].shape[0]
        num_classes = 10
        
        # Get the neutral sample (without any label information)
        neutral_sample = inputs["neutral_sample"]
        
        # Store goodness for each layer and each possible label
        goodnesses = defaultdict(list)
        
        for class_label in range(num_classes):
            # Create one-hot label for this class; shape: [num_classes]
            one_hot_label = torch.nn.functional.one_hot(
                torch.tensor(class_label), num_classes=num_classes
            ).float().to(self.opt.device)
            
            # Replicate for batch size; shape: [batch_size, num_classes]
            batch_one_hot = one_hot_label.unsqueeze(0).repeat(batch_size, 1)
            
            # Create labeled sample by setting the first pixels to the one-hot label
            labeled_sample = neutral_sample.clone() # Shape: [batch_size, 28, 28]
            # Set the first row of pixels (first 10 pixels) to the one-hot label
            labeled_sample[:, :, 0, :num_classes] = batch_one_hot.unsqueeze(1)
            
            # Forward pass through the network
            z = labeled_sample.reshape(batch_size, -1)
            z = self._layer_norm(z)
            
            with torch.no_grad():
                for idx, layer in enumerate(self.model):
                    z = layer(z)
                    z = self.act_fn.apply(z)
                    
                    # Calculate and store goodness for this layer
                    goodness = self._calc_goodness(z)
                    goodnesses[idx].append(goodness)
                    
                    z = self._layer_norm(z)
        
        # Stack all goodness values: [num_classes, batch_size]
        for idx in range(len(self.model)):
            goodnesses[idx] = torch.stack(goodnesses[idx], dim=0)
        
        # Calculate combined goodness using combinatorics
        combined_goodness = {}
        layer_indices = list(range(len(self.model)))
        for r in range(1, len(layer_indices) + 1):
            for combo in combinations(layer_indices, r):
                combo_key = '_'.join([f'l{i}' for i in combo])
                combined_goodness[combo_key] = sum(goodnesses[i] for i in combo)
        
        # Calculate accuracy for each combination
        true_labels = labels["class_labels"]
        
        for combo_key, goodness in combined_goodness.items():
            predicted_labels = torch.argmax(goodness, dim=0)
            accuracy = (predicted_labels == true_labels).float().mean().item()
            scalar_outputs["Accuracies"][f'{combo_key}'] = accuracy
        
        # Use the accuracy from the combination of all layers as the main metric
        scalar_outputs["Accuracy"] = scalar_outputs["Accuracies"][f'l{"_l".join(map(str, layer_indices))}']
        
        return scalar_outputs

    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Mode": "Classifier head",
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z = inputs["neutral_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        input_classification_model = []

        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn.apply(z)
                z = self._layer_norm(z)

                if idx >= 1:
                    input_classification_model.append(z)

        input_classification_model = torch.concat(input_classification_model, dim=-1)

        output = self.linear_classifier(input_classification_model.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0]
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = utils.get_accuracy(
            self.opt, output.data, labels["class_labels"]
        )

        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["Classification Loss"] = classification_loss
        scalar_outputs["Accuracy"] = classification_accuracy
        return scalar_outputs


class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
