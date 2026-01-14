
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, GATv2Conv, SAGEConv
from torch.utils.data import DataLoader, Dataset
import os
import torch
from torch.nn import functional as F
from torch.nn import Linear
from torch_geometric.transforms import NormalizeFeatures
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoTokenizer
import numpy as np

######### MODELS #########
## MLP
class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=64, dropout=0.5):
        super().__init__()
        
        self.lin1 = Linear(num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, data):

        x = data.x

        x = self.lin1(x)
        x = x.relu()
        if self.dropout != None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x.view(-1)  # For binary classification

## GCN
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=None, n_layers=2):
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        
        if n_layers == 1:
            self.layers.append(GCNConv(num_features, num_classes))  # Single-layer model
        elif n_layers == 2:
            self.layers.append(GCNConv(num_features, hidden_channels))  # First layer
            self.layers.append(GCNConv(hidden_channels, num_classes))  # Last layer
        else:
            self.layers.append(GCNConv(num_features, hidden_channels))  # First layer
            for _ in range(1, n_layers - 1):
                self.layers.append(GCNConv(hidden_channels, hidden_channels))  # Middle layers
            self.layers.append(GCNConv(hidden_channels, num_classes))  # Last layer

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        if len(self.layers) == 1:  # Handle single-layer case
            x = self.layers[0](x, edge_index, edge_weight=edge_weight)
        else:
            for layer in self.layers[:-1]:
                x = layer(x, edge_index, edge_weight=edge_weight)
                x = x.relu()
                if self.dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.layers[-1](x, edge_index, edge_weight=edge_weight)  # Last layer
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x.view(-1)  # For binary classification


## GAT
class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, heads, edge_dim, dropout, n_layers=2, residual=False, bias=False):
        super().__init__()

        self.layers = torch.nn.ModuleList()

        if n_layers == 1:
            self.layers.append(GATv2Conv(in_channels=num_features, out_channels = num_classes, heads=heads, \
                                         edge_dim=edge_dim, concat=False, dropout=dropout, add_self_loops=True, residual=residual, bias=bias))  # Single-layer model
        elif n_layers == 2:
            self.layers.append(GATv2Conv(in_channels=num_features, out_channels = hidden_channels, heads=heads, \
                                         edge_dim=edge_dim, concat=True, dropout=dropout, add_self_loops=True, residual=residual, bias=bias))  # First layer
            self.layers.append(GATv2Conv(in_channels=hidden_channels*heads, out_channels=num_classes, heads=heads, \
                                         edge_dim=edge_dim, concat=False, dropout=dropout, add_self_loops=True, residual=residual, bias=bias))  # Last layer
        else:
            self.layers.append(GATv2Conv(in_channels=num_features, out_channels = hidden_channels, heads=heads, \
                                         edge_dim=edge_dim, concat=True, dropout=dropout, add_self_loops=True, residual=residual, bias=bias))  # First layer
            for _ in range(1, n_layers - 1):
                self.layers.append(GATv2Conv(in_channels=hidden_channels*heads, out_channels=hidden_channels, heads=heads, \
                                             edge_dim=edge_dim, concat=True, dropout=dropout, add_self_loops=True, residual=residual, bias=bias))  # Middle layers
            self.layers.append(GATv2Conv(in_channels=hidden_channels*heads, out_channels=num_classes, heads=heads, \
                                         edge_dim=edge_dim, concat=False, dropout=dropout, add_self_loops=True, residual=residual, bias=bias))  # Last layer


    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr#torch.cat((data.edge_weight.unsqueeze(1), data.edge_attr), dim=1) # Pass weights + attrs
        
        if len(self.layers) == 1:  # Handle single-layer case
            x = self.layers[0](x=x, edge_index=edge_index, edge_attr=edge_attr)
        else:
            for layer in self.layers[:-1]:
                x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
                x = F.leaky_relu(x)

            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_attr)  # Last layer
        
        return x.view(-1) ## For binary
    

## GraphSAGE
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, dropout, aggr="max", n_layers=2, normalize=False, project=False, bias=True):
        super().__init__()

        self.layers = torch.nn.ModuleList()

        if n_layers == 1:
            self.layers.append(SAGEConv(in_channels=num_features, out_channels=num_classes, aggr=aggr, normalize=normalize, project=project, bias=bias))  # Single-layer model
        elif n_layers == 2:
            self.layers.append(SAGEConv(in_channels=num_features, out_channels=hidden_channels, aggr=aggr, normalize=normalize, project=project, bias=bias))  # First layer
            self.layers.append(SAGEConv(in_channels=hidden_channels, out_channels=num_classes, aggr=aggr, normalize=normalize, project=project, bias=bias))  # Last layer
        else:
            self.layers.append(SAGEConv(in_channels=num_features, out_channels=hidden_channels, aggr=aggr, normalize=normalize, project=project, bias=bias))  # First layer
            for _ in range(1, n_layers - 1):
                self.layers.append(SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels, aggr=aggr, normalize=normalize, project=project, bias=bias))  # Middle layers
            self.layers.append(SAGEConv(in_channels=hidden_channels, out_channels=num_classes, aggr=aggr, normalize=normalize, project=project, bias=bias))  # Last layer
        
        self.dropout = dropout
        self.aggr = aggr


    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        if self.aggr == "lstm":
            edge_index, _ = torch.sort(edge_index, dim=1)  # for LSTM

        if len(self.layers) == 1:  # Handle single-layer case
            x = self.layers[0](x, edge_index)
        else:
            for layer in self.layers[:-1]:
                x = layer(x, edge_index)
                x = x.relu()
                if self.dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.layers[-1](x, edge_index)  # Last layer
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x.view(-1)  # For binary classification
    

## Random Predictor
class RandomPredictor(torch.nn.Module):
    def __init__(self, random_seed):
        super(RandomPredictor, self).__init__()
        self.random_seed = random_seed

    def forward(self, graph):
        # Set seed for reproducibility
        np.random.seed(self.random_seed)

        # Generate random logits (before sigmoid) for binary classification
        # Shape: [num_nodes, 1] to mimic binary classification output
        num_nodes = graph.x.shape[0]
        random_logits = np.random.randn(num_nodes, 1)  # Random normal distribution

        return torch.tensor(random_logits, dtype=torch.float)

    def eval(self):
        # Included just to match model API
        return super().eval()

    
######### LOSS FUNCTIONS #########
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: raw logits (before sigmoid)
        # targets: 0 or 1
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probas = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probas, 1 - probas)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SoftF1Loss(torch.nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        #self.pos_weight = pos_weight # Increase to penalize false negatives more
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))


    def forward(self, y_pred, y_true):
        """
        Computes a differentiable F1 loss focusing on the positive class.
        
        Args:
        - y_pred: Raw model outputs (logits) before sigmoid activation.
        - y_true: Binary ground truth labels (0 or 1).

        Returns:
        - Loss value that minimizes (1 - F1-score for the positive class).
        """
        y_pred = torch.sigmoid(y_pred)  # Convert logits to probabilities

        tp = (y_true * y_pred).sum()  # True Positives
        fp = ((1 - y_true) * y_pred).sum()  # False Positives
        fn = (y_true * (1 - y_pred)).sum()  # False Negatives

        precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
        recall = tp / (tp + fn + 1e-8)

        f1_positive = (2 * precision * recall) / (precision + recall + 1e-8)

        loss = (1 - f1_positive) * self.pos_weight
    
        return loss
    

# class LayerEmbeddingGenerator(nn.Module):
#     def __init__(self, layer_descriptions, output_dim, freeze_transformer=True):
#         """
#         layer_descriptions: A list of strings, one for each layer.
#         output_dim: The dimension you want for your layer embeddings.
#         freeze_transformer: If True, do not update the pre-trained transformer weights.
#         """
#         super().__init__()
#         self.layer_descriptions = layer_descriptions  # List of length num_layers
#         self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#         self.transformer = RobertaModel.from_pretrained('roberta-base')
        
#         if freeze_transformer:
#             for param in self.transformer.parameters():
#                 param.requires_grad = False
        
#         # Projection from transformer hidden size to desired output dimension.
#         self.fc = nn.Linear(self.transformer.config.hidden_size, output_dim)

#     def forward(self):
#         """
#         Computes and returns a tensor of shape [num_layers, output_dim]
#         where each row is the embedding of a layer.
#         """
#         embeddings = []
#         for desc in self.layer_descriptions:
#             inputs = self.tokenizer(desc, return_tensors="pt", truncation=True, padding=True)
#             outputs = self.transformer(**inputs)
#             # Use the first token embedding. I could also do pooling.
#             pooled = outputs.last_hidden_state[:, 0, :]  # Shape: [1, hidden_size]
#             embeddings.append(pooled)
#         # Concatenate along dimension 0 to form [num_layers, hidden_size]
#         embeddings = torch.cat(embeddings, dim=0)
#         # Project to desired output dimension
#         projected = self.fc(embeddings)  # Shape: [num_layers, output_dim]
#         return projected
    


# def mean_pooling(model_output, attention_mask):
#     """
#     Performs mean pooling on model_output by considering the attention_mask.
#     model_output: Tensor of shape [batch_size, seq_len, hidden_size]
#     attention_mask: Tensor of shape [batch_size, seq_len]
#     """
#     token_embeddings = model_output  # [B, T, H]
#     # Expand the attention_mask to match the embedding dims, and cast to float
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     # Sum embeddings along the time dimension
#     sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
#     # Avoid division by zero: count non-masked tokens for each example
#     sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
#     return sum_embeddings / sum_mask




# class SentenceTransformerWithClassifier(nn.Module):
#     def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", projection_dim=8, num_classes=96, fine_tune_transformer=False):
#         super().__init__()
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
#         self.input_dim = self.model.config.hidden_size  # 384 for all-MiniLM-L6-v2
#         # Projection head to get an 8-dimensional embedding
#         self.projection = nn.Linear(self.input_dim, projection_dim)
#         # Classification head: takes the 8-d embedding and outputs 96 logits
#         self.classifier = nn.Linear(projection_dim, num_classes)
        
#         # Freeze transformer weights if desired.
#         if not fine_tune_transformer:
#             for param in self.model.parameters():
#                 param.requires_grad = False

#     def mean_pooling(self, model_output, attention_mask):
#         """
#         Mean pooling on the token embeddings.
#         """
#         token_embeddings = model_output.last_hidden_state  # shape: [batch_size, seq_len, hidden_dim]
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
#         sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
#         return sum_embeddings / sum_mask

#     def forward(self, sentences, return_embedding=False):
#         """
#         sentences: list of strings (your keywords or short descriptions)
#         return_embedding: if True, returns the 8-dimensional embedding; otherwise, returns logits.
#         """
#         encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
#         device = next(self.model.parameters()).device
#         encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
#         model_output = self.model(**encoded_input)
#         # Compute a pooled representation
#         pooled = self.mean_pooling(model_output, encoded_input["attention_mask"])
#         # Project to get the low-dimensional embedding
#         embedding = self.projection(pooled)
        
#         # Compute logits from the embedding
#         logits = self.classifier(embedding)
        
#         if return_embedding:
#             return embedding
#         else:
#             return logits
        


# class SentenceDataset(Dataset):
#     def __init__(self, sentences, labels):
#         super().__init__()
#         self.sentences = sentences
#         self.labels = labels

#     def __len__(self):
#         return len(self.sentences)

#     def __getitem__(self, idx):
#         return self.sentences[idx], self.labels[idx]