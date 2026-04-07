### TRAINING AND TESTING

# Numpy, Pandas, etc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score

# Torch and PyTorch Geometric
import torch
from torch.nn import functional as F
import torch_geometric
from torch_geometric import loader
from torch_geometric.transforms import NormalizeFeatures

# Others
from functools import total_ordering
import os
import pickle
import json
from pathlib import Path

# Custom
from gnnutils import *
from models import *

import warnings
warnings.filterwarnings("ignore")

############################################
############### Settings ###################
model_type = "GCN"
graphs_type = "export" # "total", "export"
layered = True
multi_graph = False
ablate = None  # "Geo-Positional"  # None, "COI", "ECI", "Geo-Positional", "HHI", "TI", "Export Value", "Avg.PCI", "# Prod", "SRCA", "Trade Agreements", "Trustworthiness"
############################################
############################################

graph_identifier = f"{'multi-graph-' if multi_graph else ''}{graphs_type}{'-layered' if (layered and not multi_graph) else ''}"
ablation_identifier = f"{'Ablation - ' + ablate if ablate else 'No Ablation'}"

# GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Best parameters from Optuna
best_params = json.load(open("./models/best_params.json", "r"))
best_params = best_params[ablation_identifier][model_type][graph_identifier]
print(f"{ablation_identifier} | {model_type} | {graph_identifier} :: {best_params}")

# Load graphs
print("Looking for pre-loaded graphs...")
train_graphs, test_graphs = get_preloaded_graphs(path=f"../../data/5. Graphs Data/{'multi-graph/' if multi_graph else ''}{graphs_type}")
print("Found pre-loaded graphs!")

if (layered or multi_graph):
    print("Adding layer embeddings...")
    # Read layer embeddings
    layer_embeddings = pickle.load(open("product_space_embeddings.pickle", "rb"))
    all_graphs = append_layer_embedding(graphs=train_graphs+test_graphs, layer_embeddings=layer_embeddings, multi_graph=multi_graph)
    train_graphs, test_graphs = train_graphs[:len(train_graphs)], all_graphs[len(train_graphs):]
    print(f"New layer shape: {train_graphs[0].x.shape}")

if ablate:

    print(f"Shapes BEFORE ablation of {ablate}")
    print(f"Number of training graphs: {len(train_graphs)}")
    print(f"Number of testing graphs: {len(test_graphs)}")
    print(f"Shape of training graphs {train_graphs[0].x.shape}, {train_graphs[0].edge_attr.shape}")
    print(f"Shape of testing graphs {test_graphs[0].x.shape}, {test_graphs[0].edge_attr.shape}")

    train_graphs, test_graphs = ablate_attribute(train_graphs, test_graphs, attribute=ablate, multi_graph=multi_graph)

    print(f"Shapes AFTER ablation of {ablate}")
    print(f"Number of training graphs: {len(train_graphs)}")
    print(f"Number of testing graphs: {len(test_graphs)}")
    print(f"Shape of training graphs {train_graphs[0].x.shape}, {train_graphs[0].edge_attr.shape}")
    print(f"Shape of testing graphs {test_graphs[0].x.shape}, {test_graphs[0].edge_attr.shape}")

# Graph constants
num_classes = 1
num_features = test_graphs[0].num_features    
pos_weight = get_pos_weight(train_graphs=train_graphs)
edge_dim = test_graphs[0].edge_attr.shape[1]
classical_models = {"RandomForest", "XGBoost"}


def stack_graph_data(graphs):
    x = torch.cat([graph.x for graph in graphs], dim=0)
    y = torch.cat([graph.y for graph in graphs], dim=0)
    return x, y


def tune_threshold(model, validation_graphs, threshold_candidates=None, device="cpu"):
    if threshold_candidates is None:
        threshold_candidates = np.linspace(0.05, 0.95, 91)

    model = model.to(device)
    model.eval()

    valid_probs = []
    valid_labels = []

    with torch.no_grad():
        for graph in validation_graphs:
            graph = graph.to(device)
            out = model(graph)
            valid_probs.append(torch.sigmoid(out).detach().cpu())
            valid_labels.append(graph.y.detach().cpu())

    valid_probs = torch.cat(valid_probs).numpy()
    y_valid_np = torch.cat(valid_labels).numpy()

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in threshold_candidates:
        threshold_preds = (valid_probs > threshold).astype(int)
        threshold_f1 = f1_score(y_valid_np, threshold_preds, pos_label=1, average="binary", zero_division=0)

        if threshold_f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(threshold_f1)

    print(f"Tuned threshold on validation split: {best_threshold:.2f} (hard F1: {best_f1:.4f})")
    return best_threshold


def fit_and_tune_threshold(model_cls, model_kwargs, train_graphs, random_seed, train_ratio=0.8):
    #threshold_candidates = np.linspace(0.05, 0.95, 91)
    threshold_candidates = np.linspace(0.05, 0.95, 50)
    threshold_search_train_graphs, threshold_search_valid_graphs = split_graphs_for_validation(
        train_graphs, train_ratio=train_ratio, random_seed=random_seed
    )

    x_train_split, y_train_split = stack_graph_data(threshold_search_train_graphs)

    threshold_model = model_cls(random_seed=random_seed, **model_kwargs)
    threshold_model.fit(x_train_split, y_train_split)
    return tune_threshold(threshold_model, threshold_search_valid_graphs, threshold_candidates=threshold_candidates)

epoch_times = None
nr_epochs = []

# Training and Testing seed-wise
for seed in range(1, 11):
    
    if epoch_times is not None:
        expected_remaing_time_secs = (11-seed) * np.mean(nr_epochs) * np.mean(epoch_times)
        expected_remaing_time = f"{expected_remaing_time_secs//3600:.0f}:{expected_remaing_time_secs%3600//60:.0f}:{expected_remaing_time_secs%3600%60:.0f}"
    else:
        expected_remaing_time = "-"
    
    print(f"Seed: {seed} | Expected remaining time: {expected_remaing_time}")
    set_seed(seed)

    if model_type == "MLP":

        model = MLP(num_features=num_features, hidden_channels=best_params["hidden_channels"], num_classes=num_classes, dropout=best_params["dropout"])

    elif model_type == "GCN":
        
        model = GCN(num_features=num_features, hidden_channels=best_params["hidden_channels"], num_classes=num_classes, \
                n_layers=best_params["n_layers"], dropout=best_params["dropout"])

    elif model_type == "GAT":

        model = GAT(num_features=num_features, num_classes=num_classes, hidden_channels=best_params['hidden_channels'],\
                        heads=best_params['heads'], dropout=best_params['dropout'], n_layers=best_params['n_layers'], residual=best_params['residual'], \
                                bias=best_params['bias'], edge_dim=edge_dim)
        
    elif model_type == "SAGE":

        model = GraphSAGE(num_features=num_features, hidden_channels=best_params["hidden_channels"], num_classes=num_classes, \
                n_layers=best_params["n_layers"], dropout=best_params["dropout"], aggr=best_params["aggr"], normalize=best_params["normalize"],\
                      project=best_params["project"], bias=best_params["bias"])

    elif model_type == "RandomForest":

        model = RandomForest(random_seed=seed, **best_params)

    elif model_type == "XGBoost":

        model = XGBoost(random_seed=seed, **best_params)
        
    else:
        raise ValueError(f"Model type {model_type} not recognized.")

    print(model)
    
    # PATHS
    training_path = Path("models") / "training" / ablation_identifier / model_type / graph_identifier / str(seed)
    evaluation_path = Path("results") / ablation_identifier / model_type / graph_identifier / str(seed)

    training_path.mkdir(parents=True, exist_ok=True)
    evaluation_path.mkdir(parents=True, exist_ok=True)

    if model_type in classical_models:
        print("Starting training.")
        model_class = type(model)
        tuned_threshold = fit_and_tune_threshold(
            model_cls=model_class,
            model_kwargs=best_params,
            train_graphs=train_graphs,
            random_seed=seed,
        )
        x_train, y_train = stack_graph_data(train_graphs)
        model.fit(x_train, y_train)
        with open(Path.joinpath(training_path, "best_model.pkl").resolve(), "wb") as f:
            pickle.dump(model, f)
        with open(Path.joinpath(training_path, "threshold.json").resolve(), "w") as f:
            json.dump({"threshold": tuned_threshold}, f, indent=4)
        nr_epochs.append(1)
        epoch_times = [0.0]
        evaluate(model, test_graphs, threshold=tuned_threshold, save_path=evaluation_path, show=False, device="cpu")
        continue

    model = model.to(device)  # move model to GPU

    # Optimizer
    if best_params['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    elif best_params['optimizer'] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    elif best_params['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'],\
                                        momentum=best_params['momentum'])
    elif best_params['optimizer'] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'],\
                                                momentum=best_params['momentum'])
    elif best_params['optimizer'] == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    
    # Define weighted loss
    criterion = SoftF1Loss(pos_weight=pos_weight).to(device)
    #criterion = torch.nn.BCEWithLogitsLoss().to(device)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = None # Not use it for now
    
    print("Starting training.")

    l, v, epoch_times = train(model=model, train_graphs=train_graphs, optimizer=optimizer, criterion=criterion, \
                       scheduler=scheduler, epochs=500, batch_size=-1, patience=50, \
                           save_path=training_path, random_seed=seed, device=device, retain_graph=True)

    nr_epochs.append(len(epoch_times))
    plot_train_curves(l, v, save_path=training_path, show=False)
    
    # Load best model from training for evaluation
    model.load_state_dict(torch.load(f"{training_path}/best_model.pt", weights_only=True))

    # Moving to CPU for evaluation
    model = model.to("cpu")
    criterion = criterion.to("cpu")

    _, valid_graphs = split_graphs_for_validation(train_graphs, random_seed=seed)
    tuned_threshold = tune_threshold(model, valid_graphs, device="cpu", threshold_candidates=np.linspace(0.5, 0.99, 50))
    with open(Path.joinpath(training_path, "threshold.json").resolve(), "w") as f:
        json.dump({"threshold": tuned_threshold}, f, indent=4)

    evaluate(model, test_graphs, threshold=tuned_threshold, save_path=evaluation_path, show=False, device="cpu")
