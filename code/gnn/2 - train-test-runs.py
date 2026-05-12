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
import sys
import argparse

# Custom
import sys;sys.path.append("../networks")
from gnnutils import *
from models import *
from utils import resolve_paths, get_product_list

import warnings
warnings.filterwarnings("ignore")

READ_DATA_PATHS, WRITE_DATA_PATHS = resolve_paths(read_datasets=["Atlas Products Data", "Graphs Data", "Results Data"], 
                                                write_datasets=["Results Data", "Training Data"])

############################################
############### Settings ###################

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models.")
    parser.add_argument("--model-type", choices=["MLP", "GCN", "GAT", "SAGE", "RandomForest", "XGBoost"], default="MLP")
    parser.add_argument("--digits", type=int, choices=[2, 4], default=4)
    parser.add_argument("--graphs-type", choices=["total", "export"], default="export")
    parser.add_argument("--layered", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multi-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ablate", default=None)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=100, help="Use -1 for full-batch training.")
    parser.add_argument("--train-from-scratch", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.batch_size != -1 and args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer, or -1 for full-batch training.")

    if args.ablate == "None":
        args.ablate = None

    allowed_ablations = {
        False: {"COI", "ECI", "# Prod", "SRCA", "Geo-Positional", "HHI", "TI", "Export Value", "Avg.PCI", "Trade Agreements"},
        True: {"COI", "ECI", "Geo-Positional", "HHI", "Export Value", "Avg.PCI", "# Prod", "Trade Agreements", "Trustworthiness"},
    }
    if args.ablate is not None and args.ablate not in allowed_ablations[args.multi_graph]:
        parser.error(
            f"--ablate must be one of {sorted(allowed_ablations[args.multi_graph])}, or None "
            f"when --{'multi-graph' if args.multi_graph else 'no-multi-graph'} is used."
        )

    return args


args = parse_args()
model_type = args.model_type
digits = args.digits
graphs_type = args.graphs_type
layered = args.layered
multi_graph = args.multi_graph
ablate = args.ablate
use_gpu = args.use_gpu
batch_size = args.batch_size
train_from_scratch = args.train_from_scratch
# Ablations
## Multi-Layers (10): "COI", "ECI", "# Prod", "SRCA", "Geo-Positional", "HHI", "TI", "Export Value", "Avg.PCI", "Trade Agreements"
## Multi-Graph (9): "COI", "ECI", "Geo-Positional", "HHI", "Export Value", "Avg.PCI", "# Prod", "Trade Agreements", "Trustworthiness"
############################################
############################################

graph_identifier = f"{'multi-graph-' if multi_graph else ''}{graphs_type}{'-layered' if (layered and not multi_graph) else ''}"
ablation_identifier = f"{'Ablation - ' + ablate if ablate else 'No Ablation'}"

if use_gpu:
    # GPU if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'
print("Device: ", device)

# Best parameters from Optuna
best_params = json.load(open("./models/best_params.json", "r"))
best_params = best_params[ablation_identifier][model_type][graph_identifier]
print(f"{ablation_identifier} | {model_type} | {graph_identifier} :: {best_params}")

# Load graphs
print("Looking for pre-loaded graphs...")
train_graphs, test_graphs = get_preloaded_graphs(path=READ_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/{'multi-graph/' if multi_graph else ''}{graphs_type}")
print("Found pre-loaded graphs!")

if (layered or multi_graph):
    print("Adding layer embeddings...")

    layer_ids = get_product_list(digits=digits)
    # Read layer embeddings
    layer_embeddings = pickle.load(open(READ_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/product_space_embeddings.pickle", "rb"))
    all_graphs = append_layer_embedding(graphs=train_graphs+test_graphs, layer_embeddings=layer_embeddings, layer_ids=layer_ids, multi_graph=multi_graph)
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
    training_path = WRITE_DATA_PATHS["Training Data"] + f"/{digits}_digits/{ablation_identifier}/{model_type}/{graph_identifier}/{seed}"
    evaluation_path = WRITE_DATA_PATHS["Results Data"] + f"/{digits}_digits/{ablation_identifier}/{model_type}/{graph_identifier}/{seed}"

    os.makedirs(training_path, exist_ok=True)
    os.makedirs(evaluation_path, exist_ok=True)

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

        with open(training_path + "/best_model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open(training_path + "/threshold.json", "w") as f:
            json.dump({"threshold": tuned_threshold}, f, indent=4)

        nr_epochs.append(1)
        epoch_times = [0.0]
        evaluate(model, test_graphs, threshold=tuned_threshold, save_path=evaluation_path, show=False, device="cpu")
        continue

    model = model.to(device)  # move model to GPU

    # Optimizer
    if "weight_decay" not in best_params:
        best_params["weight_decay"] = 0.01 # Default weight decay if not specified

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
    else:
        raise ValueError(f"Optimizer {best_params['optimizer']} not recognized.")
    
    # Define weighted loss
    criterion = SoftF1Loss(pos_weight=pos_weight).to(device)
    #criterion = torch.nn.BCEWithLogitsLoss().to(device)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = None # Not use it for now
    
    if train_from_scratch:
        print("Starting training.")

        l, v, epoch_times = train(model=model, train_graphs=train_graphs, optimizer=optimizer, criterion=criterion, \
                        scheduler=scheduler, epochs=500, batch_size=batch_size, patience=50, \
                            save_path=training_path, random_seed=seed, device=device, retain_graph=False)

        nr_epochs.append(len(epoch_times))
        plot_train_curves(l, v, save_path=training_path, show=False)
    
    
    # Load best model from training for evaluation
    model.load_state_dict(torch.load(f"{training_path}/best_model.pt", weights_only=True))

    # Moving to CPU for evaluation
    model = model.to("cpu")
    criterion = criterion.to("cpu")
    
    _, valid_graphs = split_graphs_for_validation(train_graphs, random_seed=seed)
    tuned_threshold = tune_threshold(model, valid_graphs, device="cpu", threshold_candidates=np.linspace(0.5, 0.99, 50))
    with open(training_path + "/threshold.json", "w") as f:
        json.dump({"threshold": tuned_threshold}, f, indent=4)

    evaluate(model, test_graphs, threshold=tuned_threshold, save_path=evaluation_path, show=False, device="cpu")
