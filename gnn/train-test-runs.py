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

# Custom
from gnnutils import *
from models import *

import warnings
warnings.filterwarnings("ignore")
#import time; print("WE ARE WAITING..."); time.sleep(60*60*7.5)
#print("REMEMBER TO SET THE CORRECT LOOP!")
#import sys; sys.exit()

############################################
############### Settings ###################
model_type = "GAT"
graphs_type = "total" # "total", "export"
layered = False
multi_graph = True
ablate = "Geo-Positional"  # None, "COI", "ECI", "Geo-Positional", "HHI", "TI", "Export Value", "Avg.PCI", "# Prod", "SRCA", "Trade Agreements", "Trustworthiness"
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
train_graphs, test_graphs = get_preloaded_graphs(path=f"../../data/graphs_data/{'multi-graph/' if multi_graph else ''}{graphs_type}")
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

epoch_times = None
nr_epochs = []

# Training and Testing seed-wise
for seed in range(6, 8):
    
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
        
    else:
        raise ValueError(f"Model type {model_type} not recognized.")
    
    
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

    print(model)
    
    # PATHS
    training_path = f"models/training/{ablation_identifier}/{model_type}/{graph_identifier}/{seed}"
    evaluation_path = f"results/{ablation_identifier}/{model_type}/{graph_identifier}/{seed}"
    
    if not os.path.exists(training_path):
        os.makedirs(training_path)
    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)
    
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

    evaluate(model, test_graphs, save_path=evaluation_path, show=False, device="cpu")