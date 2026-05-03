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
import time
import sys;sys.path.append("../networks")
from utils import resolve_paths, get_product_list

READ_DATA_PATHS, WRITE_DATA_PATHS = resolve_paths(read_datasets=["Atlas Products Data",
                                                                 "Results Data",
                                                                 "Graphs Data"], 
                                                write_datasets=["Results Data"])

# Optional, delayed start
delay_start = 60*60* 0

if delay_start > 0:
    print(f"Delaying start by {delay_start} seconds...")
    time.sleep(delay_start)

print("Starting...")
start_time = time.time()

### SETTINGS ###
model_type = "GAT"
graphs_type = "total" # "total", "export"
layered = False
multi_graph = False
graph_identifier = f"{'multi-graph-' if multi_graph else ''}{graphs_type}{'-layered' if (layered and not multi_graph) else ''}"
digits = 2 # 2 or 4
use_seed = 10
##############

# Read all products
products = get_product_list(digits=digits)

# GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"
print("Device: ", device)

# Best parameters from Optuna
best_params = json.load(open("./models/best_params.json", "r"))
best_params = best_params["No Ablation"][model_type][graph_identifier]
print(f"{model_type} | {graph_identifier} :: {best_params}")

graphs_path = READ_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/{'multi-graph/' if multi_graph else ''}{graphs_type}"
training_path = f"models/training/probabilities/{digits}_digits/{model_type}/{graph_identifier}"
os.makedirs(training_path, exist_ok=True)

#Load graphs (2012-2021)
graphs_2012_2020, graphs_2021 = get_preloaded_graphs(path=graphs_path)
# Load 2022 test graphs

with open(f'{graphs_path}/test-2022-graphs.pkl', 'rb') as inp:
    graphs2022 = pickle.load(inp)

all_graphs = graphs_2012_2020 + graphs_2021 + graphs2022

# Indices, for debugging
graphs_indeces = list([f"{c}-{y}" for c in products
                       for y in range(2012, 2023)])

if (layered or multi_graph):
    # Read layer embeddings
    layer_embeddings = pickle.load(open(READ_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/product_space_embeddings.pickle", "rb"))
    all_graphs = append_layer_embedding(graphs=all_graphs, layer_embeddings=layer_embeddings, multi_graph=multi_graph, years=list(range(2012, 2023)))


results = {
    "year": [],
    "model_type": [],
    "graph_type": [],
    "commodity": [],
    "y": [],
    "preds": [],
    "probs": []
}


# Reshuffle graphs to mask year
for pos, year in enumerate(range(2012, 2023)):

    print(f"Year: {year}")

    if year != 2022: # Just to test on 2022
        continue

    # If testing on 2022, load 2022 test graphs and use all other graphs for training
    if year < 2022:
        # All graphs with labels for training - We can't use 2022 graphs for training, since it doesn't have labels
        all_graphs = all_graphs[:-96]
    
    # If multi-graph, split by year
    if multi_graph:
        test_graphs = [all_graphs[pos]]
        train_graphs = all_graphs[:pos] + all_graphs[pos+1:]

    # Else, it's comoodity-specific graphs
    else:

        # We need to retrieve the correct graphs
        # Extract graphs for this year across all products
        test_graphs = [all_graphs[pos + product * 11] for product in range(96)]
        test_indeces = [graphs_indeces[pos + product * 11] for product in range(96)]
        
        # Extract graphs for all *other* years for each product
        train_graphs = [all_graphs[y + product * 11] for product in range(96) for y in range(11) if y != pos]
        train_indeces = [graphs_indeces[y + product * 11] for product in range(96) for y in range(11) if y != pos]
    
        # Debugging info
        print()  
        print("Train graphs: ")
        print(train_indeces)
        print()
        print("Test graphs: ")
        print(test_indeces)
        print()
    
    # Graph constants
    num_classes = 1
    num_features = test_graphs[0].num_features    
    pos_weight = get_pos_weight(train_graphs=train_graphs)
    edge_dim = test_graphs[0].edge_attr.shape[1]
    
    ## Training and testing
    if model_type == "GAT":
        model = GAT(num_features=num_features, num_classes=num_classes, hidden_channels=best_params['hidden_channels'],\
                        heads=best_params['heads'], dropout=best_params['dropout'], n_layers=best_params['n_layers'], residual=best_params['residual'], \
                                bias=best_params['bias'], edge_dim=edge_dim)
    elif model_type == "MLP":
        model = MLP(num_features=num_features, hidden_channels=best_params["hidden_channels"], num_classes=num_classes, dropout=best_params["dropout"])
    elif model_type == "GCN":
        model = GCN(num_features=num_features, hidden_channels=best_params["hidden_channels"], num_classes=num_classes, \
            n_layers=best_params["n_layers"], dropout=best_params["dropout"])
    elif model_type == "SAGE":
        model = GraphSAGE(num_features=num_features, hidden_channels=best_params["hidden_channels"], num_classes=num_classes, \
            n_layers=best_params["n_layers"], dropout=best_params["dropout"], aggr=best_params["aggr"], normalize=best_params["normalize"],\
                project=best_params["project"], bias=best_params["bias"])
    else:
        raise ValueError("Unknown model type")


    model = model.to(device)  # move model to GPU

    # Optimizer
    if "optimizer" in best_params:
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'])

    # Define weighted loss
    criterion = SoftF1Loss(pos_weight=pos_weight).to(device)
    scheduler = None # Not use it for now

    # Load best model from training for evaluation
    model.load_state_dict(torch.load(f"models/training/No Ablation/{model_type}/{graph_identifier}/{use_seed}/best_model.pt", 
                                     weights_only=True))

    # Moving to CPU for evaluation
    model = model.to("cpu")
    criterion = criterion.to("cpu")

    _, valid_graphs = split_graphs_for_validation(train_graphs, random_seed=7)
    tuned_threshold = tune_threshold(model, valid_graphs, device="cpu", threshold_candidates=np.linspace(0.5, 0.99, 50))

    # Get probs
    if len(test_graphs) > 1:

        for i, graph in enumerate(test_graphs):

            commodity, year = test_indeces[i].split("-")
            preds, y, probs = test(model, graph.to("cpu"), threshold=tuned_threshold, return_probs=True)
            results["year"] += [year] * len(probs)
            results["model_type"] += [model_type] * len(probs)
            results["graph_type"] += [graphs_type] * len(probs)
            results["commodity"] += [commodity] * len(probs)
            results["y"] += y.tolist()
            results["preds"] += preds.tolist()
            results["probs"] += probs.tolist()
    else:
        preds, y, probs = test(model, test_graphs[0].to("cpu"), threshold=tuned_threshold, return_probs=True)
        results["year"] += [year] * len(probs)
        results["model_type"] += [model_type] * len(probs)
        results["graph_type"] += [graphs_type] * len(probs)
        results["commodity"] += [-1] * len(probs)
        results["y"] += y.tolist()
        results["preds"] += preds.tolist()
        results["probs"] += probs.tolist()
        

resultes = pd.DataFrame(results)
resultes.to_csv(WRITE_DATA_PATHS["Results Data"] + f"/Probabilities/{digits}_digits/{model_type}-{graph_identifier}-results.csv", index=False)
end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")
