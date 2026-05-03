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
import sys;sys.path.append("../networks")
from utils import resolve_paths
from gnnutils import *
from models import *

READ_DATA_PATHS, WRITE_DATA_PATHS = resolve_paths(read_datasets=["Graphs Data",
                                                                 "Results Data"], 
                                                write_datasets=["Graphs Data"])

##########################################################


### SETTINGS ###
model_type = "GAT"
layered = True
multi_graph = False
digits = 2 # 2 or 4
#################

prob_results = {"Graph": [], "Model": [], "Seed": [], "Year": [], "NodePosition": [], "Probability": []}

for model_type in ["GAT"]:
    for graphs_type in ["export"]:
        for layered in [True]:
            if graphs_type == "total" and layered:
                continue
            for multi_graph in [True, False]:
                if multi_graph and not layered:
                    continue

                graph_identifier = f"{'multi-graph-' if multi_graph else ''}{graphs_type}{'-layered' if (layered and not multi_graph) else ''}"

                # GPU if possible
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print("Device: ", device)

                # Best parameters from Optuna
                best_params = json.load(open("./models/best_params.json", "r"))
                best_params = best_params["No Ablation"][model_type][graph_identifier]
                print(f"{model_type} | {graph_identifier} :: {best_params}")

                # Load graphs
                print("Looking for pre-loaded graphs...")
                train_graphs, test_graphs = get_preloaded_graphs(path=READ_DATA_PATHS["Graphs Data"] + 
                                                                 f"/{'multi-graph/' if multi_graph else ''}{graphs_type}")
                print("Found pre-loaded graphs!")

                if layered:
                    print("Adding layer embeddings...")
                    # Read layer embeddings
                    layer_embeddings = pickle.load(open(READ_DATA_PATHS["Graphs Data"] + "/product_space_embeddings.pickle", "rb"))
                    train_graphs = append_layer_embedding(train_graphs, layer_embeddings, 
                                                                       multi_graph=multi_graph)
                    test_graphs = append_layer_embedding(test_graphs, layer_embeddings, 
                                                                       multi_graph=multi_graph)
                    print(f"New layer shape: {train_graphs[0].x.shape}")

                # Graph constants
                num_classes = 1
                num_features = test_graphs[0].num_features    
                pos_weight = get_pos_weight(train_graphs=train_graphs)
                edge_dim = test_graphs[0].edge_attr.shape[1]

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

                # Define weighted loss
                criterion = SoftF1Loss(pos_weight=pos_weight).to(device)
                #GAT_criterion = torch.nn.BCEWithLogitsLoss().to(device)
                #GAT_scheduler = torch.optim.lr_scheduler.StepLR(GAT_optimizer, step_size=10, gamma=0.5)
                scheduler = None # Not use it for now

                for seed in range(1, 11):

                    # Load best model from training for evaluation
                    print(model_type, graph_identifier, seed)
                    model.load_state_dict(torch.load(f"./models/training/No ablation/{model_type}/{graph_identifier}/{seed}/best_model.pt", 
                                                     weights_only=True, map_location=device))

                    # Moving to CPU for evaluation
                    model = model.to("cpu")
                    criterion = criterion.to("cpu")

                    #evaluate(GATmodel, test_graphs, save_path=f"{model_type}-{graph_identifier}-{seed}", show=False, device="cpu")
                    for year, graph in enumerate(test_graphs, start=2012):
                        preds, y, probs = test(model, graph, threshold=0.5, return_probs=True)
                        #print(classification_report(y, preds, digits=4))

                        prob_results["Graph"] += [graph_identifier] * len(probs)
                        prob_results["Model"] += [model_type] * len(probs)
                        prob_results["Seed"] += [seed] * len(probs)
                        prob_results["Year"] += [year] * len(probs)
                        prob_results["NodePosition"] += range(len(probs))
                        prob_results["Probability"] += probs.tolist()

df = pd.DataFrame(prob_results)
df.to_csv(WRITE_DATA_PATHS["Results Data"] + "/Probabilities.csv", index=False)