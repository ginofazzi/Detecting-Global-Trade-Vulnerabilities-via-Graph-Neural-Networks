# Install required packages.
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# Helper function for visualization.
#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch_geometric
from torch_geometric import loader
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

import torch
from torch.nn import functional as F

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn.manifold import TSNE

from functools import total_ordering
import os
import pickle
import optuna
import random
import json
import time



def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def add_labels(node_df, labels):
    # Takes lablels as a two column dataframe with columns 'country_id' and 'label'
    node_df = node_df.merge(labels, on=["country_id",], how="left")
    if len(node_df[node_df.isna().any(axis=1)]) > 0:
      print(node_df[node_df.isna().any(axis=1)])
      raise Exception("Missing labels.")

    return node_df


def generate_edge_index(edge_list, node_df):
  src = [node_df[node_df.country_id == x].index[0] for x in edge_list["src"]]
  dst = [node_df[node_df.country_id == x].index[0] for x in edge_list["tgt"]]
  edge_index = torch.tensor([src, dst], dtype=torch.int64)

  return edge_index


def load_graph(graph_name, labels, label_map, root_path):

  # Load node features
  node_df = pd.read_csv(f"{root_path}/node_features-{graph_name}.csv")  # Ensure it has 'node_id' and feature columns
  nr_nodes = node_df.shape[0]
  node_df = add_labels(node_df, labels)
  assert node_df.shape[0] == nr_nodes, "Number of nodes changed after adding labels"
  node_features = torch.tensor(node_df.iloc[:, 1:-1].values, dtype=torch.float)  # Exclude 'node_id' and 'label'
  # Load edge list
  edge_df = pd.read_csv(f"{root_path}/edge_features-{graph_name}.csv")
  edge_list = edge_df.iloc[:,:2] # SRC and TGT

  edge_index = generate_edge_index(edge_list, node_df) # Edge index 2xE
  edge_weight = edge_df.iloc[:,2] # Edge weight 1xE
  edge_weight = torch.tensor(edge_weight.values.T, dtype=torch.float)
  edge_attr = torch.tensor(edge_df.iloc[:,2:].values, dtype=torch.float) # Edge attributes (include weight as attribute)

  # Load labels (0 = unaffected, 1 = lost exporter, 2 = affected importer)
  node_labels = node_df.iloc[:,-1].map(label_map)
  node_labels = torch.tensor(node_labels.values, dtype=torch.long)

  # Generate graph data
  g = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr, y=node_labels)

  # Split data
  train_mask = balance_masking(g.y)
  g.train_mask = train_mask
  g.val_mask = ~train_mask

  g.num_classes = len(label_map) # Lost exporter, affected, unaffected, both

  return g


def load_data(years, code, suffix, labels, label_map, root_path="../data/graphs_data/total"):
  data_list = []
  # Define multiple graphs
  for y in years:
    print(f"{root_path}/{y}-{code}-{suffix}")
    g = load_graph(graph_name=f"{y}-{code}-{suffix}", labels=labels.loc[(labels.year==y) & (labels.product_code==code), ["country_id", "label"]], label_map=label_map, root_path=root_path)
    data_list.append(g)

  return data_list


def balance_masking(y, train_ratio=0.8):

  # Find indices of each class
  class_0_idx = (y == 0).nonzero(as_tuple=True)[0]
  class_1_idx = (y == 1).nonzero(as_tuple=True)[0]

  # Get minimum class count
  min_count = int(min(len(class_0_idx), len(class_1_idx)) * train_ratio)

  # Randomly select equal samples from each class
  selected_0 = class_0_idx[torch.randperm(len(class_0_idx))[:min_count]]
  selected_1 = class_1_idx[torch.randperm(len(class_1_idx))[:min_count]]

  # Create balanced train mask
  selected_indices = torch.cat([selected_0, selected_1])
  train_mask = torch.zeros(len(y), dtype=torch.bool)
  train_mask[selected_indices] = True

  return train_mask


def print_graph_info(graph):
  print()
  print(graph)
  print('===========================================================================================================')

  # Gather some statistics about the graph.
  print(f'Number of nodes: {graph.num_nodes}')
  print(f'Number of edges: {graph.num_edges}')
  print(f'Average node degree: {graph.num_edges / graph.num_nodes:.2f}')
  print(f'Number of training nodes: {graph.train_mask.sum()}')
  print(f'Training node label rate: {" | ".join([str(int((graph.y[graph.train_mask] == x).sum()) / graph.num_nodes) for x in range(2)])}')
  print(f'Has isolated nodes: {graph.has_isolated_nodes()}')
  print(f'Has self-loops: {graph.has_self_loops()}')
  print(f'Is undirected: {graph.is_undirected()}')

  print('===========================================================================================================')


def feature_sanity_check(graph):
  assert type(graph.x.mean().item()) == float, "Sanity check failed: x.mean() is not a float"
  assert type(graph.x.std().item()) == float, "Sanity check failed: x.std() is not a float"

  print("Sanity check passed!")
  return



def get_preloaded_graphs(path="."):
  with open(f'{path}/train-graphs.pkl', 'rb') as inp:
    train_graphs = pickle.load(inp)
  with open(f'{path}/test-graphs.pkl', 'rb') as inp:
    test_graphs = pickle.load(inp)

  return train_graphs, test_graphs


def train_layers(model, train_graphs, optimizer, criterion, epochs=100):

  model.train()

  epoch_list = []
  loss_list = []

  for epoch in range(epochs):

    total_loss = 0

    for graph in train_graphs:

      x, edge_index, edge_weight = graph.x, graph.edge_index, graph.edge_weight
      y = graph.y

      optimizer.zero_grad()  # Clear gradients.
      out = model(graph) # Perform a single forward pass.
      loss = criterion(out[graph.train_mask], y[graph.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      total_loss += loss.item()

    epoch_list.append(epoch+1)
    loss_list.append(total_loss / len(train_graphs))

  return epoch_list, loss_list



def train(model, train_graphs, optimizer, criterion, scheduler=None,
           epochs=100, batch_size=-1, patience=-1, train_ratio=0.8, random_seed=None, save_path="", device="cpu", retain_graph=False):
    
    torch.autograd.set_detect_anomaly(True)

    train_loss_list = []
    val_loss_list = []

    if not os.path.exists(save_path):
            os.mkdir(save_path)

    with open(f"./{save_path}/training-s{random_seed}.log", "w") as f:
       f.write("epoch,train_loss,val_loss\n")

    best_val_loss = float("inf")
    patience_counter = 0

    graphs_with_pos = [g for g in train_graphs if g.y.sum() > 0] # Graph containing positive labels
    graphs_without_pos = [g for g in train_graphs if g not in graphs_with_pos] # Graphs without positive labels

    if len(graphs_without_pos) > 0: # If there are no empty graphs
      train_graphs_pos, valid_graphs_pos = train_test_split(graphs_with_pos, train_size=train_ratio, random_state=random_seed, shuffle=True)
      train_graphs_w_pos, valid_graphs_w_pos = train_test_split(graphs_without_pos, train_size=train_ratio, random_state=random_seed, shuffle=True)
      train_graphs = train_graphs_pos + train_graphs_w_pos # Reuse the graphs without positive labels
      valid_graphs = valid_graphs_pos + valid_graphs_w_pos # Reuse the graphs without positive labels
    else:
      train_graphs, valid_graphs = train_test_split(graphs_with_pos, train_size=train_ratio, random_state=random_seed, shuffle=True)

    # Shuffle graphs
    random.Random(random_seed).shuffle(train_graphs)
    random.Random(random_seed).shuffle(valid_graphs)

    train_idx = list(range(len(train_graphs))) # Indeces for batching

    # Create mini-batches
    if batch_size != -1:
        batch_idxs = [train_idx[i:i+batch_size] for i in range(0, len(train_idx), batch_size)]
        nbatches = len(batch_idxs)
        print(f"Using {nbatches} batches of {batch_size} graphs each.")
    else:
        print("Training on all graphs (no batching)")
        batch_idxs = [train_idx]
        nbatches = 1

    epoch_times = []

    for epoch in range(epochs):
        
        epoch_start_time = time.time()

        print(f"Epoch {epoch+1}/{epochs} (Avg. time: {np.mean(epoch_times):.1f})", end="\r")
        total_train_loss = 0
        total_val_loss = 0

        # Pick batch (cycled if nbatches < epochs)
        batch_ids = batch_idxs[epoch % nbatches]
        training_batch = [train_graphs[i] for i in batch_ids]

        model.train()
        for graph in training_batch:
            #if graph.y.sum() == 0:
            #    continue  # Skip graphs without training nodes
            graph = graph.to(device)  # Move to GPU
            optimizer.zero_grad()
            out = model(graph)

            loss = criterion(out, graph.y.float())
            loss.backward(retain_graph=retain_graph)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        # Validation
        val_preds = []
        val_labels = []

        model.eval()
        with torch.no_grad():
            for graph in valid_graphs:
                graph = graph.to(device)  # Move to GPU
                out = model(graph)
                val_loss = criterion(out, graph.y.float())
                total_val_loss += val_loss.item()
                probs = torch.sigmoid(out).cpu()
                preds = (probs > 0.5).long()
                labels = graph.y.cpu().long()
                val_preds.append(preds)
                val_labels.append(labels)

        # Avoid division by zero if all graphs were skipped
        train_loss_count = len(training_batch)#len([g for g in training_batch if g.y.sum() > 0])
        val_loss_count = len(valid_graphs)

        # Concatenate all graphs' predictions & labels
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()

        # Compute metrics
        precision = precision_score(val_labels, val_preds, pos_label=1, average='binary', zero_division=0)
        recall = recall_score(val_labels, val_preds, zero_division=0, pos_label=1, average='binary')
        f1 = f1_score(val_labels, val_preds, zero_division=0, pos_label=1, average="binary")

        avg_train_loss = total_train_loss / max(train_loss_count, 1)
        avg_val_loss = total_val_loss / max(val_loss_count, 1)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} (patience: {patience_counter}) (Best: {best_val_loss:.4f}) | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)

        with open(f"./{save_path}/training-s{random_seed}.log", "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"./{save_path}/best_model.pt")
        else:
            patience_counter += 1

        if (patience != -1) and (patience_counter >= patience):
            print(f"\nEarly stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
            break

        if scheduler is not None:
            scheduler.step()

        epoch_end_time = time.time()

        epoch_times.append(epoch_end_time - epoch_start_time)

    return train_loss_list, val_loss_list, epoch_times



def plot_train_curves(train_loss_list, val_loss_list, save_path=None, show=True):
      # Plotting
      fig, ax = plt.subplots(figsize=(12, 5))
      ax.plot(range(1, len(train_loss_list)+1), train_loss_list, label="Train Loss", linewidth=2, color="navy")
      ax.plot(range(1, len(val_loss_list)+1), val_loss_list, label="Validation Loss", linewidth=2, color="crimson")
      ax.set_xlim(1)
      ax.set_xlabel("Epoch", fontweight="bold")
      ax.set_ylabel("Loss", fontweight="bold")
      plt.legend()
      plt.grid(True)
      
      if save_path != None:
         plt.savefig(f"{save_path}/loss.png", dpi=300, bbox_inches="tight")
      if show:
        plt.show()  




def test(model, graph, threshold=0.5, return_probs=False):
  model.eval()
  #x, edge_index, edge_weight = graph.x, graph.edge_index, graph.edge_weight
  y = graph.y

  with torch.no_grad():
    out = model(graph)
    #preds = out.argmax(dim=1)  # Use the class with highest probability.
    probs = torch.sigmoid(out)  # For binary; Convert to probabilities
    preds = (probs > threshold).long()  # For binary; Threshold at 0.5 for binary classification
  
  if return_probs:
     return preds, y, probs
  else:
    return preds, y



def evaluate(model, test_graphs, target_names=["Not Affected", "Affected"], threshold=0.5, save_path=None, show=True, device="cpu"):

  if type(test_graphs) == list:
    pred = []
    labels = []
    for graph in test_graphs:
      graph.to(device)  # Move to GPU
      graph_pred, graph_labels = test(model, graph, threshold=threshold)
      pred.extend(graph_pred)
      labels.extend(graph_labels)

  else:
    pred, labels = test(model, test_graphs.to(device))

  # De-tensor these
  pred = [int(x) for x in pred]
  labels = [int(x) for x in labels]

  acc = accuracy_score(labels, pred)
  f1_macro_avg = f1_score(labels, pred, average="macro")
  f1_pos = f1_score(labels, pred, pos_label=1, average="binary")

  print(f'Test Accuracy: {acc:.4f}')
  print(f'Test Avg. F1-Score: {f1_macro_avg:.4f}')
  print(f'Test Pos. F1-Score: {f1_pos:.4f}')
  print(classification_report(labels, pred, target_names=target_names))

  if save_path != None:
    d = classification_report(labels, pred, target_names=target_names, output_dict=True)
    d["Accuracy"] = acc
    d["F1 - Avg. Macro"] = f1_macro_avg
    d["F1 - Positives"] = f1_pos

    with open(f"./{save_path}/report.json", "w") as file:
      json.dump(d, file, indent=4)
    
    # Save also all predictions
    results = {"predictions": pred, "labels": labels}
    with open(f"./{save_path}/predictions.json", "w") as file:
      json.dump(results, file, indent=4)

  # Compute confusion matrix
  cm = confusion_matrix(labels, pred)

  # Plot confusion matrix
  # Plot with seaborn
  fig, ax = plt.subplots(figsize=(7, 7))
  
  ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
              xticklabels=target_names, yticklabels=target_names, linewidths=1, linecolor='white', robust=True,
                annot_kws={"fontsize": 12, "fontweight": "bold"}, square=True)
  
  ax.set_xticklabels(target_names, fontsize=10, fontweight='bold')
  ax.set_yticklabels(target_names, fontsize=10, fontweight='bold')
  # Styling
  plt.xlabel(" "*32 + "Predicted Labels" + " "*32, fontsize=12, fontweight='bold', labelpad=25, \
             bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='square,pad=0.3', lw=1, alpha=0.7))
  plt.ylabel(" "*34 + "True Labels" + " "*35, fontsize=12, fontweight='bold', labelpad=25, \
             bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='square,pad=0.3', lw=1, alpha=0.7))

  # Adjust layout to increase space between labels and plot
  fig.tight_layout(pad=40)

  if save_path != None:
     plt.savefig(f"./{save_path}/CM.png", dpi=300, bbox_inches="tight")

  # Show plot
  if show:
    plt.show()



def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def create_batches(train_graphs):

    graphs_with_pos = [g for g in train_graphs if g.y.sum() > 0] # Graph containing positive labels
    graphs_without_pos = [g for g in train_graphs if g not in graphs_with_pos] # Graphs without positive labels

    # Shuffle the lists to randomize batches
    random.shuffle(graphs_with_pos)
    random.shuffle(graphs_without_pos)

    # Determine batch distribution
    num_pos = len(graphs_with_pos)  # Number of positive-label graphs
    num_neg = len(graphs_without_pos)  # Number of negative-label graphs
    neg_per_batch = num_neg // num_pos  # Number of negatives per batch

    # Create batches
    batches = []
    for i in range(num_pos):
        batch = [graphs_with_pos[i]] + graphs_without_pos[i * neg_per_batch : (i + 1) * neg_per_batch]
        batches.append(batch)

    # If there are remaining negative graphs, distribute them evenly
    remaining_negatives = graphs_without_pos[num_pos * neg_per_batch:]
    random.shuffle(remaining_negatives)

    for i, graph in enumerate(remaining_negatives):
        batches[i % num_pos].append(graph)

    return batches


def get_pos_weight(train_graphs):
    pos = 0
    neg = 0
    for g in train_graphs:
        pos += len(g.y[g.y == 1])
        neg += len(g.y[g.y == 0])

    pos_weight = torch.tensor([neg / pos])
    print("Pos Weight: ", pos_weight)

    return pos_weight


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    


def append_layer_embedding(graphs, layer_embeddings, multi_graph=False, years=None):

  if years == None:
    years = list(range(2012, 2022))  # Default years from 2012 to 2021
  
  print(f"Previous edge features shape: {graphs[0].edge_attr.shape}")
  
  # If multi-graph, the layer embedding gets appended to edges
  if multi_graph:
    # Mapping for annying Layer ids
    valid_codes = [i for i in range(1, 100) if i not in (77, 98, 99)]
    code2idx = {code: idx for idx, code in enumerate(valid_codes)}
    lut = torch.full((98,), -1, dtype=torch.long)  # default -1 for “invalid”
    
    for code, idx in code2idx.items():
      lut[code] = idx

    for year, graph in zip(years, graphs):
      
      edge_index = lut[graph.edge_id]
      layer_emb = layer_embeddings[year][edge_index].to(dtype=torch.float32)  # Get layer embedding
      graph.edge_attr = torch.cat([graph.edge_attr, layer_emb], dim=1)
    
    print(f"New edge features shape: {graphs[0].edge_attr.shape}")
       

  # Otherwise, we append it to nodes, per commodity
  else:

    for i, graph in enumerate(graphs):
      layer_id = torch.tensor(i // (len(graphs)//96)) # 9 years per layer
      year = years[i % (len(graphs)//96)] # years from 2012 to 2020, and restart
      print(f"Graph {i} | Year: {year} | Layer ID: {layer_id.item()}")
      # Get Layer Embedding
      layer_emb = layer_embeddings[year][layer_id].to(dtype=torch.float32)  # Get layer embedding
      # Concat layer embedding with node features
      layer_emb = layer_emb.repeat(graph.x.size(0), 1)  # Repeat for each node
      graph.x = torch.cat((graph.x, layer_emb), dim=1)  # Concatenate layer embedding with node features
    

  return graphs



def ablate_attribute(train_graphs, test_graphs, attribute, multi_graph=False):
    """
    Remove an attribute from either edge_attr or x (node attribute).
    
    train_graphs, test_graphs: lists of torch_geometric.data.Data
    attribute: one of the keys in attribute_indices
    multi_graph: whether the graph is multi-graph (affects which indices to use)
    """
    if multi_graph:
        # intervals are [start, end) on edge_attr
        attribute_indices = {
           "nodes": {
              "COI": (0, 1),
              "ECI": (1, 2),
              "Geo-Positional": (2, 10),
              "HHI": (10, 11),
            },
            "edges": {
               "Export Value": (0, 1),
               "Avg.PCI": (1, 2),
               "# Prod": (2, 3),
               "Trade Agreements": (3, 4),
               "Trustworthiness": (4, 5),
               "Layer Embedding": (5, 13)
            }
        }
    else:
        # intervals are [start, end) on x
        attribute_indices = {
           "nodes": {
              "COI": (0, 1),
              "ECI": (1, 2),
              "# Prod": (2, 3),
              "SRCA": (3, 4),
              "Geo-Positional": (4, 12),
              "HHI": (12, 13),
              "TI": (13, 14),
           },
           "edges": {
               "Export Value": (0, 1),
               "Avg.PCI": (1, 2),
               "# Prod": (2, 3),
               "Trade Agreements": (3, 4)
            }
        }

    if (attribute not in attribute_indices["nodes"]) and (attribute not in attribute_indices["edges"]):
        raise ValueError(f"Unknown attribute {attribute}. Allowed attributes: {list(attribute_indices['nodes'].keys()) + list(attribute_indices['edges'].keys())}")

    if attribute in attribute_indices["nodes"]:
      start, end = attribute_indices["nodes"][attribute]
    else:
      start, end = attribute_indices["edges"][attribute]

    def _ablate(feature_tensor: torch.Tensor):
        # feature_tensor: [*, F]
        F = feature_tensor.size(1)
        if not (0 <= start < end <= F):
            raise ValueError(f"Ablation interval {(start,end)} out of bounds for width {F}")
        # keep [:start] and [end:]
        left  = feature_tensor[:, :start]   if start > 0 else None
        right = feature_tensor[:, end:]     if end  < F else None
        if left is None:
            return right
        if right is None:
            return left
        return torch.cat([left, right], dim=1)

    # apply to train and test
    for G in train_graphs + test_graphs:
        
        if attribute in attribute_indices["nodes"]:
           G.x = _ablate(G.x)
        else:
           G.edge_attr = _ablate(G.edge_attr)

    return train_graphs, test_graphs



def sample_random_subgraph(data: Data, k: int) -> Data:
    # if the graph is already small enough, just return it
    if data.num_nodes <= k:
        return data

    # pick k nodes at random
    perm = torch.randperm(data.num_nodes, device=data.x.device)
    subset = perm[:k]

    # extract the induced subgraph (and relabel its nodes to [0..k-1])
    edge_index, edge_attr = subgraph(subset, data.edge_index, data.edge_attr,
                                     relabel_nodes=True, num_nodes=data.num_nodes)

    # slice the node features (and any other per-node data)
    x = data.x[subset]
    y = data.y[subset]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


### DEPRECATED: DO NOT USE -> Finishes optimization after first trial.
def safe_optimize(study, objective, n_trials, gc_after_trial, n_jobs, timeout):
    
    backoff = 5

    while True:
        try:
            study.optimize(objective, n_trials, gc_after_trial, n_jobs, timeout)
            break # If it works, exit the loop
        except:
              print(f"DB locked, retrying in {backoff}s…")
              time.sleep(backoff)
              backoff = min(backoff * 2, 60)  # exponential backoff up to 1m
