####### COMPILE ALL GRAPHS #######
import numpy as np
import pandas as pd
import sys;sys.path.append("./gnn")
from utils import *
from gnnutils import *

# Settings
graphs_type = "export" # "total", "export"
multi_graph = False

labels = pd.read_csv("../data/labels-affected_importers.csv", dtype={"product_code": str})
labels.drop_duplicates(inplace=True) # Some countries might be "affected" by more than one exporter lost

label_map = {"affected_importer": 1, "not_affected": 0}

print("Loading data... (takes around 1h30)")

train_graphs = []
test_graphs = []

if multi_graph:
  # We don't care for individual products
  labels = labels.groupby(["year", "country_id"])["label"].min().reset_index()
  
  for year in range(2012, 2021):
    print(f"Year: {year}")
    g = load_graph(graph_name=f"{year}-{graphs_type}", labels=labels.loc[(labels.year==year), ["country_id", "label"]], label_map=label_map, \
          root_path=f"../data/5. Graphs Data{'/multi-graph/' if multi_graph else '/'}{graphs_type}")
    # Remove the commodity class ID from attributes, and pass it to special id
    g.edge_id = g.edge_attr[:,0].long() # Extract the id
    g.edge_attr = g.edge_attr[:,1:] # Remove the id
    train_graphs.append(g)
    
  # Last year
  g = load_graph(graph_name=f"2021-{graphs_type}", labels=labels.loc[(labels.year==2021), ["country_id", "label"]], label_map=label_map, \
          root_path=f"../data/5. Graphs Data/{'/multi-graph/' if multi_graph else '/'}{graphs_type}")
  g.edge_id = g.edge_attr[:,0].long() # Extract the id
  g.edge_attr = g.edge_attr[:,1:] # Remove the id
  test_graphs.append(g)
  

else:
  for i, prod_code in enumerate([f"{x:02d}" for x in range(1, 100)]):
      if prod_code in ["77", "98", "99"]: # 77 & 98 don't exist. 99 is not present in BACI
          continue
      train_graphs += load_data(years=list(range(2012,2021)), code=prod_code, labels=labels, label_map=label_map, suffix=graphs_type, \
                                  root_path=f"../data/5. Graphs Data/{graphs_type}")
      test_graphs.append(load_graph(graph_name=f"2021-{prod_code}-{graphs_type}", labels=labels.loc[(labels.year==2021) & (labels.product_code==prod_code), \
                                                                                                      ["country_id", "label"]], label_map=label_map, \
                                                                                                      root_path="../data/5. Graphs Data/{graphs_type}"))
      #test_graphs.append(load_graph(graph_name=f"2022-{prod_code}-{graphs_type}", labels=labels.loc[(labels.year==2021) & (labels.product_code==prod_code), \
      #                                                                                                ["country_id", "label"]], label_map=label_map, \
      #                                                                                                root_path="../data/5. Graphs Data/{graphs_type}"))
      print(f"{i/100:.2%}")

for graph in train_graphs:
  feature_sanity_check(graph)
  print_graph_info(graph)

save_object(train_graphs, f"../data/5. Graphs Data{'/multi-graph/' if multi_graph else '/'}{graphs_type}/train-graphs.pkl")
save_object(test_graphs, f"../data/5. Graphs Data{'/multi-graph/' if multi_graph else '/'}{graphs_type}/test-graphs.pkl")
#save_object(test_graphs, f"../data/5. Graphs Data{'/multi-graph/' if multi_graph else '/'}{graphs_type}/test-2022-graphs.pkl")