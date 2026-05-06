####### COMPILE ALL GRAPHS #######
import numpy as np
import pandas as pd
import sys;sys.path.append("../gnn")
from utils import get_product_list, resolve_paths, load_atlas_data
from gnnutils import load_graph, load_data, feature_sanity_check, print_graph_info, save_object

READ_DATA_PATHS, WRITE_DATA_PATHS = resolve_paths(read_datasets=["Atlas Trade Data",
                                                                "Atlas Products Data",
                                                                 "UN Comtrade Reporters",
                                                                 "Graphs Data"], 
                                                write_datasets=["Graphs Data"])




### SETTINGS ###
graphs_type = "export" # "total", "export"
multi_graph = True
digits = 4 # 2 or 4
################

# Read data
labels = pd.read_csv(READ_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/labels-affected_importers.csv", dtype={"product_code": str})
labels.drop_duplicates(inplace=True) # Some countries might be "affected" by more than one exporter lost
products = get_product_list(digits=digits)

print(products[:10])
label_map = {"affected_importer": 1, "not_affected": 0}

print("Loading data... (takes around 1h30)")

train_graphs = []
test_graphs = []

graphs_root_path = READ_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/{'multi-graph' if multi_graph else ''}/{graphs_type}"

if multi_graph:
  # We don't care for individual products
  labels = labels.groupby(["year", "country_id"])["label"].min().reset_index()

  for year in range(2012, 2021):
    print(f"Year: {year}")
    g = load_graph(graph_name=f"{year}-{graphs_type}", labels=labels.loc[(labels.year==year), ["country_id", "label"]], label_map=label_map, \
          root_path=graphs_root_path)
    # Remove the commodity class ID from attributes, and pass it to special id
    g.edge_id = g.edge_attr[:,0].long() # Extract the id
    g.edge_attr = g.edge_attr[:,1:] # Remove the id
    train_graphs.append(g)
    
  # Last year
  g = load_graph(graph_name=f"2021-{graphs_type}", labels=labels.loc[(labels.year==2021), ["country_id", "label"]], label_map=label_map, \
          root_path=graphs_root_path)
  g.edge_id = g.edge_attr[:,0].long() # Extract the id
  g.edge_attr = g.edge_attr[:,1:] # Remove the id
  test_graphs.append(g)
  

else:

  for i, prod_code in enumerate(products):
      
      train_graphs += load_data(years=list(range(2012,2021)), code=prod_code, labels=labels, label_map=label_map, suffix=graphs_type, \
                                  root_path=graphs_root_path)
      test_graphs.append(
        load_graph(
          graph_name=f"2021-{prod_code}-{graphs_type}", 
          labels=labels.loc[(labels.year==2021) & (labels.product_code==prod_code), ["country_id", "label"]], 
          label_map=label_map, 
          root_path=graphs_root_path
          )
        )
      # For tests on 2022 -> We don't have true labels for it, but we can still evaluate the model's performance on the graph structure
      # test_graphs.append(
      #    load_graph(
      #       graph_name=f"2022-{prod_code}-{graphs_type}", 
      #       labels=labels.loc[(labels.year==2021) & (labels.product_code==prod_code), ["country_id", "label"]], 
      #       label_map=label_map, 
      #       root_path=graphs_root_path
      #       )
      # )
      if i % 10 == 0:
       save_object(train_graphs, graphs_root_path + "/train-graphs.pkl")
       save_object(test_graphs, graphs_root_path + "/test-graphs.pkl")
      
      print(f"{i/len(products):.2%} -> Product code: {prod_code}")

for graph in train_graphs:
  feature_sanity_check(graph)
  print_graph_info(graph)

save_object(train_graphs, graphs_root_path + "/train-graphs.pkl")
save_object(test_graphs, graphs_root_path + "/test-graphs.pkl")
#save_object(test_graphs,  graphs_root_path + "/test-2022-graphs.pkl")
