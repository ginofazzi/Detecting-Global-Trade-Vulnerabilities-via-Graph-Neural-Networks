import pandas as pd
import networkx as nx
import backboning as bb

def reconnect_singletons(df, df_bb, nodes):
   additional_edges = []
   missing_nodes = nodes - (set(df_bb["src"]) | set(df_bb["trg"]))
   for missing_node in missing_nodes:
      try:
         missing_node_edges = df[(df["src"] == missing_node) | (df["trg"] == missing_node)]
         additional_edges.append(pd.DataFrame(missing_node_edges.loc[missing_node_edges["score"].idxmax()]).T)
      except ValueError:
         pass
   return pd.concat([df_bb, pd.concat(additional_edges).drop_duplicates()])

def reconnect_components(df, df_bb):
   G = nx.from_pandas_edgelist(df_bb, source = "src", target = "trg")
   ccs = list(nx.connected_components(G))
   ccs = {n: i for i in range(len(ccs)) for n in ccs[i]}
   df["src_comp"] = df["src"].map(ccs)
   df["trg_comp"] = df["trg"].map(ccs)
   while df[["src_comp", "trg_comp"]].value_counts().size > 1:
      new_edge = df.loc[df.loc[df["src_comp"] != df["trg_comp"], "score"].idxmax()]
      df_bb = pd.concat([df_bb, pd.DataFrame(new_edge[["src", "trg", "nij", "score"]]).T])
      new_comp = new_edge[["src_comp", "trg_comp"]].min().min()
      old_comp = new_edge[["src_comp", "trg_comp"]].max().max()
      df.loc[df["src_comp"] == old_comp, "src_comp"] = new_comp
      df.loc[df["trg_comp"] == old_comp, "trg_comp"] = new_comp
   return df_bb

# Tee input required is a pandas dataframe with columns src, trg, and nij
# src and trg are the node ids of the origin an destination nodes
# nij is the weight of the edge, should be a discrete count
# If the network is undirected, if you have the src,trg edge you also MUST have the trg,src edge with the same weight
# A possible way to ensure that: df = pd.concat([df, df.rename(columns = {"src": "trg", "trg": "src"})])
#df = pd.read_csv(...)

# Change undirected to False if the network is undirected
df_nc = bb.noise_corrected(co_occur, undirected = True)

# Useful to know how many nodes and edges you're dropping
# Change the values of start_threshold, end_threshold, and step to explore different options for the threshold
# The value you like should be plugged in the bb.thresholding function
# Change undirected to False if the network is undirected
start_threshold = 1
end_threshold = 32
step = 1
pd.DataFrame(data = bb.test_densities(df_nc, start_threshold, end_threshold, step, undirected = True), columns = ("threshold", "nodes", "nodes%", "edges", "edges%", "avgdeg", "avgdeg%"))

df_nc_bb = bb.thresholding(df_nc, ???)

df_nc_bb = reconnect_singletons(df_nc, df_nc_bb, set(df["src"]) | set(df["trg"]))
df_nc_bb = reconnect_components(df_nc, df_nc_bb)