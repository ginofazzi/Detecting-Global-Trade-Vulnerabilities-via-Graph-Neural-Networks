####### NODE FEATURES #######

# Pandas, Numpy
import pandas as pd
import numpy as np

# SKLearn
from sklearn.preprocessing import StandardScaler

# Custom
from utils import *

READ_DATA_PATHS, WRITE_DATA_PATHS = resolve_paths(read_datasets=["Atlas Trade Data",
                                                                "Atlas Countries Data", 
                                                                "Atlas Products Data",
                                                                 "Graphs Data"], 
                                                write_datasets=["Graphs Data"])

############ SETTINGS ############
transaction = "total" # "export" or "total"
digits = 4 # 2 or 4
##################################

# Read data
df = load_atlas_data(READ_DATA_PATHS["Atlas Trade Data"])
products = pd.read_csv(READ_DATA_PATHS["Atlas Products Data"], dtype={"code": str})
countries = pd.read_csv(READ_DATA_PATHS["Atlas Countries Data"], encoding="latin1")
trustworhiness = pd.read_csv(READ_DATA_PATHS["Graphs Data"] + "/trustworthiness_scores.csv", dtype={"cmd": str})
geo_embeddings = pd.read_csv(READ_DATA_PATHS["Graphs Data"] + "/geo-embeddings.vec", sep=" ", skiprows=1, header=None)
geo_embeddings.rename(columns={0: "iso_code"}, inplace=True)
geo_embeddings = countries[["country_id", "iso_code"]].merge(geo_embeddings, on="iso_code", how="left")
geo_embeddings.fillna(0, inplace=True)
geo_embeddings.drop("iso_code", axis=1, inplace=True)
geo_embeddings.columns = ["country_id"] + [f"geo_{x}" for x in geo_embeddings.columns[1:]]
trading_agreements = pd.read_csv(READ_DATA_PATHS["Graphs Data"] + "/trading_agreements_edges.csv",
                                  dtype={"country_id": str, "partner_country_id": str})

# Include product code
df = df.merge(products[["product_id", "code"]], how="left", on="product_id")
df.rename(columns={"code": "product_code"}, inplace=True)

if digits == 2:
    df["product_code"] = df.product_code.str[:2] ## Reduce product code to 2 digits

# Include total trade value
df["total_value"] = df.export_value + df.import_value

# Scaler for features
scaler = StandardScaler()


for year in range(2012,2023):
    
    print(f"Computing year: {year}")

    ### NODE FEATURES FOR THE YEAR ###
    
    # Start with country list
    nodes_full = countries[["country_id", "country", "iso_code"]]

    # Add COI and ECI year
    countries_attr = df.loc[df.year == year, ["country_id", "coi", "eci"]].drop_duplicates(subset="country_id", keep="first")
    nodes_full = nodes_full.merge(countries_attr, on="country_id", how="left").fillna(0)
    # Add geo-embedding
    nodes_full = nodes_full.merge(geo_embeddings, on="country_id", how="left")
    # Remove unuseful cols    
    node_features = nodes_full.drop(columns=["country", "iso_code"])
    # HHI year
    trade_volumes = df.loc[df.year == year, ["country_id", "partner_country_id", "export_value", "import_value", "total_value"]]\
        .groupby(["country_id", "partner_country_id"]).sum().reset_index()
    risk = hhi_risk(trade_volumes, trade_col=f"{transaction}_value").reset_index().rename(columns={"trade_share": "risk"})
    node_features = node_features.merge(risk[["country_id", "risk"]], on="country_id", how="inner")

    os.makedirs(WRITE_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/multi-graph/{transaction}", exist_ok=True)
    node_features.to_csv(WRITE_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/multi-graph/{transaction}/node_features-{year}-{transaction}.csv", index=False)

    ### EDGES ###
    all_edges = pd.DataFrame()
    for product_code in sorted(list(df.product_code.unique())):

        if (product_code.startswith("77") or product_code.startswith("98") or
        product_code.startswith("99") or product_code.startswith("XX")):
            continue

        print(f"{year} - {product_code}")
        
        subset = df[(df.year==year) & (df.product_code == product_code)]

        # Add TRANSACTION VALUE, Avg PCI, # UNIQUE PRODS
        edges = subset[subset[f"{transaction}_value"] > 0].groupby(["country_id", "partner_country_id", "product_code"])\
            .agg({f"{transaction}_value": "sum", "pci": "mean", "product_id": "nunique"}).reset_index().rename(columns={"product_id": "num_prods", "country_id": "src", "partner_country_id": "tgt"})
        #edges.drop("product_code", axis=1, inplace=True)

        # Add trading aggrements
        edges = edges.merge(trading_agreements, on=["src", "tgt"], how="left")
        edges.loc[:, "trade_agreement"] = edges["trade_agreement"].fillna(0)

        # Add Trustworthiness
        trustworthiness_year_product = trustworhiness.loc[(trustworhiness.year == year) & (trustworhiness.cmd == product_code), ["country_id", "trustworthiness"]]
        mean_trust = trustworthiness_year_product["trustworthiness"].mean()
        edges = edges.merge(trustworthiness_year_product, left_on="src", right_on="country_id", how="left").rename(columns={"trustworthiness": "trust_src"})
        edges["trust_src"] = edges["trust_src"].fillna(mean_trust) # Fill with mean trust
        edges = edges.merge(trustworthiness_year_product, left_on="tgt", right_on="country_id", how="left").rename(columns={"trustworthiness": "trust_tgt"})
        edges["trust_tgt"] = edges["trust_tgt"].fillna(mean_trust) # Fill with mean trust
        edges["trustworthiness"] = (edges.trust_src + edges.trust_tgt) / 2
        edges.loc[edges.trustworthiness == 0, "trustworthiness"] = 0.001 # To avoid zero error in log  
        edges.drop(columns=["country_id_x", "country_id_y", "trust_src", "trust_tgt"], inplace=True)

        # Scale
        edges["num_prods"] = scaler.fit_transform(edges[["num_prods"]].astype(float))
        edges.loc[:, f"{transaction}_value"] = np.log(edges[[f"{transaction}_value"]])
        edges["trade_agreement"] = scaler.fit_transform(edges[["trade_agreement"]].astype(float))
        edges["trustworthiness"] = np.log(edges["trustworthiness"]) # Convert trustworthiness to log scale

        # Append this layer's edges
        all_edges = pd.concat([all_edges, edges])
    
    os.makedirs(WRITE_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/multi-graph/{transaction}", exist_ok=True)
    all_edges.to_csv(WRITE_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/multi-graph/{transaction}/edge_features-{year}-{transaction}.csv", index=False)

        