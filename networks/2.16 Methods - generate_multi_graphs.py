####### NODE FEATURES #######

# Basic
import os
import pickle

# Pandas, Numpy
import pandas as pd
import numpy as np

# SKLearn
from sklearn.preprocessing import StandardScaler

# Custom
from utils import *

############ SETTINGS ############
transaction = "total" # "import" or "export" or "total"

##################################

df = pd.concat([pd.read_stata("../data/atlas/hs12_country_country_product_year_4_2012_2016.dta"),
                pd.read_stata("../data/atlas/hs12_country_country_product_year_4_2017_2021.dta"),
                pd.read_stata("../data/atlas/hs12_country_country_product_year_4_2022.dta")])
products = pd.read_csv("../data/atlas/product_hs12.csv", dtype={"code": str})
countries = pd.read_csv("../data/atlas/location_country.csv")
trustworhiness = pd.read_csv("../data/graphs_data/trustworthiness_scores.csv", dtype={"cmd": str})
geo_embeddings = pd.read_csv("./data/geo-embeddings.vec", sep=" ", skiprows=1, header=None)
geo_embeddings.rename(columns={0: "iso3_code"}, inplace=True)
geo_embeddings = countries[["country_id", "iso3_code"]].merge(geo_embeddings, on="iso3_code", how="left")
geo_embeddings.fillna(0, inplace=True)
geo_embeddings.drop("iso3_code", axis=1, inplace=True)
geo_embeddings.columns = ["country_id"] + [f"geo_{x}" for x in geo_embeddings.columns[1:]]
trading_agreements = pd.read_csv("../data/graphs_data/trading_agreements_edges.csv")

# Include product code
df = df.merge(products[["product_id", "code"]], how="left", on="product_id")
df.rename(columns={"code": "product_code"}, inplace=True)

df["product_code"] = df.product_code.str[:2] ## Reduce product code to 2 digits

# Include total trade value
df["total_value"] = df.export_value + df.import_value

# Scaler for features
scaler = StandardScaler()


for year in range(2012,2023):
    
    print(f"Computing year: {year}")

    ### NODE FEATURES FOR THE YEAR ###
    # Start with country list
    nodes_full = countries[["country_id", "name_short_en", "iso3_code"]]
    # Add COI and ECI year
    countries_attr = df.loc[df.year == year, ["country_id", "coi", "eci"]].drop_duplicates(subset="country_id", keep="first")
    nodes_full = nodes_full.merge(countries_attr, on="country_id", how="left").fillna(0)
    # Add geo-embedding
    nodes_full = nodes_full.merge(geo_embeddings, on="country_id", how="left")
    # Remove unuseful cols    
    node_features = nodes_full.drop(columns=["name_short_en", "iso3_code"])
    # HHI year
    trade_volumes = df.loc[df.year == year, ["country_id", "partner_country_id", "export_value", "import_value", "total_value"]]\
        .groupby(["country_id", "partner_country_id"]).sum().reset_index()
    risk = hhi_risk(trade_volumes, trade_col=f"{transaction}_value").reset_index().rename(columns={"trade_share": "risk"})
    node_features = node_features.merge(risk[["country_id", "risk"]], on="country_id", how="inner")

    node_features.to_csv(f"../data/graphs_data/multi-graph/{transaction}/node_features-{year}-{transaction}.csv", index=False)

    ### EDGES ###
    all_edges = pd.DataFrame()
    for product_code in [f"{x:02d}" for x in range(1, 100)]:

        if product_code in ["77", "98", "99"]: # 77 & 98 don't exist. 99 is not present in BACI
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
        
    all_edges.to_csv(f"../data/graphs_data/multi-graph/{transaction}/edge_features-{year}-{transaction}.csv", index=False)

        