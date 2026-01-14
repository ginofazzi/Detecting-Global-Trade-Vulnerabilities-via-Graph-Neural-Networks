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

df = pd.concat([pd.read_stata("../../data/2. Atlas/hs12_country_country_product_year_4_2012_2016.dta"),
                pd.read_stata("../../data/2. Atlas/hs12_country_country_product_year_4_2017_2021.dta"),
                pd.read_stata("../../data/2. Atlas/hs12_country_country_product_year_4_2022.dta")])
products = pd.read_csv("../../data/2. Atlas/product_hs12.csv", dtype={"code": str})
countries = pd.read_csv("../../data/2. Atlas/location_country.csv")
trustworhiness = pd.read_csv("../../data/trustworthiness_scores.csv", dtype={"cmd": str})
geo_embeddings = pd.read_csv("../../data/geo-embeddings.vec", sep=" ", skiprows=1, header=None)
geo_embeddings.rename(columns={0: "iso3_code"}, inplace=True)
geo_embeddings = countries[["country_id", "iso3_code"]].merge(geo_embeddings, on="iso3_code", how="left")
geo_embeddings.fillna(0, inplace=True)
geo_embeddings.drop("iso3_code", axis=1, inplace=True)
geo_embeddings.columns = ["country_id"] + [f"geo_{x}" for x in geo_embeddings.columns[1:]]
trading_agreements = pd.read_csv("../../data/trading_agreements_edges.csv")

with open("../../data/SRCA.pickle", "rb") as f:
    rca_dict = pickle.load(f)

# Include product code
df = df.merge(products[["product_id", "code"]], how="left", on="product_id")
df.rename(columns={"code": "product_code"}, inplace=True)

df["product_code"] = df.product_code.str[:2] ## Reduce product code to 2 digits

# Include total trade value
df["total_value"] = df.export_value + df.import_value

# Scaler for features
scaler = StandardScaler()

for product_code in [f"{x:02d}" for x in range(1, 100)]:

    if product_code in ["77", "98"]: # Not valid codes
        continue

    for year in range(2012,2023):

        print(f"{product_code} - {year}")
        
        subset = df[(df.year==year) & (df.product_code.str.startswith(product_code))]

        prod_subset = subset[(subset[f"{transaction}_value"] > 0)]
        unique_prods = prod_subset[["country_id", "product_id"]].groupby("country_id").nunique().reset_index()
        
        # Start with country list
        nodes_full = countries[["country_id", "name_short_en", "iso3_code"]]

        # Add COI and ECI
        countries_attr = df.loc[df.year == year, ["country_id", "coi", "eci"]].drop_duplicates(subset="country_id", keep="first")
        nodes_full = nodes_full.merge(countries_attr, on="country_id", how="left").fillna(0)

        # Add # unique products in commodity category
        nodes_full = nodes_full.merge(unique_prods, on="country_id", how="left").fillna(0)
        nodes_full.rename(columns={"product_id": "prod_num"}, inplace=True)
        
        # SRCA
        rca_scaled = pd.DataFrame(scaler.fit_transform(rca_dict[year].T), index=rca_dict[year].T.index, columns=rca_dict[year].T.columns)
        rca_year_prod = rca_scaled.T.loc[:, product_code].reset_index().rename(columns={product_code: "rca"})
        nodes_full = nodes_full.merge(rca_year_prod, on="country_id", how="left")

        # Add geo-embedding
        nodes_full = nodes_full.merge(geo_embeddings, on="country_id", how="left")
        
        node_features = nodes_full.drop(columns=["name_short_en", "iso3_code"])

        node_features.loc[:, "prod_num"] = scaler.fit_transform(node_features[["prod_num"]])

        #trade_partners = subset[["country_id", "partner_country_id"]].groupby("country_id").nunique().reset_index().rename(columns={"partner_country_id": "num_partners"})

        trade_volumes = subset[["country_id", "partner_country_id", "export_value", "import_value", "total_value"]].groupby(["country_id", "partner_country_id"]).sum().reset_index()

        risk = hhi_risk(trade_volumes, trade_col=f"{transaction}_value").reset_index().rename(columns={"trade_share": "risk"})

        node_features = node_features.merge(risk[["country_id", "risk"]], on="country_id", how="inner")
        node_features = node_features.merge(trustworhiness.loc[(trustworhiness.year == year) & (trustworhiness.cmd == product_code), ["country_id", "trustworthiness"]], on="country_id", how="left")
        mean_trust = node_features["trustworthiness"].mean()
        node_features["trustworthiness"] = node_features["trustworthiness"].fillna(mean_trust) # Fill with mean trust
        node_features.loc[node_features.trustworthiness == 0, "trustworthiness"] = 0.001 # To avoid zero error in log
        # Convert trustworthiness to log scale
        node_features["trustworthiness"] = np.log(node_features["trustworthiness"])

        edges = subset[subset[f"{transaction}_value"] > 0].groupby(["country_id", "partner_country_id", "product_code"])\
            .agg({f"{transaction}_value": "sum", "pci": "mean", "product_id": "nunique"}).reset_index().rename(columns={"product_id": "num_prods", "country_id": "src", "partner_country_id": "tgt"})
        edges.drop("product_code", axis=1, inplace=True)

        # Add trading aggrements
        edges = edges.merge(trading_agreements, on=["src", "tgt"], how="left")
        edges.loc[:, "trade_agreement"] = edges["trade_agreement"].fillna(0)

        # Scale
        edges.loc[:, "num_prods"] = edges.loc[:, "num_prods"].astype(float)
        edges.loc[:, "num_prods"] = scaler.fit_transform(edges[["num_prods"]])
        edges.loc[:, f"{transaction}_value"] = np.log(edges[[f"{transaction}_value"]])
        edges.loc[:, "trade_agreement"] = scaler.fit_transform(edges[["trade_agreement"]])
        
        node_features.to_csv(f"../../data/5. Graphs Data/{transaction}/node_features-{year}-{product_code}-{transaction}.csv", index=False)
        edges.to_csv(f"../../data/5. Graphs Data/{transaction}/edge_features-{year}-{product_code}-{transaction}.csv", index=False)
        