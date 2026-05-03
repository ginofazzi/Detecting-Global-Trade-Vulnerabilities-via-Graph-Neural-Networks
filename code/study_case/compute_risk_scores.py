####### COMPILE ALL GRAPHS #######
import numpy as np
import pandas as pd
import sys;sys.path.append("../gnn");sys.path.append("../networks")
from utils import resolve_paths, load_atlas_data, get_product_list
from gnnutils import *

'''
Reads the probabilities and import values, and computes risk scores as probability * import value. 
Also aggregates to a country-year level by summing the risk scores of all commodities imported by a country in a year.
'''
READ_DATA_PATHS, WRITE_DATA_PATHS = resolve_paths(read_datasets=["Atlas Trade Data",
                                                                "Atlas Countries Data", 
                                                                "Atlas Products Data",
                                                                 "Graphs Data",
                                                                 "Results Data"], 
                                                write_datasets=["Graphs Data",
                                                                "Results Data"])

### SETTINGS ###
digits = 2
################

# Read imports volume data
atlas = load_atlas_data(READ_DATA_PATHS["Atlas Trade Data"])
# Read Atlas product ID to commodity mapping
products = pd.read_csv(READ_DATA_PATHS["Atlas Products Data"], dtype={"code": str})
products = products[["product_id", "code"]]

if digits == 2:
    products["code"] = products["code"].str[:2] ## Reduce product code to 2 digits

# # Get commodity from mapping
atlas = atlas.merge(products, on="product_id", how="left", indicator=False)

imports = atlas[["year", "country_id", "code", "import_value"]].groupby(["year", "country_id", "code"]).sum().reset_index()

for graph_type in ["export", "export-layered", "total"]:

    print(graph_type)

    # Get Country IDs into results
    country_ids = pd.DataFrame()

    for year in range(2012, 2023):
        for prod_code in get_product_list(digits=digits):
            _ = pd.read_csv(READ_DATA_PATHS["Graphs Data"] 
                            + f"/{digits}_digits/{graph_type.split('-')[0]}/node_features-{year}-{prod_code}-{graph_type.split('-')[0]}.csv")[["country_id"]]
            _["year"] = year
            _["commodity"] = prod_code
            _ = _.reset_index(drop=False, inplace=False).rename(columns={"index": "local_index"}, inplace=False)
            country_ids = pd.concat([country_ids, _], axis=0)

    for model in ["MLP", "GCN", "GAT", "SAGE"]:
        
        print(model)

        # Read results
        prob_results = pd.read_csv(READ_DATA_PATHS["Results Data"] + f"/Probabilities/{digits}_digits/{model}-{graph_type}-results.csv", 
                                   dtype={"commodity": str})
        prob_results["local_index"] = prob_results.groupby(["year", "commodity"]).cumcount()
        prob_results["commodity"] = prob_results["commodity"].str.zfill(digits)

        prob_results = prob_results.merge(country_ids, 
                                          on=["year", "commodity", "local_index"], 
                                          how="left", 
                                          validate="one_to_one", 
                                          indicator=False)
        #print(prob_results)
        assert len(prob_results[prob_results.country_id.isna()]) == 0, f"Some rows did not merge correctly: {len(prob_results[prob_results.country_id.isna()])}"
        prob_results.drop(columns=["local_index"], inplace=True)

        # Get import values into results
        prob_results = prob_results.merge(imports, 
                                          left_on=["year", "country_id", "commodity"], 
                                          right_on=["year", "country_id", "code"], 
                                          how="left", 
                                          indicator=False)
        prob_results.drop(columns=["code"], inplace=True)
        prob_results["risk_score"] = prob_results["probs"] * prob_results["import_value"]

        # Save results
        prob_results.to_csv(WRITE_DATA_PATHS["Results Data"] + f"/Risk Scores/{digits}_digits/{model}-{graph_type}-risk-scores.csv", index=False)

        # Aggregate to a country–year level
        yearly_risk = prob_results.groupby(["country_id", "year"])[["risk_score", "import_value"]].sum().reset_index()
        yearly_risk["average"] = yearly_risk["risk_score"] / yearly_risk["import_value"]
        yearly_risk.to_csv(WRITE_DATA_PATHS["Results Data"] + f"/Risk Scores/{digits}_digits/{model}-{graph_type}-yearly-risk-scores.csv", index=False)