import numpy as np
import pandas as pd
import sys;sys.path.append("./code")
from utils import *
import itertools
import pickle

READ_DATA_PATHS, WRITE_DATA_PATHS = resolve_paths(read_datasets=["Atlas Trade Data",
                                                                "Atlas Countries Data", 
                                                                "Atlas Products Data",
                                                                 "Graphs Data"], 
                                                write_datasets=["Graphs Data"])

### SETTINGS ###
digits = 4 # 2 or 4
#################

# Read data
df = load_atlas_data(READ_DATA_PATHS["Atlas Trade Data"])
products = pd.read_csv(READ_DATA_PATHS["Atlas Products Data"], dtype={"code": str})
countries = pd.read_csv(READ_DATA_PATHS["Atlas Countries Data"], encoding="latin1")
# Include product code
df = df.merge(products[["product_id", "code"]], how="left", on="product_id")
df.rename(columns={"code": "product_code"}, inplace=True)

if digits == 2:
    df["product_code"] = df.product_code.str[:2] ## Reduce product code to 2 digits

# Years period
years = list(range(2012, 2023))

# Get the country-product matrix for a all years
with open(READ_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/CPM.pickle", "rb") as f:
    compute_country_product_matrix_dict = pickle.load(f)

with open(READ_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/SRCA.pickle", "rb") as f:
    rca_dict = pickle.load(f)

# Collector
subsets = []
# For each year up to 2021
for year in years[:-1]:

    rca_exports_0 = rca_dict[year]
    rca_exports_1 = rca_dict[year+1]
    # Compute the change in RCA
    rca_change = (rca_exports_1 / rca_exports_0) - 1
    rca_change.replace([np.inf, -np.inf], 0, inplace=True)
    rca_change.fillna(0, inplace=True)
    # Filter for Country-Product pairs that satisfy: Stopped being "EXPORTERS" and dropped by 20%
    rca_change = rca_change[(rca_exports_1<=1) & (rca_exports_0>=1) & (rca_change<-0.2)]
    rca_matrix = rca_change.stack().reset_index().rename(columns={0: "rca", "level_1": "product_code"})
    #display(rca_matrix)
    i=0
    for ix, row in rca_matrix.iterrows():
        cid = row["country_id"]
        pid = row["product_code"]
        drop = row["rca"]
        try:
            print(f"Country: {cid}: {countries.loc[countries.country_id==cid, 'country'].values[0]} | {pid}: {products.loc[products.code==pid, 'name_short_en'].values[0]}")
        
        except:
            print(f"Country: {cid} | {pid}")
            
        # Affected importers
        importers = df.loc[(df.country_id == cid) & (df.product_code == pid) & (df.year == year) & (df.export_value > 0), "partner_country_id"].values
        # If no affected importers, continue
        if len(importers) == 0:
            continue

        subset = df[(df.country_id.isin(importers)) & (df.product_code==pid) & (df.year.isin([year, year+1]))]\
            .groupby(["country_id", "year"]).agg({"partner_country_id": "count", "import_value": "sum"}).reset_index()

        # Create a complete set of country-year pairs
        all_combinations = pd.DataFrame(list(itertools.product(importers, [year, year + 1])), columns=["country_id", "year"])

        # Merge with the subset to ensure all combinations exist
        subset = all_combinations.merge(subset, on=["country_id", "year"], how="left").fillna(0)

        # Ensure correct data types
        subset["partner_country_id"] = subset["partner_country_id"].astype(int)
        subset["import_value"] = subset["import_value"].astype(float)

        subset = pd.pivot_table(subset, 
                                index=["country_id"], 
                                values=["partner_country_id", "import_value"],
                                columns="year").reset_index()

        subset.columns = ["affected_importer", "year_from_value", "year_to_value", "year_from_n_exporters", "year_to_n_exporters" ]
        subset[["year_from", "year_to", "pid", "exporter_id", "drop_pct"]] = year, year+1, pid, cid, drop

        subsets.append(subset)
        
        i += 1
        print(f"Year {year} {i/len(rca_matrix):.3%}")
        

# Concatenate all subsets and save
df_impact_drop_exporters = pd.concat(subsets, ignore_index=True)
df_impact_drop_exporters.to_csv(WRITE_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/df_impact_drop_exporters.csv", index=False)