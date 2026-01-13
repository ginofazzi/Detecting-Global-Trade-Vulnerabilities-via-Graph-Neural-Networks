import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sys;sys.path.append("./code")
from utils import *
import itertools

# Read data
df = pd.concat([pd.read_stata("../data/2. Atlas/hs12_country_country_product_year_4_2012_2016.dta"),
                pd.read_stata("../data/2. Atlas/hs12_country_country_product_year_4_2017_2021.dta"),
                pd.read_stata("../data/2. Atlas/hs12_country_country_product_year_4_2022.dta")])
products = pd.read_csv("../data/2. Atlas/product_hs12.csv")
countries = pd.read_csv("../data/2. Atlas/location_country.csv")

# Include product code
df = df.merge(products[["product_id", "code"]], how="left", on="product_id")
df.rename(columns={"code": "product_code"}, inplace=True)

df["product_code"] = df.product_code.str[:2] ## Reduce product code to 2 digits

# Collecter
df_impact_drop_exporters = pd.DataFrame()

# Years period
years = list(range(2012, 2023))

# Get the country-product matrix for a all years
compute_country_product_matrix_dict = {}

# Compute all years country_product matrices (export values)
for year in years:
    compute_country_product_matrix_dict[year] = pd.DataFrame(df.country_id.unique(), columns=["country_id"])\
        .merge(compute_country_product_matrix(df[df.year==year], product_col="product_code", value_col=f"export_value"), on="country_id", how="left").fillna(0)

#print(compute_country_product_matrix_dict)
# For each year up to 2021
for year in years[:-1]:

    # Get country-product matrices for the last 3 years
    year_0_prev = [compute_country_product_matrix_dict[i] for i in range(year, year-3, -1) if i in compute_country_product_matrix_dict]
    rca_exports_0 = compute_SRCA(year_0_prev) # Compute smoothed RCA
    rca_exports_0 = rca_exports_0.fillna(0)
    # SRCA for next year
    year_1_prev = [compute_country_product_matrix_dict[i] for i in range(year+1, year-2, -1) if i in compute_country_product_matrix_dict]
    rca_exports_1 = compute_SRCA(year_1_prev)
    rca_exports_1 = rca_exports_1.fillna(0)
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
            print(f"Country: {cid}: {countries.loc[countries.country_id==cid, 'name_short_en'].values[0]} | {pid}: {products.loc[products.code==pid, 'name_short_en'].values[0]}")
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

        df_impact_drop_exporters = pd.concat([df_impact_drop_exporters, subset])

        i += 1
        print(f"Year {year} {i/len(rca_matrix):.3%}")
        

df_impact_drop_exporters.to_csv("../data/df_impact_drop_exporters.csv", index=False)