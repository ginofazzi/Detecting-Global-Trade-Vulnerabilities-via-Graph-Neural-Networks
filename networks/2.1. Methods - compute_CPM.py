###############################
# Country-Product Matrix (CPM) #
################################

# Basic
import os
import pickle

# Pandas, Numpy
import pandas as pd
import numpy as np

# Custom
from utils import *

############ SETTINGS ############
transaction = "export" # "import" or "export" or "total"
##################################

# Load Atlas data
df = pd.concat([pd.read_stata("../data/2. Atlas/hs12_country_country_product_year_4_2012_2016.dta"),
                pd.read_stata("../data/2. Atlas/hs12_country_country_product_year_4_2017_2021.dta"),
                pd.read_stata("../data/2. Atlas/hs12_country_country_product_year_4_2022.dta")])
products = pd.read_csv("../data/2. Atlas/product_hs12.csv", dtype={"code": str})


# Include product code
df = df.merge(products[["product_id", "code"]], how="left", on="product_id")
df.rename(columns={"code": "product_code"}, inplace=True)

# Include total trade value
df["total_value"] = df.export_value + df.import_value

##### CPM at 4-digit level #####
# Compute all years country_product matrices for period
compute_country_product_matrix_dict = {}

for y in range(2012, 2023):
    compute_country_product_matrix_dict[y] = pd.DataFrame(df.country_id.unique(), columns=["country_id"])\
        .merge(compute_country_product_matrix(df[df.year==y], product_col="product_code", \
                                              value_col=f"{transaction}_value"), on="country_id", how="left").fillna(0)
    
# Save CPMs
with open('../data/CPM_4d.pickle', 'wb') as f:
    pickle.dump(compute_country_product_matrix_dict, f)

##### CPM at 2-digit level #####
df["product_code"] = df.product_code.str[:2] ## Reduce product code to 2 digits
# Compute all years country_product matrices for period
compute_country_product_matrix_dict = {}

for y in range(2012, 2023):
    compute_country_product_matrix_dict[y] = pd.DataFrame(df.country_id.unique(), columns=["country_id"])\
        .merge(compute_country_product_matrix(df[df.year==y], product_col="product_code", \
                                              value_col=f"{transaction}_value"), on="country_id", how="left").fillna(0)
    
# Save CPMs
with open('../data/CPM_2d.pickle', 'wb') as f:
    pickle.dump(compute_country_product_matrix_dict, f)