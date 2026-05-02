###############################
# Country-Product Matrix (CPM) #
################################

# Basic
import os
import pickle
import yaml

# Pandas, Numpy
import pandas as pd
import numpy as np

# Custom
from utils import *

READ_DATA_PATHS, WRITE_DATA_PATHS = resolve_paths(read_datasets=["Atlas Trade Data", "Atlas Products Data"], write_datasets=["Graphs Data"])

############ SETTINGS ############
transaction = "export" # "import" or "export" or "total"
digits = 2 # 2 or 4
save_pickle = True
##################################

# Load Atlas data
if digits in [2, 4]:
    df = load_atlas_data(READ_DATA_PATHS["Atlas Trade Data"])
    
    # Add product code, since the old that has only product_id
    # NOTE: Remember that 'product_id' is the internal Atlas product id used in the Atlas project
    # whereas 'product_code' is the actual HS code that we use for our analysis
    products = pd.read_csv(READ_DATA_PATHS["Atlas Products Data"], dtype={"code": str})
    df = df.merge(products[["product_id", "code"]], how="left", on="product_id")
    df.rename(columns={"code": "product_code"}, inplace=True)

    if digits == 2:
        ##### CPM at 2-digit level #####
        df["product_code"] = df.product_code.str[:2] ## Reduce product code to 2 digits

else:
    raise ValueError("Digits must be 2 or 4")

# Include total trade value
df["total_value"] = df.export_value + df.import_value

# Compute all years country_product matrices for period
compute_country_product_matrix_dict = {}

for y in range(2012, 2023):
    print(y)
    compute_country_product_matrix_dict[y] = pd.DataFrame(df.country_id.unique(), columns=["country_id"])\
        .merge(compute_country_product_matrix(df[df.year==y], product_col="product_code", \
                                            value_col=f"{transaction}_value"), on="country_id", how="left").fillna(0)
    
# Save CPMs
if save_pickle:
    with open(f'{WRITE_DATA_PATHS["Graphs Data"]}/{digits}_digits/CPM.pickle', 'wb') as f:
        pickle.dump(compute_country_product_matrix_dict, f)