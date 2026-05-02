###############################
# SRCA #
################################

# Basic
import os
import pickle

# Pandas, Numpy
import pandas as pd
import numpy as np

# Custom
from utils import *


READ_DATA_PATHS, WRITE_DATA_PATHS = resolve_paths(read_datasets=["Graphs Data"], write_datasets=["Graphs Data"])

## SETTINGS ##
digits = 2 # 2 or 4

# Load CPM
with open(READ_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/CPM.pickle", 'rb') as f:
    compute_country_product_matrix_dict = pickle.load(f)

rca_dict = {}

# SRCA
for year in range(2012, 2023):
    year_n_prev = [compute_country_product_matrix_dict[i] for i in range(year, year-3, -1) if i in compute_country_product_matrix_dict]
    rca = compute_SRCA(year_n_prev)
    rca_dict[year] = rca.fillna(0)


# Write SRCA
with open(WRITE_DATA_PATHS["Graphs Data"] + f"/{digits}_digits/SRCA.pickle", 'wb') as f:
    pickle.dump(rca_dict, f)