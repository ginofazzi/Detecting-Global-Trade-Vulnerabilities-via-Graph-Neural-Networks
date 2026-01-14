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


# Load CPM
with open('../../data/CPM_4d.pickle', 'rb') as f:
    compute_country_product_matrix_dict_4d = pickle.load(f)
with open('../../data/CPM_2d.pickle', 'rb') as f:
    compute_country_product_matrix_dict_2d = pickle.load(f)


rca_dict_4d = {}
rca_dict_2d = {}

# SRCA
for year in range(2012, 2023):
    year_n_prev_4d = [compute_country_product_matrix_dict_4d[i] for i in range(year, year-3, -1) if i in compute_country_product_matrix_dict_4d]
    rca_4d = compute_SRCA(year_n_prev_4d)
    rca_dict_4d[year] = rca_4d.fillna(0)

    year_n_prev_2d = [compute_country_product_matrix_dict_2d[i] for i in range(year, year-3, -1) if i in compute_country_product_matrix_dict_2d]
    rca_2d = compute_SRCA(year_n_prev_2d)
    rca_dict_2d[year] = rca_2d.fillna(0)


# Write SRCA
with open('../../data/SRCA_4d.pickle', 'wb') as f:
    pickle.dump(rca_dict_4d, f)

with open('../../data/SRCA_2d.pickle', 'wb') as f:
    pickle.dump(rca_dict_2d, f)