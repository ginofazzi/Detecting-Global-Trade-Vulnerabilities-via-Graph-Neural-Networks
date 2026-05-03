# Detecting Global Trade Vulnerabilities via Graph Neural Networks

This repository contains the research code and data used for the paper "Detecting Global Trade Vulnerabilities via Graph Neural Networks".

## Overview

Global trade is a highly interconnected and fragile system. This project uses graph-based representations of international trade and Graph Neural Networks (GNNs) to identify countries and country-product pairs that are at risk of experiencing trade disruptions.

The main idea is to move beyond traditional econometric models and capture systemic complexity using a node classification framework over multilayer trade graphs.

## Key Contributions

- First application of GNNs specifically for classifying countries as at risk of disruptions in the global trade network.
- Comparison of multiple graph representations, including multilayer and multigraph models, to determine which structure is most useful for vulnerability prediction.
- Ablation studies showing the value of network topology over attribute-only baselines.
- A framework to aggregate model predictions into a global trade risk score.

## Approach

The research frames trade vulnerability assessment as two related prediction tasks:

1. **Country vulnerability**: Predict whether a country is at risk of any trade disruption in the current year.
2. **Country-product vulnerability**: Predict whether a specific importer-product pair will experience disruption.

### Graph representations

The repository uses several trade network views:

- **Multi-Layer Export**: directed multilayer graph where each layer represents exports for a commodity class.
- **Multi-Layer Total**: undirected multilayer representation that aggregates import and export links.
- **Multi-Graph Export**: flattened multigraph with parallel commodity-specific edges.
- **Multi-Graph Total**: undirected multigraph that summarizes trade between countries for each product.

### Node and edge features

Node features include:

- Economic Complexity Index (ECI)
- Complexity Outlook Index (COI)
- Smoothed Revealed Comparative Advantage (SRCA)
- Geo-positional embeddings
- Herfindahl–Hirschman index (HHI)
- Trustworthiness index (TI)

Edge features include:

- Transaction volume
- Product Complexity Index (PCI)
- Number of distinct products traded
- Number of active trade agreements
- Trustworthiness index for bilateral reporting reliability

### Models

The repository evaluates several architectures:

- Graph Convolutional Network (GCN)
- Graph Attention Network (GATv2)
- GraphSAGE
- Multi-Layer Perceptron (MLP) as a non-graph baseline
- Additional classical baselines such as Random Forest and XGBoost

## Results Summary

The paper demonstrates that graph-based models outperform attribute-only baselines in trade vulnerability prediction, especially when the network representation is chosen carefully.

- GNNs show improved F1 scores over MLP for country-level prediction.
- Simpler network representations can outperform more complex structures when predicting country-product risks.
- A global risk score derived from model outputs provides useful insight into real-world shocks such as the 2021–2022 Russia sanctions.

## Data Sources

This project integrates multiple datasets, including:

- UN Comtrade data
- Atlas of Economic Complexity (ECI, COI, PCI)
- BACI (unit prices, metric-ton normalization)
- Global Preferential Trade Agreements Database
- Trade discrepancy and trustworthiness indexes
- Border adjacency and maritime distance data

The cleaned and processed graph data are stored under `data/5. Graphs Data/`.

### Data collection

This project relies on multiple external data sources, most of which are not included in this repository. This is due to copyright restrictions and file size limitations. Instead, we provide links and instructions so you can download the data and organize it locally.

After downloading the datasets, you must specify their locations in DATA_PATHS.yaml. A recommended directory structure is provided in that file.

1. UN Comtrade data
Reference lists (reporters, products):
https://comtradeplus.un.org/ListOfReferences
Bilateral trade data:
Retrieve via the API using the script: `at code/networks/1. Data - collect_discrepancies.py`

2. Atlas of Economic Complexity
- Data portal:
    https://atlas.hks.harvard.edu/data-downloads
- Required files:
    `hs12_country_country_product_year_4_2012_2016.dta`
    `hs12_country_country_product_year_4_2017_2021.dta`
    `hs12_country_country_product_year_4_2022.dta`
    `countries.csv`
    `product_hs12.csv`
    `location_country.csv`

3. For BACI data:
    - Data portal: https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37
    - Required files::
        - `BACI_HS12_Y2012_V202501.csv`
        - `BACI_HS12_Y2013_V202501.csv`
        - `BACI_HS12_Y2014_V202501.csv`
        - `BACI_HS12_Y2015_V202501.csv`
        - `BACI_HS12_Y2016_V202501.csv`
        - `BACI_HS12_Y2017_V202501.csv`
        - `BACI_HS12_Y2018_V202501.csv`
        - `BACI_HS12_Y2019_V202501.csv`
        - `BACI_HS12_Y2020_V202501.csv`
        - `BACI_HS12_Y2021_V202501.csv`
        - `BACI_HS12_Y2022_V202501.csv`
        - `BACI_HS12_Y2023_V202501.csv`
        - `country_codes_V202501.csv`
        - `product_codes_HS12_V202501.csv`
        
4. Additional data sources:
Download the following publicly available datasets:

| File | Link |
| --- | --- |
| `Trading agreements.xlsx` | https://wits.worldbank.org/gptad/library.aspx |
| `DISCREPANCY_INDEX_H5_2017_csv.zip` | https://datacatalog.worldbank.org/search/dataset/0064901/Discrepancy-Index-H5 |
| `DISCREPANCY_INDEX_H5_2018_csv.zip` | https://datacatalog.worldbank.org/search/dataset/0064901/Discrepancy-Index-H5 |
| `DISCREPANCY_INDEX_H5_2019_csv.zip` | https://datacatalog.worldbank.org/search/dataset/0064901/Discrepancy-Index-H5 |
| `DISCREPANCY_INDEX_H5_2020_csv.zip` | https://datacatalog.worldbank.org/search/dataset/0064901/Discrepancy-Index-H5 |
| `DISCREPANCY_INDEX_H5_2021_csv.zip` | https://datacatalog.worldbank.org/search/dataset/0064901/Discrepancy-Index-H5 |
| `DISCREPANCY_INDEX_H5_2022_csv.zip` | https://datacatalog.worldbank.org/search/dataset/0064901/Discrepancy-Index-H5 |
| `Corruption Perception Index.csv` | https://www.transparency.org/en/cpi/2023/index/dnk |
| `Country Borders.txt` | http://download.geonames.org/export/dump/countryInfo.txt |
| `CERDI-seadistance.xlsx` | https://zenodo.org/records/46822#.VvFcNWMvyjp |

Once downloaded, you should include the paths to each file in the DATA_PATHS.yaml. A recommended path structure is provided in the file.


## Repository Structure

- `code/networks/`: Data cleaning, graph generation, network representations, and feature engineering.
- `code/gnn/`: GNN models, training/testing code, ablation analysis, and optimization scripts.
- `data/`: Raw and processed datasets used to build the trade graphs.
- `main.pdf`: The research paper describing the methodology and results.

## Running the Code

The main training and evaluation script is:

- `code/gnn/train-test-runs.py`

Important setup notes:

- The script reads preprocessed graphs from `data/5. Graphs Data/`.
- Use the `model_type`, `graphs_type`, `layered`, and `multi_graph` variables at the top of `train-test-runs.py` to select the model and graph representation.
- Best hyperparameters are loaded from `code/gnn/models/best_params.json`.
- The script saves trained models and evaluation results under `code/gnn/models/training/` and `code/gnn/results/`.

### Dependencies

The repository relies on common data science and graph learning packages including:

- Python 3
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- PyTorch
- PyTorch Geometric
- optuna
- networkx

> Note: A requirements.txt file is included in the repository, that can be use to install via Conda.

## Reproducibility

The research is designed to be reproducible. The paper and code reference the same dataset construction, graph representations, and model evaluation procedures.

If you want to reproduce the experiments:

1. Download and prepare the raw trade and auxiliary data in `data/`.
2. Run the graph generation scripts in `code/networks/`.
3. Run `code/gnn/train-test-runs.py`.

## Contact

For questions about the research, see the author details in `main.pdf`.

---
