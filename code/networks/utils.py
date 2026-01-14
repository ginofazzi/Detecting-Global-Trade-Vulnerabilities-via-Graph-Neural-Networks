
# UTILS
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from numpy import inf
import math
import urllib3
import json
import networkx as nx


def getBilateralData(subscription_key, typeCode, freqCode, clCode, period, reporterCode, cmdCode,
                   flowCode, partnerCode, includeDesc, format_output="JSON"):
    '''
    Function to retrieve Bilateral Data from UN Comtrade. It caps the results to 100k rows.
    '''

    baseURL = 'https://comtradeapi.un.org/tools/v1/getBilateralData/' + \
                typeCode + '/' + freqCode + '/' + clCode

    PARAMS = dict(reportercode=reporterCode, flowCode=flowCode,
                  period=period, cmdCode=cmdCode, partnerCode=partnerCode,
                  format=format_output, includeDesc=includeDesc)
    
    PARAMS["subscription-key"] = subscription_key
    fields = dict(filter(lambda item: item[1] is not None, PARAMS.items()))

    http = urllib3.PoolManager()
    if format_output is None:
        format_output = 'JSON'
    if format_output != 'JSON':
        print("Only JSON output is supported with this function")
    else:
        try:
            resp = http.request("GET", baseURL, fields=fields, timeout=120)
            if resp.status != 200:
                print(resp.data.decode('utf-8'))
            else:
                jsonResult = json.loads(resp.data)
                df = pd.json_normalize(jsonResult['data'])
                return df
        except urllib3.exceptions.RequestError as err:
            print(f'Request error: {err}')


def compute_country_product_matrix(itd_df, product_col="product_id", value_col="export_value"):
    # Takes a International Trade Data DF with
    # at least the columns 'country_id', 'product_id' and "export_value"
    df = itd_df[["country_id", product_col, value_col]].groupby(["country_id", product_col]).sum(value_col).reset_index()
    return df.pivot_table(index="country_id", values=value_col, columns=product_col).fillna(0)



def compute_SRCA(country_product_matrices: list):
    '''
    Smoothed RCA, following https://observatory-economic-complexity.github.io/oec-documentation/the-mathematics-of-economic-complexity.html#revealed-comparative-advantage-rca
    '''
    country_product_matrices = [df.set_index("country_id") for df in country_product_matrices]
    df_t = country_product_matrices[0]  # Most recent year
    N = len(country_product_matrices) - 1  # Number of past years

    # Compute smoothed exports: (2*x_t + sum of past years) / (2 + N)
    X_tilde = (2 * df_t + sum(country_product_matrices[1:])) / (2 + N)

    # Compute sums needed for SRCA
    sum_c_X_tilde = X_tilde.sum(axis=0)  # Sum over countries (columns: products)
    sum_p_X_tilde = X_tilde.sum(axis=1).to_numpy().reshape(-1, 1)  # Sum over products (rows: countries)
    total_sum_X_tilde = X_tilde.values.sum()  # Overall sum

    # Compute SRCA
    SRCA = (X_tilde / sum_c_X_tilde) / (sum_p_X_tilde / total_sum_X_tilde)

    return pd.DataFrame(SRCA, index=df_t.index, columns=df_t.columns)



def rca_(country_product_matrix_dict: dict): ## DEPRECATED

    rca_per_year = {}
    # Takes a dictionary of year: country_product_matrix
    # and returns a dictionary of year: RCA matrices (same shape as country_product)
    for year in sorted(country_product_matrix_dict.keys()):

        # The Sum of column for Product P: Total trade for Product P across all countries
        colP = country_product_matrix_dict[year].sum(axis=0)

        # We average the denominator by the three previous years
        prev_years = [k for k in country_product_matrix_dict if k in range(year-3, year)]

        # If the year is the first, we can't average so we use the current year
        if len(prev_years) == 0:
            # Matrix row containing all product exports for country in year i
            rowC = country_product_matrix_dict[year].sum(axis=1)
            # Total trade for year i
            sumM = country_product_matrix_dict[year].sum().sum()
            denominator = (rowC / sumM).T
        
        # If there are multiple years, average the denominator
        else:
            denominator = 0
            for prev_year in prev_years:
                # Matrix row containing all product exports for country in year i
                rowC = country_product_matrix_dict[prev_year].sum(axis=1)
                # Total trade for year i
                sumM = country_product_matrix_dict[prev_year].sum().sum()
                # Denominator for a single non-averaged year
                denominator += (rowC / sumM).T
            denominator /= len(prev_years)

        rca_per_year[year] = (country_product_matrix_dict[year] / colP).div(denominator, axis=0)
        rca_per_year[year][rca_per_year[year] == inf] = 0
        rca_per_year[year].fillna(0, inplace=True)

    return rca_per_year

    

def compute_lost_exporters(df, years, import_volumes, drop_threshold=0.2):
    '''
    Function to compute all lost exporters and their potentially affected importers.

    Arguments:
     - df: An ATLAS formatted dataframe, containing country_id, product_id, year and export_value
     - years: The years required for the computations
     - 
    '''
    # Exports
    total_exportes_per_country_product = df[["country_id", "product_id", "year", "export_value"]].groupby(["country_id", "product_id", "year"]).sum().reset_index()

    # Compute all years country_product matrices for period
    compute_country_product_matrix_dict = {}

    for year in years:
        compute_country_product_matrix_dict[year] = compute_country_product_matrix(df[df.year==year], col="export_value")

    rca_dict = rca(compute_country_product_matrix_dict)

    # Compute the RCA changes in period
    rca_matrix_all_years = pd.DataFrame()
    for year in years[:-1]:
        rca_exports_0 = rca_dict[year]
        rca_exports_1 = rca_dict[year+1]
        # Compute the change in RCA
        rca_change = (rca_exports_1 / rca_exports_0) - 1
        # Filter for Country-Product pairs that satisfy: Stopped being "EXPORTERS" and dropped by 20%
        rca_change = rca_change[(rca_exports_1<1) & (rca_exports_0>=1) & (rca_change<=-drop_threshold)] ## CONDITION FOR LOSING 'EXPORTER STATUS'
        rca_matrix = rca_change.stack(level="product_id").reset_index().rename(columns={0: "rca"})
        rca_matrix["prev_year"] = year
        rca_matrix["year"] = year+1
        rca_matrix_all_years = pd.concat([rca_matrix_all_years, rca_matrix])

    # Additional constraint: Lost Exporters must have decreased their exports for product p
    rca_matrix_all_years = rca_matrix_all_years.merge(total_exportes_per_country_product, on=["country_id", "product_id", "year"], how="left")\
        .merge(total_exportes_per_country_product, left_on=["country_id", "product_id", "prev_year"], right_on=["country_id", "product_id", "year"], how="left", suffixes=[None, "_prev_year"])
    rca_matrix_all_years[["export_value", "export_value_prev_year"]] = rca_matrix_all_years[["export_value", "export_value_prev_year"]].fillna(0)
    rca_matrix_all_years = rca_matrix_all_years[rca_matrix_all_years.export_value < rca_matrix_all_years.export_value_prev_year]
    rca_matrix_all_years.drop(["year_prev_year", "export_value", "export_value_prev_year"], axis=1, inplace=True)
    #display(rca_matrix_all_years)
    # Include previous year importers
    all_importers = df[df.export_value > 0] # Filter only for positive values in export, leaving only country-product exports to other countries

    # For the Country-Product combinations where a Country "Dropped" as a main exporter, join all affected importers for year n
    lost_exporters_prev = rca_matrix_all_years.merge(all_importers[["country_id", "partner_country_id", "year", "product_id", "export_value"]], 
            left_on=["country_id", "product_id", "prev_year"], right_on=["country_id", "product_id", "year"], how="left")
    lost_exporters_prev.rename(columns={"rca": "rca_drop", "year_x": "year", "partner_country_id": "importer_id", "export_value": "prev_year_export_value"}, inplace=True)
    lost_exporters_prev.drop("year_y", axis=1, inplace=True)
    
    # For the Country-Product combinations where a Country "Dropped" as a main exporter, join all affected importers for year n+1
    lost_exporters = rca_matrix_all_years.merge(all_importers[["country_id", "partner_country_id", "year", "product_id", "export_value"]], 
            left_on=["country_id", "product_id", "year"], right_on=["country_id", "product_id", "year"], how="left")
    lost_exporters.rename(columns={"rca": "rca_drop", "partner_country_id": "importer_id"}, inplace=True)

    # Join lost exporters from year n and n+1
    lost_exporters = lost_exporters_prev\
        .merge(lost_exporters, on=["country_id", "product_id", "rca_drop", "prev_year", "year", "importer_id"], how="inner")\
            .sort_values(["country_id", "product_id", "prev_year", "importer_id"])


    # Include import volumes for importers
    lost_exporters = lost_exporters.merge(import_volumes, left_on=["importer_id", "product_id", "prev_year"], right_on=["importer_id", "product_id", "year"], how="left")
    lost_exporters.rename(columns={"import_value": "prev_year_importer_volume", "year_x": "year"}, inplace=True)
    lost_exporters.drop(["year_y"], axis=1, inplace=True)
    lost_exporters = lost_exporters.merge(import_volumes, left_on=["importer_id", "product_id", "year"], right_on=["importer_id", "product_id", "year"], how="left")
    lost_exporters.rename(columns={"import_value": "importer_volume"}, inplace=True)

    lost_exporters["affected"] = (lost_exporters.export_value < lost_exporters.prev_year_export_value) & (lost_exporters.importer_volume < lost_exporters.prev_year_importer_volume)

    return lost_exporters



### BUSTOS-YILDRIM IMPLEMENTATION
def corrected_values(df, export_column, import_column, disc_ratio=0.08):
    # Basically correct CIF-to-FOB values. Source: https://observatory-economic-complexity.github.io/oec-documentation/data-processing.html
    df["modif_imports"] = df[import_column] / (1 + disc_ratio)
    df["modif_exports"] = df[import_column] / (1 + disc_ratio)
    df.loc[:, export_column] = (df.loc[:, [export_column, "modif_imports"]]).max(axis=1)
    df.loc[:, import_column] = (df.loc[:, [export_column, "modif_imports"]]).max(axis=1)


# Function to assign reliability scores (simplified)
def assign_reliability_scores(data):
    reliability = data.groupby('Reporter')['Value'].std()
    reliability = reliability / reliability.max()  # Normalize between 0 and 1
    return reliability.to_dict()


# Function to reconcile trade values
def reconcile_trade(data, reliability_scores):
    reconciled = []
    pairs = data.groupby(['Reporter', 'Partner'])
    
    for (reporter, partner), group in pairs:
        if len(group) < 2:
            continue  # Skip if there's no bilateral report
        
        imports = group[group['Trade Flow'] == 'Import']['Value'].values
        exports = group[group['Trade Flow'] == 'Export']['Value'].values
        
        if len(imports) == 0 or len(exports) == 0:
            continue  # Skip incomplete pairs
        
        rep_score = reliability_scores.get(reporter, 0.5)
        par_score = reliability_scores.get(partner, 0.5)
        
        weight_rep = rep_score / (rep_score + par_score)
        weight_par = par_score / (rep_score + par_score)
        
        reconciled_value = (weight_rep * imports[0]) + (weight_par * exports[0])
        
        reconciled.append({'Reporter': reporter, 'Partner': partner, 'Reconciled Value': reconciled_value})
    
    return pd.DataFrame(reconciled)


# Main function to clean trade data
def clean_trade_data(trade_data):
    reliability_scores = assign_reliability_scores(trade_data)
    cleaned_data = reconcile_trade(trade_data, reliability_scores)
    return cleaned_data


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping



def gini_coefficient(trade_values):
    """
    Compute Gini coefficient for trade distribution.
    trade_values: List or array of trade values (USD) with partners.
    """
    trade_values = np.sort(trade_values)  # Sort trade values
    n = len(trade_values)
    if n == 1:
        return 1  # If no trade partners, Gini = 1 (total dependency)
    
    index = np.arange(1, n+1)  # Rank of each trade partner
    numerator = 2 * np.sum(index * trade_values)
    denominator = n * np.sum(trade_values)
    
    return (numerator / denominator) - (n + 1) / n


def gini_to_risk(g, n, N, alpha, beta):

    #N = max(n, N)

    #return (g+(alpha*(1-g)))**math.log(N/(N-n+1))
    return ((g+(alpha*(1-g))) ** (beta * (n/N)))
    


def risk_score(trade_values, N, alpha=0.01, beta=5):
    
    g = gini_coefficient(trade_values)

    return gini_to_risk(g=g, n=len(trade_values), N=N, alpha=alpha, beta=beta)


def compute_hhi(trade_shares):
    """Compute Herfindahl-Hirschman Index given trade shares."""
    return np.sum(np.square(trade_shares))


def hhi_risk(trade_values, trade_col="total_trade"):
    # Compute trade share in %
    trade_volume_share = trade_values.merge(trade_values.groupby("country_id")[trade_col].sum().reset_index().rename(columns={trade_col: "total_country_trade"}),\
                                            on="country_id", how="left")
    trade_volume_share["trade_share"] = trade_volume_share[trade_col] / trade_volume_share.total_country_trade
    # Compute HHI
    risk_hhi = trade_volume_share.groupby("country_id")["trade_share"].apply(compute_hhi).reset_index().rename(columns={"trade_share": "risk"})

    return risk_hhi


def risk_label(risk, risk_zones):
    if risk < risk_zones[0]:
        return "Diversified"
    elif risk > risk_zones[1]:
        return "Highly Dependent"
    else:
        return "Moderate"
    


def get_total_trade(df):

    trade_volumes = df[["country_id", "partner_country_id", "export_value", "import_value"]].groupby(["country_id", "partner_country_id"]).sum().reset_index()
    trade_volumes["total_trade"] = (trade_volumes.export_value + trade_volumes.import_value)

    return trade_volumes


def get_trade_partners(df):
    
    trade_partners = df[["country_id", "partner_country_id"]].groupby("country_id").nunique().reset_index().rename(columns={"partner_country_id": "num_partners"})

    return trade_partners


##### TRUSTWORTHINESS
def compute_initial_trustworthiness(df):
    # Sum all products for country-partner pairs
    subset = df[["country_id", "partner_country_id", "export_value", "import_value"]].groupby(["country_id", "partner_country_id"]).sum().reset_index()
    # Merge to get both directions of reports
    subset = subset.merge(subset, left_on=["country_id", "partner_country_id"], right_on=["partner_country_id", "country_id"], how="inner")
    # Calculate Mismatch M for all Country pairs
    M = abs(subset["export_value_x"] - subset["import_value_y"]) + abs(subset["import_value_x"] - subset["export_value_y"])
    # Calculate All transactions volume (Psi) for all country pairs
    Psi = subset[["export_value_x", "import_value_x", "export_value_y", "import_value_y"]].sum(axis=1)
    # Calculate Trustwothiness at N=0 (T0)
    T0 = 1-(M/Psi)

    subset["T_ab"] = T0

    return subset[["country_id_x", "partner_country_id_x", "T_ab"]]


def collapse_edges(df, source_col="country_id_x", target_col="partner_country_id_x"):
    df = df.copy()
    # Collapse edges since now it's undirected
    df["edge_pair"] = df[[source_col, target_col]].min(axis=1).astype(str) + "-" + df[[source_col, target_col]].max(axis=1).astype(str)
    df.drop_duplicates(subset="edge_pair", keep="first", inplace=True)
    df.drop("edge_pair", axis=1, inplace=True)
    return df


def compute_trustworthiness(df, n_iter=100):

    P_prime = compute_initial_trustworthiness(df)
    T = P_prime[["country_id_x", "T_ab"]].groupby("country_id_x").mean().rename(columns={"T_ab": "T_a"})
    P_prime = P_prime.merge(T, on="country_id_x", how="left").merge(T.rename(columns={"T_a": "T_b"}), left_on="partner_country_id_x", right_on="country_id_x", how="left")
    
    avg_trust = []

    for i in range(n_iter):
        P_prime["T_ab"] = (P_prime["T_a"] / P_prime["T_b"]) / ((P_prime["T_a"] / P_prime["T_b"]) + (1-P_prime["T_a"]))
        # Recompute Trustworthiness of nodes
        P_prime.drop(["T_a", "T_b"], axis=1, inplace=True)
        T = P_prime[["country_id_x", "T_ab"]].groupby("country_id_x").mean().rename(columns={"T_ab": "T_a"})
        P_prime = P_prime.merge(T, on="country_id_x", how="left").merge(T.rename(columns={"T_a": "T_b"}), left_on="partner_country_id_x", right_on="country_id_x", how="left")
        avg_trust.append(P_prime["T_a"].mean())

    return P_prime, avg_trust

# For Geo Embeddings #
# Country parser
def parse_country_neighbors(filepath):

    df = pd.read_csv(filepath, sep="\t", na_filter=False)
    df.neighbours = df.neighbours.fillna("")
    G = nx.Graph()

    for ix, row in df.iterrows():

        country_code = row["ISO3"]
        neighbors = row.neighbours.split(',')
        for neighbor in neighbors:
            if neighbor != "":
                G.add_edge(country_code, df.loc[df["#ISO"] == neighbor, "ISO3"].values[0], weight=1)
    return G



def distance_to_weight(distances, w_min=0.05, w_max=0.95):
    distances = np.array(distances, dtype=float)
    d_min = distances.min()
    d_max = distances.max()

    # Avoid division by zero
    if d_max == d_min:
        return np.full_like(distances, (w_min + w_max) / 2)

    # Linear scaling and reversing
    weights = w_min + (w_max - w_min) * (1 - (distances - d_min) / (d_max - d_min))
    return weights



def maritime_distance_to_weight(distances, max_distance=20037.5, epsilon=0.01):
    distances = np.clip(np.array(distances), 0, max_distance)
    weights = epsilon + (1 - epsilon) * (1 - distances / max_distance)
    return weights
