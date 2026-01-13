
'''
Get the bilateral discrepancies for all countries and all years, from UN Comtrade, using the UNCOMTRADE API. 
Used to compute the Trustworthiness index.
'''

from utils import *


# Discrepancies
api_key = "<YOUR-API-KEY-HERE>"

# Read full list of reporter codes (countries)
reporters = pd.read_csv(f"../data/1. UN Comtrade/reporters.csv", dtype=str)
# Filter out old country denominations (e.g., Yugoslavia). They usually have the "(... former X)"
reporters = reporters[~reporters.text.str.contains("(...", regex=False)]
# Parse the country ids for the API
reporters = list(reporters.id)
reporters = ",".join(map(str, reporters))

# Select the columns to keep
cols = ["period", "reporterCode", "flowCode", "partnerCode", "cmdCode", "mirrorQty", "mirrorNetWgt", "cifvalue", "fobvalue", "primaryValue", "mirrorPrimaryValue", "primaryValueDisc", "mirror2PrimaryValue"]

# Collect all DFs
all_dfs = []

# Go year-wise
for year in range(2012, 2023):
    # Codes
    i = 1
    while i < 100:

        print(year)

        if i == 91:
            j=99
        else:
            j=i+9

        cmdCode = ",".join([f'{x:02d}' for x in range(i,j+1)])
        print(cmdCode)

        df = getBilateralData(subscription_key=api_key, typeCode='C', freqCode='A', clCode='HS', period=year,\
                               reporterCode=reporters, cmdCode=cmdCode, flowCode="X", partnerCode=reporters, includeDesc=False)
        df = df[cols]
        all_dfs.append(df)

        i+=10 # Let's go every 10 CMD codes

# Save all DFs to disk
all_dfs = pd.concat(all_dfs)
all_dfs.to_parquet("bilateral.parquet", index=False)