# Program for creating demonstration NACE data
# Data is based on reeal Norwegian data with real description and NACE rev. 2.1
# NACE 2.0 is added based on a set of rules, 1-to-1 relationships and random choice

import pandas as pd
import numpy as np
from datetime import datetime
import os
import s3fs

from one_to_one import get_1to1_dict
from rules_classification import classify_rules, norwegian_rules
from ml_classification import get_possibles

# Set up paths to previous data and conversion excel file 
train_path = "https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/train_norwaydata_2026-01-13.parquet"
test_path = "https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/test_norwaydata_2026-01-13.parquet"
conversion_path = "https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/Cluster5/NACE2.1-NACE2_Table_V1.05.xlsx"

# Function for filling missing NACE with random choice
def fill_nace(row):
    poss = row['possibles']
    if pd.isna(row['nace_20_code']):
        if len(poss) > 0:
            return np.random.choice(poss)
        else:
            return np.nan   # fallback
    else:
        return row['nace_20_code']


if __name__ == "__main__":
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    print(f'Number of nace 2.0 codes needed: {train.shape[0]}')

    # Create text variable
    train['company_text'] = train.company_name + train.company_activity
    test['company_text'] = test.company_name + test.company_activity

    # 1:1 to one
    one_to_one_dict = get_1to1_dict(conversion_path)
    train["nace_20_code"] = train.nace_21_code.map(one_to_one_dict)
    test["nace_20_code"] = test.nace_21_code.map(one_to_one_dict)
    print(f'Number of nace 2.0 codes filled after 1-to-1: {train.nace_20_code.notna().sum()}')

    # Rules based
    train_rules = classify_rules(train, keyword_rules=norwegian_rules, company_nace21="nace_21_code", 
            company_text="company_name_description")
    test_rules = classify_rules(test, keyword_rules=norwegian_rules, company_nace21="nace_21_code", 
            company_text="company_name_description")

    train["nace_20_code"] = train["nace_20_code"].fillna(train_rules)
    test["nace_20_code"] = test["nace_20_code"].fillna(test_rules)

    print(f'Number of nace 2.0 codes filled after rules: {train.nace_20_code.notna().sum()}')

    # Get possible NACE codes
    train["possibles"] = get_possibles(train, "nace_21_code", conversion_path)
    test["possibles"] = get_possibles(test, "nace_21_code", conversion_path)

    # Random choice of one possible code
    train['nace_20_code'] = train.apply(fill_nace, axis=1)
    test['nace_20_code'] = test.apply(fill_nace, axis=1)

    print(f'Number of nace 2.0 codes filled after random choice: {train.nace_20_code.notna().sum()}')

    # Save data
    datestamp = datetime.now().strftime("%Y-%m-%d")
    s3_endpoint_url = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': s3_endpoint_url})

    bucket_out = "projet-aiml4os-wp10"
    filename_train = f"{bucket_out}/Cluster5/train_doublenace_{datestamp}.parquet"
    filename_test = f"{bucket_out}/Cluster5/test_doublenace_{datestamp}.parquet"

    with fs.open(filename_train, 'w') as file_out:
        train.to_parquet(file_out)
    with fs.open(filename_test, 'w') as file_out:
        test.to_parquet(file_out)