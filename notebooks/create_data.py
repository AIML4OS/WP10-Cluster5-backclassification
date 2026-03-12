import pandas as pd
from utils.one_to_one import get_1to1_dict
from utils.rules_classification import classify_rules, norwegian_rules
from config import path_info

train_path = "https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/train_norwaydata_2026-01-13.parquet"
test_path = "https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/test_norwaydata_2026-01-13.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)
print(f'Number of nace 2o codes needed: {train.shape[0]}')

# Create text variable
train['company_text'] = train.company_name + train.company_activity
test['company_text'] = test.company_name + test.company_activity

# check
print(f'Number with 24.54: {(train.nace_21_code == "24.54").sum()}')
print(f'Number with 14.24: {(train.nace_21_code == "14.24").sum()}')

# 1:1 to one
one_to_one_dict = get_1to1_dict(path_info["conversion_path"])
train["nace_20_code"] = train.nace_21_code.map(one_to_one_dict)
test["nace_20_code"] = test.nace_21_code.map(one_to_one_dict)

print(f'Number of nace 20 codes fille after 1-to-1: {train.nace_20_code.notna().sum()}')


# Rules based
train_rules = classify_rules(train, keyword_rules=norwegian_rules, company_nace21="nace_21_code", 
        company_text="company_name_description")
test_rules = classify_rules(test, keyword_rules=norwegian_rules, company_nace21="nace_21_code", 
        company_text="company_name_description")

train["nace_20_code"] = train["nace_20_code"].fillna(train_rules)
test["nace_20_code"] = test["nace_20_code"].fillna(test_rules)

print(f'Number of nace 20 codes filled after rules: {train.nace_20_code.notna().sum()}')


