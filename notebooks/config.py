from datetime import datetime
import os
import getpass

dato = str(datetime.today().strftime('%Y-%m-%d'))
seed = int(dato.replace("-", ""))

bucket_info = {
    bucket = "projet-aiml4os-wp10/Cluster5"
    bucket_url = f"https://minio.lab.sspcloud.fr/{bucket}"
    }

path_info = {
    'conversion_path': f"{bucket_info['bucket_url']}/NACE2.1-NACE2_Table_V1.05.xlsx",
    'train_path': f"{bucket_info['bucket_url']}/train_doublenace_2026-03-23.parquet",
    'test_path': f"{bucket_info['bucket_url']}/test_doublenace_2026-03-23.parquet",
    }

criteria = {
    'employee_threshold': 9,
    'turnover_threshold': 5000,
}

model_hyperparameter = {'C': 1, 
                        'max_features' :500, # Small for demonstration purposes
                        }

model_date = "2025-09-25"
