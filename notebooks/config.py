from datetime import datetime
import os
import getpass


dato = str(datetime.today().strftime('%Y-%m-%d'))
seed = int(dato.replace("-", ""))

path_info = {
    'conversion_path': "https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/Cluster5/NACE2.1-NACE2_Table_V1.05.xlsx",
    'train_path': "https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/train_norwaydata_2026-01-13.parquet",
    'test_path': "https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/NorwayData/test_norwaydata_2026-01-13.parquet"
    }

criteria = {
    'employee_threshold': 9,
    'turnover_threshold': 5000,
}

model_date = "2025-09-25"
