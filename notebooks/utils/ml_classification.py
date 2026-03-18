import pandas as pd


def get_possibles(dt, company_nace21, conversion_path):
    """
    Maps NACE rev.2.1 codes in a DataFrame to possible corresponding NACE 2.0 codes using a conversion file.

    This function reads a excel file containing with mappings, groups the reev. 2.0 codes
    by rev. 2.1, and maps the codes in the input DataFrame to their corresponding rev. 2.0 lists.

    Args:
        dt (pandas.DataFrame): The input DataFrame containing a 'sn2025_1' column.
        conversion_file (str): Path of the excel conversion file with rev. 2.1 to 2.0 mappings.

    Returns:
        pandas.Series: A Series where each entry is a list of possible rev. 2.0 codes
        corresponding to the rev. 2.1 code in the input DataFrame.
    """
    convert = pd.read_excel(conversion_path, sheet_name="Sheet0")
    convert4 = convert.loc[(convert.LEVEL == 4) & (convert.TypeOfCorrespondence.notna()), :].copy()

    mulige_dict = convert4.groupby('NACE21_Code')['NACE2_Code'].apply(list).to_dict()
    possibles = (dt[company_nace21]
        .map(mulige_dict)
        .apply(lambda x: x if isinstance(x, list) else [])
        .reset_index(drop=True)
    )
    return possibles


if __name__ == "__main__":
    dt = pd.DataFrame({'id': [1, 2, 3], 'nace21': ['14.21', '15.20', '16.11']})

    poss = get_possibles(dt, "nace21", conversion_path="https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/Cluster5/NACE2.1-NACE2_Table_V1.05.xlsx")
    print(poss)