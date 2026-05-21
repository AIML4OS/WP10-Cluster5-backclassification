# Program for one-to-one mapping dictionary for backclassification of NACE rev. 2.1 to 2.0

import pandas as pd
import openpyxl # used indirectly


def _get_norwegian_dict():
    norway_one_to_one = {
        "00.00": "00.00",
        "01.41": "01.41",
        "01.63": "01.63",
        "10.62": "10.62",
        "11.07": "11.07",
        "25.21": "25.21",
        "26.70": "26.70",
        "26.60": "32.50",
        "55.90": "55.90",
        "58.19": "58.19",
        "77.22": "77.29",
        "78.20": "78.20",
    }
    return norway_one_to_one


def get_problem_NACE(path_excel: str):
    """
    Get a list of NACE revision 2.1 codes that are problem groups for manual classisifcation
    
    Parameters
    ----------
    path_excel : str
        A path to an excel file where sheet name 'sheet0' contains infomration on conversion of NACE 
        between revisioni 2.1 and 2.0. Variables should include 'LEVEL', 'NACE21_Code' and 'TypeOfCorrespondence'

    Returns
    -------
    list
       A list of NACE rev. 2.1 codes that are missing correspondence to rev. 2.0
    
    """
    # Read in excel correspondence table
    dt = pd.read_excel(path_excel, sheet_name="Sheet0")

    # Filter for level 4
    dt_level4 = dt.loc[dt.LEVEL == 4, :].copy()

    # Get problem group with no correspondence
    mask_problem = dt_level4.TypeOfCorrespondence.isna()
    dt_problem = dt_level4.loc[mask_problem, :]

    # Format and convert to list
    problem_ls = dt_problem.NACE21_Code.tolist()

    return problem_ls


def get_1to1_dict(path_excel: str, national_dict: dict =None):
    """
    Build a 1-to-1 correspondence dictionary between NACE21 and NACE2 codes 
    based on a standard EU correspondence table.

    A national dictionary of 1-to-1 mappings may optionally be provided. If
    ``national_dict`` is ``None``, a default Norwegian dictionary example is 
    used.

    Parameters
    ----------
    national_dict : dict or None, optional
        A dictionary containing additional or overriding 1-to-1 mappings.
        If ``None`` (default), a predefined Norwegian mapping is loaded.

    Returns
    -------
    dict
        A dictionary where keys are ``NACE21_Code`` values and values are the
        corresponding ``NACE2_Code`` values.
    """
    # Read in excel correspondence table
    dt = pd.read_excel(path_excel, sheet_name="Sheet0")

    # Filter for level 4
    dt_level4 = dt.loc[dt.LEVEL == 4, :].copy()

    # Filter for 1:1 or many:1 and not missing
    mask_problem = dt_level4.TypeOfCorrespondence.isna()
    mask_1 = dt_level4.NumberOfCorresponding_NACE2 == 1
    dt_xto1 = dt_level4.loc[mask_1 & ~mask_problem, :].copy()

    # Create dict
    one_to_one = dt_xto1.set_index('NACE21_Code')['NACE2_Code'].to_dict()

    # Add in national dictionary of 1-to-1
    if national_dict is None:
        national_dict = _get_norwegian_dict()
    if isinstance(national_dict, dict):
        one_to_one_dict = one_to_one | national_dict
    else:
        raise TypeError(f"Expected national_dict to be a dict, but got a {type(national_dict).__name__}.")

    return one_to_one_dict


if __name__ == "__main__":
    excel_path = "https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/Cluster5/NACE2.1-NACE2_Table_V1.05.xlsx"
    one_to_one_dict = get_1to1_dict(excel_path)
    print(one_to_one_dict)