import pandas as pd
import random
import pandas as pd
import numpy as np


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

def get_label(labels, probs, possibles, all_labels, choose_type="random", seed=42):
    """
    Selects a label for each instance based on model probabilities and allowed label restrictions.

    This function applies label restrictions (from `possibles`) to the predicted probabilities
    and selects either the most probable label (`choose_type="best"`) or a random label weighted
    by probability (`choose_type="random"`). If no restricted labels are found in the model output,
    it falls back to a random choice from the original `possibles`.

    Args:
        labels (list of list of str): List of predicted label sets for each instance.
        probs (list of list of float): Corresponding list of probability scores for each label.
        possibles (list of list of str): Allowed labels for each instance.
        all_labels (list of str): All possible labels in the model.
        choose_type (str, optional): Strategy for choosing the label. Either "best" or "random".
            Defaults to "random".
        seed (int, optional): Random seed for reproducibility when using random selection.
            Defaults to 42.

    Returns:
        tuple:
            - list of str: Selected labels for each instance.
            - int: Count of instances where fallback random selection was used due to no valid restrictions.

    Raises:
        TypeError: If `choose_type` is not "best" or "random".

    Notes:
        - Ensures all labels in `all_labels` are present in the probability dictionary for consistency.
    """
    preds_restricted = []
    rand_count = 0
    random.seed(seed)
    

    for i in range(len(labels)):
        label_prob_dict = dict(zip([l for l in labels[i]], probs[i]))

        # Clean labels and check for missing
        for label in all_labels:
            if label not in label_prob_dict:
                label_prob_dict[label] = 0.0
    
        # Filter possibles to only those that exist in model labels
        valid_possibles = [label for label in possibles[i] if label in all_labels]

        # Restrict to allowed labels
        restricted_probs = {label: label_prob_dict[label] for label in valid_possibles}
        
        # Pick the label with the highest probability or random
        if choose_type == "best":
            best_label = max(restricted_probs, key=restricted_probs.get)
        elif choose_type == "random":
            if not restricted_probs:
                best_label = np.nan # In case not restricted labels are found in the model send to manual
                rand_count += 1
            else:
                new_label = get_random(restricted_probs, seed = seed)
                best_label = new_label[0]
                rand_count += new_label[1]
        else:
            raise TypeError("choose_type should be 'best' or 'random'")
        preds_restricted.append(best_label)

    return preds_restricted, rand_count


def get_random(probs_dict, seed):
    """
    Selects a label randomly from a probability dictionary, weighted by the given probabilities.

    This function normalizes the input probabilities and uses them to randomly select one label.
    If all probabilities are zero, it falls back to uniform random selection.

    Args:
        probs_dict (dict): A dictionary where keys are labels and values are their associated probabilities.
        seed (int): Random seed for reproducibility.

    Returns:
        str: The selected label.

    Notes:
        - Uses `random.choices()` for weighted selection.
        - If the sum of probabilities is zero, falls back to `random.choice()` to avoid division by zero.
    """
    
    labels = list(probs_dict.keys())
    probs = list(probs_dict.values())
    total = sum(probs)
    rand_count = 0

    if total == 0:
        selected_label = np.nan # if no valid text return na for sending to manual control
        rand_count = 1
    else:
        normalized_probs = [p / total for p in probs]
        selected_label = random.choices(labels, weights=normalized_probs, k=1)[0]
    return selected_label, rand_count


if __name__ == "__main__":
    dt = pd.DataFrame({'id': [1, 2, 3], 'nace21': ['14.21', '15.20', '16.11']})

    poss = get_possibles(dt, "nace21", conversion_path="https://minio.lab.sspcloud.fr/projet-aiml4os-wp10/Cluster5/NACE2.1-NACE2_Table_V1.05.xlsx")
    print(poss)