# Program for automatic backclassification using keywords

import pandas as pd

# Keys er NACE revision 2.1
keyword_rules = {
    '14.24': {
        '14.20': ['pels'],
        '14.11': []
    },
    '24.54': {
        '24.53': ['sink', 'zink'],
        '24.54': [],
    },
}


def _classify_company_rules(company, keyword_rules, company_nace21, company_text):
    nace = company[company_nace21]
    text = company.get(company_text, "").lower() # returns "" if missing

    # Check keyword-based rules
    if nace in keyword_rules:
        keyword_map = keyword_rules[nace]
        for target_code, keywords in keyword_map.items():
            if any(kw in text for kw in keywords):
                return target_code
        # No keyword matched, return the default
        for target_code, keywords in keyword_map.items():
            if not keywords:  # Default target
                return target_code

    # Step 3: If no rules apply
    return None


def classify_rules(dt, keyword_rules, company_nace21, company_text):

    nace_rules_new = dt.apply(
        lambda row: _classify_company_rules(row, keyword_rules, company_nace21, company_text),
        axis=1)

    return nace_rules_new


if __name__ == "__main__":
    # Example of use
    df = pd.DataFrame({'sn2025':["14.24", "14.24", "24.54", "24.54", "25.21"],
                        'company_name_description': ["Fancy clothes company, pels og skinnprodukter", 
                                "Klær for alle, økologisk klær produksent",
                                "Steel AS, Lage produkter fra stål", "Metal 4 alle, produsere produkter av zink", 
                                "Bygg varmt, Radiator installasjon i privat bolig."]
                  })

    NACE2_new = classify_rules(df, keyword_rules=keyword_rules, company_nace21="sn2025", company_text="company_name_description")
    print(NACE2_new)
