import pandas as pd
import numpy as np


def add_manual_flag(df, criteria, employee_var, turnover_var= None):
    """
    This function evaluates thresholds for number of employees and revenue to determine if
    a unit should be flagged for manual review.

    Args:
        df (pandas.DataFrame): The input DataFrame containing unit data and a 'new_flag' column.
        criteria (dict): A dictionary with threshold values. Expected keys:
            - 'employee_threshold'
            - 'turnover_threshold'
        employee_var: A string for the name of the variable to use in dt for the number of 
            employees.
        turnover_var (optional): A string for the name of the variable to used in dt for the 
            latest turnover. 

    Returns:
        pandas.Series: A Series of integers (0 or 1), where 1 indicates the unit should be flagged
        for manual control.
    """
    # Threshold check for employee count
    employee_rule = df[employee_var] > criteria['employee_threshold']

    # Threshold check for turnover
    if not turnover_var:
        turnover_rule = df[turnover_var] > criteria['turnover_threshold']
    else:
        turnover_rule = False
    
    manual_control = employee_rule | turnover_rule
                                                        
    return manual_control.astype(int)