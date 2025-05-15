import numpy as np
import pandas as pd

def normalize_data(projects):
    """
    Normalizes project data by dividing each criterion value by the square root of
    the sum of squares for that criterion.
    
    Args:
        projects: List of projects, each containing [cost, profit, expert_score]
        
    Returns:
        tuple: (normalized_profits, normalized_expert_scores, normalization_data)
    """
    # Extract profits and expert scores for normalization
    profits = [project[1] for project in projects]
    expert_scores = [project[2] for project in projects]
    
    # Calculate squares of values
    squared_profits = [profit**2 for profit in profits]
    squared_expert = [score**2 for score in expert_scores]
    
    # Calculate sum of squares for each criterion
    sum_squared_profits = sum(squared_profits)
    sum_squared_expert = sum(squared_expert)
    
    # Calculate normalizing factor (square root of sum of squares)
    norm_factor_profits = (sum_squared_profits) ** 0.5
    norm_factor_expert = (sum_squared_expert) ** 0.5
    
    # Normalize each value
    norm_profits = [profit / norm_factor_profits for profit in profits]
    norm_expert = [score / norm_factor_expert for score in expert_scores]
    
    # Create normalization data for display
    normalization_data = {
        'profits': profits,
        'squared_profits': squared_profits,
        'norm_profits': norm_profits,
        'norm_factor_profits': norm_factor_profits,
        'expert_scores': expert_scores,
        'squared_expert': squared_expert,
        'norm_expert': norm_expert,
        'norm_factor_expert': norm_factor_expert,
    }
    
    return norm_profits, norm_expert, normalization_data

def create_normalization_df(projects, norm_data):
    """
    Creates a pandas DataFrame with normalization details for display
    
    Args:
        projects: List of projects
        norm_data: Dictionary with normalization data
        
    Returns:
        pandas.DataFrame: DataFrame with normalization information
    """
    df = pd.DataFrame({
        'Project': [f"x{i+1}" for i in range(len(projects))],
        'Profit': norm_data['profits'],
        'Profit²': norm_data['squared_profits'],
        'Norm. Profit': [round(x, 4) for x in norm_data['norm_profits']],
        'Expert Score': norm_data['expert_scores'],
        'Expert Score²': norm_data['squared_expert'],
        'Norm. Expert Score': [round(x, 4) for x in norm_data['norm_expert']]
    })
    
    # Add totals row
    totals = pd.DataFrame({
        'Project': ['√Σ'],
        'Profit': [''],
        'Profit²': [''],
        'Norm. Profit': [round(norm_data['norm_factor_profits'], 4)],
        'Expert Score': [''],
        'Expert Score²': [''],
        'Norm. Expert Score': [round(norm_data['norm_factor_expert'], 4)]
    })
    
    return pd.concat([df, totals], ignore_index=True)

def verify_normalization(norm_profits, norm_expert):
    """
    Verifies that the normalization was done correctly by checking if the sum
    of squared normalized values equals 1.
    
    Args:
        norm_profits: List of normalized profit values
        norm_expert: List of normalized expert scores
        
    Returns:
        dict: Verification results
    """
    sum_squared_norm_profits = sum([x**2 for x in norm_profits])
    sum_squared_norm_expert = sum([x**2 for x in norm_expert])
    
    return {
        'profit_sum': round(sum_squared_norm_profits, 4),
        'profit_valid': abs(sum_squared_norm_profits - 1.0) < 0.0001,
        'expert_sum': round(sum_squared_norm_expert, 4),
        'expert_valid': abs(sum_squared_norm_expert - 1.0) < 0.0001
    }