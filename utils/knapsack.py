import pandas as pd
import numpy as np

def solve_knapsack(projects, budget, criterion_index):
    """
    Solves the knapsack problem using dynamic programming.
    
    Args:
        projects: List of projects, each containing [cost, profit, expert_score]
        budget: Available budget
        criterion_index: Index of criterion to maximize (1 for profit, 2 for expert score)
        
    Returns:
        tuple: (selected_projects, max_value, dp_table, dp_solution_path)
    """
    n = len(projects)
    
    # Create DP table
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    
    # Fill the table
    for i in range(1, n + 1):
        cost = projects[i-1][0]
        value = projects[i-1][criterion_index]
        
        for w in range(budget + 1):
            if cost <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-cost] + value)
            else:
                dp[i][w] = dp[i-1][w]
    
    # Reconstruct the solution
    solution = [0] * n
    w = budget
    
    # Path of cells in the solution (for visualization)
    solution_path = []
    
    for i in range(n, 0, -1):
        cost = projects[i-1][0]
        value = projects[i-1][criterion_index]
        
        if w >= cost and dp[i][w] == dp[i-1][w-cost] + value:
            solution[i-1] = 1
            solution_path.append((i, w))
            w -= cost
        else:
            solution_path.append((i, w))
    
    # Start at 0,0
    if len(solution_path) > 0 and solution_path[-1][0] > 1:
        solution_path.append((0, 0))
    
    # Reverse to get path from start to end
    solution_path.reverse()
    
    return solution, dp[n][budget], dp, solution_path

def create_dp_table_df(dp, budget, criterion_name):
    """
    Creates a pandas DataFrame from the DP table for display
    
    Args:
        dp: Dynamic programming table
        budget: Maximum budget
        criterion_name: Name of criterion being maximized
        
    Returns:
        pandas.DataFrame: DataFrame version of DP table
    """
    rows = []
    
    for i in range(len(dp)):
        row = {'i\\S': i}
        for w in range(budget + 1):
            row[str(w)] = dp[i][w]
        rows.append(row)
    
    return pd.DataFrame(rows)