import math
import pandas as pd

def generate_combinations(projects, budget):
    """
    Generates all possible combinations of projects that do not exceed the budget.
    
    Args:
        projects: List of projects, each containing [cost, profit, expert_score]
        budget: Available budget
        
    Returns:
        list: List of tuples (combination, cost, profit, expert_score)
    """
    n = len(projects)
    result = []
    
    def backtrack(index, current_combo, current_cost, current_profit, current_expert):
        if index == n:
            # Add current combination to results
            result.append((current_combo.copy(), current_cost, current_profit, current_expert))
            return
        
        # Skip current project
        backtrack(index + 1, current_combo + [0], current_cost, current_profit, current_expert)
        
        # Include current project if possible
        if current_cost + projects[index][0] <= budget:
            backtrack(
                index + 1,
                current_combo + [1],
                current_cost + projects[index][0],
                current_profit + projects[index][1],
                current_expert + projects[index][2]
            )
    
    backtrack(0, [], 0, 0, 0)
    return result

def calculate_distances(combinations, norm_profits, norm_expert, ideal_profit, ideal_expert):
    """
    Calculates the distances from each combination to the ideal point.
    
    Args:
        combinations: List of project combinations
        norm_profits: Normalized profit values
        norm_expert: Normalized expert scores
        ideal_profit: Ideal profit value
        ideal_expert: Ideal expert score
        
    Returns:
        list: List of tuples with distance information
    """
    distances = []
    
    for combo, total_cost, total_profit, total_expert in combinations:
        # Calculate normalized sums for the combination
        norm_total_profit = sum([norm_profits[i] for i, x in enumerate(combo) if x == 1])
        norm_total_expert = sum([norm_expert[i] for i, x in enumerate(combo) if x == 1])
        
        # Calculate Euclidean distance to ideal point
        distance = math.sqrt((norm_total_profit - ideal_profit)**2 + (norm_total_expert - ideal_expert)**2)
        distances.append((combo, total_cost, total_profit, total_expert, norm_total_profit, norm_total_expert, distance))
    
    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[6])
    return distances

def create_combinations_df(distances):
    """
    Creates a pandas DataFrame with combination details for display
    
    Args:
        distances: List of tuples with distance information
        
    Returns:
        pandas.DataFrame: DataFrame with combination information
    """
    rows = []
    
    for i, (combo, cost, profit, expert, norm_profit, norm_expert, distance) in enumerate(distances, start=1):
        combo_str = ', '.join([f'x{j+1}' for j, x in enumerate(combo) if x == 1]) or "None"
        
        rows.append({
            'Rank': i,
            'Combination': combo_str,
            'Cost': cost,
            'Profit': profit,
            'Expert Score': expert,
            'Norm. Profit': round(norm_profit, 4),
            'Norm. Expert Score': round(norm_expert, 4),
            'Distance': round(distance, 4)
        })
    
    return pd.DataFrame(rows)