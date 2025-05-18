import pandas as pd
import numpy as np
from .knapsack import solve_knapsack

def initialize_sequential_concessions(projects, budget, primary_criterion_index=1, secondary_criterion_index=2):
    """
    Ініціалізує процес послідовних поступок для двох критеріїв.
    
    Аргументи:
        projects: Список проєктів, кожен містить [вартість, критерій1, критерій2]
        budget: Доступний бюджет
        primary_criterion_index: Індекс основного критерію (1 або 2)
        secondary_criterion_index: Індекс другорядного критерію (1 або 2)
    
    Повертає:
        dict: Початковий стан процесу послідовних поступок
    """
    # Крок 1: Оптимізація за основним критерієм
    primary_solution, primary_max, _, _ = solve_knapsack(projects, budget, primary_criterion_index)
    primary_cost = sum(projects[i][0] for i, x in enumerate(primary_solution) if x == 1)
    secondary_value = sum(projects[i][secondary_criterion_index] for i, x in enumerate(primary_solution) if x == 1)
    
    # Генеруємо всі можливі комбінації для подальшого використання
    all_combinations = generate_all_combinations(projects, budget)
    
    return {
        "projects": projects,
        "budget": budget,
        "primary_criterion_index": primary_criterion_index,
        "secondary_criterion_index": secondary_criterion_index,
        "current_solution": primary_solution,
        "current_primary_value": primary_max,
        "current_secondary_value": secondary_value,
        "current_cost": primary_cost,
        "original_primary_max": primary_max,
        "all_combinations": all_combinations,
        "iteration": 0,
        "history": [{
            "solution": primary_solution,
            "primary_value": primary_max,
            "secondary_value": secondary_value,
            "cost": primary_cost,
            "concession_amount": 0,
            "message": "Початкове рішення за основним критерієм."
        }]
    }

def make_next_concession(state, concession_amount):
    projects = state["projects"]
    budget = state["budget"]
    primary_criterion_index = state["primary_criterion_index"]
    secondary_criterion_index = state["secondary_criterion_index"]
    current_primary_value = state["current_primary_value"]
    all_combinations = state["all_combinations"]
    
    # Визначаємо мінімально прийнятне значення основного критерію після поступки
    min_acceptable_primary = current_primary_value - concession_amount
    
    # Фільтруємо комбінації
    acceptable_combinations = []
    for combo, cost in all_combinations:
        combo_primary = sum(projects[i][primary_criterion_index] for i, x in enumerate(combo) if x == 1)
        combo_secondary = sum(projects[i][secondary_criterion_index] for i, x in enumerate(combo) if x == 1)
        if combo_primary >= min_acceptable_primary:
            acceptable_combinations.append((combo, cost, combo_primary, combo_secondary))
    
    # Перевіряємо, чи є прийнятні комбінації
    if not acceptable_combinations:
        message = f"Немає комбінацій з основним критерієм >= {min_acceptable_primary}."
        state["history"].append({
            "solution": state["current_solution"],
            "primary_value": state["current_primary_value"],
            "secondary_value": state["current_secondary_value"],
            "cost": state["current_cost"],
            "concession_amount": concession_amount,
            "message": message,
            "acceptable_combinations": []
        })
        return state
    
    # Вибираємо найкращу за другорядним критерієм
    final_solution = max(acceptable_combinations, key=lambda x: x[3])
    combo, combo_cost, combo_primary, combo_secondary = final_solution
    
    # Оновлюємо стан
    state["current_solution"] = combo
    state["current_primary_value"] = combo_primary
    state["current_secondary_value"] = combo_secondary
    state["current_cost"] = combo_cost
    state["iteration"] += 1
    state["history"].append({
        "solution": combo,
        "primary_value": combo_primary,
        "secondary_value": combo_secondary,
        "cost": combo_cost,
        "concession_amount": concession_amount,
        "message": f"Поступка {concession_amount}: основний = {combo_primary}, другорядний = {combo_secondary}.",
        "acceptable_combinations": acceptable_combinations
    })
    
    return state

def get_current_result(state):
    return {
        "final_solution": state["current_solution"],
        "final_primary_value": state["current_primary_value"],
        "final_secondary_value": state["current_secondary_value"],
        "final_cost": state["current_cost"],
        "iterations": state["iteration"],
        "concession_amount": state["history"][-1]["concession_amount"],
        "total_concession": state["original_primary_max"] - state["current_primary_value"],
        "history": state["history"]
    }

def create_concessions_df(acceptable_combinations, final_solution):
    """
    Створює DataFrame з результатами.
    
    Аргументи:
        acceptable_combinations: Список прийнятних комбінацій
        final_solution: Фінальне вибране рішення
    
    Повертає:
        pandas.DataFrame: Таблиця результатів
    """
    rows = []
    for i, (combo, cost, primary_value, secondary_value) in enumerate(acceptable_combinations, 1):
        combo_str = ', '.join([f'x{j+1}' for j, x in enumerate(combo) if x == 1]) or "Жодного"
        is_final = np.array_equal(combo, final_solution)
        rows.append({
            'Ранг': i,
            'Комбінація': combo_str,
            'Вартість': cost,
            'Критерій 1': primary_value, 
            'Критерій 2': secondary_value,
            'Фінальне': '✓' if is_final else ''
        })
    return pd.DataFrame(rows)

def generate_all_combinations(projects, budget):
    """
    Генерує всі можливі комбінації проєктів у межах бюджету.
    
    Аргументи:
        projects: Список проєктів [вартість, критерій1, критерій2]
        budget: Доступний бюджет
    
    Повертає:
        list: Список кортежів (комбінація, вартість)
    """
    n = len(projects)
    result = []
    
    def backtrack(index, current_combo, current_cost):
        if index == n:
            result.append((current_combo.copy(), current_cost))
            return
        backtrack(index + 1, current_combo + [0], current_cost)
        if current_cost + projects[index][0] <= budget:
            backtrack(index + 1, current_combo + [1], current_cost + projects[index][0])
    
    backtrack(0, [], 0)
    return result

def get_history_df(state):
    """
    Створює DataFrame з історією ітерацій.
    
    Аргументи:
        state: Поточний стан процесу послідовних поступок
    
    Повертає:
        pandas.DataFrame: Таблиця історії ітерацій
    """
    rows = []
    for i, entry in enumerate(state["history"]):
        projects_str = ', '.join([f'x{j+1}' for j, x in enumerate(entry["solution"]) if x == 1]) or "Жодного"
        rows.append({
            'Ітерація': i,
            'Поступка': entry["concession_amount"],
            'Вибрані проєкти': projects_str,
            'Критерій 1': entry["primary_value"],
            'Критерій 2': entry["secondary_value"],
            'Вартість': entry["cost"],
            'Повідомлення': entry["message"]
        })
    return pd.DataFrame(rows)