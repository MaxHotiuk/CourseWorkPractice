import numpy as np
import pandas as pd

def normalize_data(projects):
    """
    Нормалізує дані проєктів, поділивши значення кожного критерію на квадратний корінь
    суми квадратів цього критерію.
    
    Аргументи:
        projects: Список проєктів, кожен містить [вартість, прибуток, експертна_оцінка]
        
    Повертає:
        tuple: (нормалізовані_прибутки, нормалізовані_експертні_оцінки, дані_нормалізації)
    """
    # Витягуємо прибутки та експертні оцінки для нормалізації
    profits = [project[1] for project in projects]
    expert_scores = [project[2] for project in projects]
    
    # Обчислюємо квадрати значень
    squared_profits = [profit**2 for profit in profits]
    squared_expert = [score**2 for score in expert_scores]
    
    # Обчислюємо суму квадратів для кожного критерію
    sum_squared_profits = sum(squared_profits)
    sum_squared_expert = sum(squared_expert)
    
    # Обчислюємо нормалізуючий фактор (квадратний корінь суми квадратів)
    norm_factor_profits = (sum_squared_profits) ** 0.5
    norm_factor_expert = (sum_squared_expert) ** 0.5
    
    # Нормалізуємо кожне значення
    norm_profits = [profit / norm_factor_profits for profit in profits]
    norm_expert = [score / norm_factor_expert for score in expert_scores]
    
    # Створюємо дані нормалізації для відображення
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
    Створює pandas DataFrame з деталями нормалізації для відображення
    
    Аргументи:
        projects: Список проєктів
        norm_data: Словник з даними нормалізації
        
    Повертає:
        pandas.DataFrame: DataFrame з інформацією про нормалізацію
    """
    df = pd.DataFrame({
        'Проєкт': [f"x{i+1}" for i in range(len(projects))],
        'Прибуток': norm_data['profits'],
        'Норм. прибуток': [round(x, 4) for x in norm_data['norm_profits']],
        'Експертна оцінка': norm_data['expert_scores'],
        'Норм. експертна оцінка': [round(x, 4) for x in norm_data['norm_expert']]
    })
    
    # Додаємо рядок з підсумками
    totals = pd.DataFrame({
        'Проєкт': ['√Σ'],
        'Прибуток': [''],
        'Норм. прибуток': [round(norm_data['norm_factor_profits'], 4)],
        'Експертна оцінка': [''],
        'Норм. експертна оцінка': [round(norm_data['norm_factor_expert'], 4)]
    })
    
    return pd.concat([df, totals], ignore_index=True)

def verify_normalization(norm_profits, norm_expert):
    """
    Перевіряє правильність нормалізації, перевіряючи, чи дорівнює сума
    квадратів нормалізованих значень 1.
    
    Аргументи:
        norm_profits: Список нормалізованих значень прибутку
        norm_expert: Список нормалізованих експертних оцінок
        
    Повертає:
        dict: Результати перевірки
    """
    sum_squared_norm_profits = sum([x**2 for x in norm_profits])
    sum_squared_norm_expert = sum([x**2 for x in norm_expert])
    
    return {
        'profit_sum': round(sum_squared_norm_profits, 4),
        'profit_valid': abs(sum_squared_norm_profits - 1.0) < 0.0001,
        'expert_sum': round(sum_squared_norm_expert, 4),
        'expert_valid': abs(sum_squared_norm_expert - 1.0) < 0.0001
    }