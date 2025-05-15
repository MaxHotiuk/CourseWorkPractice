import pandas as pd
import numpy as np

def solve_knapsack(projects, budget, criterion_index):
    """
    Вирішує задачу про рюкзак за допомогою динамічного програмування.
    
    Аргументи:
        projects: Список проєктів, кожен містить [вартість, прибуток, експертна_оцінка]
        budget: Доступний бюджет
        criterion_index: Індекс критерію для максимізації (1 для прибутку, 2 для експертної оцінки)
        
    Повертає:
        tuple: (обрані_проєкти, максимальне_значення, таблиця_дп, шлях_рішення_дп)
    """
    n = len(projects)
    
    # Створюємо таблицю ДП
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    
    # Заповнюємо таблицю
    for i in range(1, n + 1):
        cost = projects[i-1][0]
        value = projects[i-1][criterion_index]
        
        for w in range(budget + 1):
            if cost <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-cost] + value)
            else:
                dp[i][w] = dp[i-1][w]
    
    # Відновлюємо рішення
    solution = [0] * n
    w = budget
    
    # Шлях комірок у рішенні (для візуалізації)
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
    
    # Починаємо з 0,0
    if len(solution_path) > 0 and solution_path[-1][0] > 1:
        solution_path.append((0, 0))
    
    # Обертаємо для отримання шляху від початку до кінця
    solution_path.reverse()
    
    return solution, dp[n][budget], dp, solution_path

def create_dp_table_df(dp, budget, criterion_name):
    """
    Створює pandas DataFrame з таблиці ДП для відображення
    
    Аргументи:
        dp: Таблиця динамічного програмування
        budget: Максимальний бюджет
        criterion_name: Назва критерію, який максимізується
        
    Повертає:
        pandas.DataFrame: Версія таблиці ДП у форматі DataFrame
    """
    rows = []
    
    for i in range(len(dp)):
        row = {'i\\S': i}
        for w in range(budget + 1):
            row[str(w)] = dp[i][w]
        rows.append(row)
    
    return pd.DataFrame(rows)