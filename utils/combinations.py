import math
import pandas as pd

def generate_combinations(projects, budget):
    """
    Генерує всі можливі комбінації проєктів, які не перевищують бюджет.
    
    Аргументи:
        projects: Список проєктів, кожен містить [вартість, прибуток, експертна_оцінка]
        budget: Доступний бюджет
        
    Повертає:
        list: Список кортежів (комбінація, вартість, прибуток, експертна_оцінка)
    """
    n = len(projects)
    result = []
    
    def backtrack(index, current_combo, current_cost, current_profit, current_expert):
        if index == n:
            # Додаємо поточну комбінацію до результатів
            result.append((current_combo.copy(), current_cost, current_profit, current_expert))
            return
        
        # Пропускаємо поточний проєкт
        backtrack(index + 1, current_combo + [0], current_cost, current_profit, current_expert)
        
        # Включаємо поточний проєкт, якщо це можливо
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
    Обчислює відстані від кожної комбінації до ідеальної точки.
    
    Аргументи:
        combinations: Список комбінацій проєктів
        norm_profits: Нормалізовані значення прибутку
        norm_expert: Нормалізовані експертні оцінки
        ideal_profit: Ідеальне значення прибутку
        ideal_expert: Ідеальна експертна оцінка
        
    Повертає:
        list: Список кортежів з інформацією про відстані
    """
    distances = []
    
    for combo, total_cost, total_profit, total_expert in combinations:
        # Обчислюємо нормалізовані суми для комбінації
        norm_total_profit = sum([norm_profits[i] for i, x in enumerate(combo) if x == 1])
        norm_total_expert = sum([norm_expert[i] for i, x in enumerate(combo) if x == 1])
        
        # Обчислюємо евклідову відстань до ідеальної точки
        distance = math.sqrt((norm_total_profit - ideal_profit)**2 + (norm_total_expert - ideal_expert)**2)
        distances.append((combo, total_cost, total_profit, total_expert, norm_total_profit, norm_total_expert, distance))
    
    # Сортуємо за відстанню (за зростанням)
    distances.sort(key=lambda x: x[6])
    return distances

def create_combinations_df(distances):
    """
    Створює pandas DataFrame з деталями комбінацій для відображення
    
    Аргументи:
        distances: Список кортежів з інформацією про відстані
        
    Повертає:
        pandas.DataFrame: DataFrame з інформацією про комбінації
    """
    rows = []
    
    for i, (combo, cost, profit, expert, norm_profit, norm_expert, distance) in enumerate(distances, start=1):
        combo_str = ', '.join([f'x{j+1}' for j, x in enumerate(combo) if x == 1]) or "Жодного"
        
        rows.append({
            'Ранг': i,
            'Комбінація': combo_str,
            'Вартість': cost,
            'Прибуток': profit,
            'Експертна оцінка': expert,
            'Норм. прибуток': round(norm_profit, 4),
            'Норм. експертна оцінка': round(norm_expert, 4),
            'Відстань': round(distance, 4)
        })
    
    return pd.DataFrame(rows)