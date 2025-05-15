import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import os

# Add the parent directory to the path to import utils modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.normalize import normalize_data, create_normalization_df, verify_normalization
from utils.knapsack import solve_knapsack, create_dp_table_df
from utils.combinations import generate_combinations, calculate_distances, create_combinations_df

def main():
    st.set_page_config(page_title="Метод ідеальної точки для вибору проєктів", 
                       page_icon="📊", 
                       layout="wide")
    
    st.title("Метод ідеальної точки для вибору проєктів")
    st.markdown("""
    Цей додаток допомагає особам, які приймають рішення, вибрати оптимальний портфель проєктів, 
    враховуючи кілька критеріїв та бюджетні обмеження. Він використовує Метод ідеальної точки 
    для знаходження рішень, що балансують різні цілі.
    """)
    
    with st.sidebar:
        st.header("Вхідні параметри")
        
        # Budget input
        budget = st.number_input("Доступний бюджет", min_value=1, value=6)
        
        # Number of projects
        num_projects = st.number_input("Кількість проєктів", min_value=1, max_value=200, value=4)
        
        st.subheader("Опції аналізу")
        show_normalization = st.checkbox("Показати деталі нормалізації", value=True)
        show_knapsack = st.checkbox("Показати рішення задачі про рюкзак", value=True)
        show_combinations = st.checkbox("Показати всі комбінації", value=True)
        num_top_combinations = st.slider("Кількість найкращих комбінацій для відображення", 
                                         min_value=1, max_value=20, value=10)
    
    # Project data input
    st.header("Дані про проєкти")
    
    # Allow selecting input method
    input_method = st.radio("Спосіб введення даних", 
                            ["Ручне введення", "Приклад даних", "Завантажити CSV"],
                            horizontal=True)
    
    if input_method == "Ручне введення":
        col_headers = st.columns([1, 1, 1, 1])
        with col_headers[0]:
            st.markdown("**Проєкт**")
        with col_headers[1]:
            st.markdown("**Вартість**")
        with col_headers[2]:
            st.markdown("**Прибуток**")
        with col_headers[3]:
            st.markdown("**Експертна оцінка**")
        
        # Initialize project data list
        projects = []
        
        # Create input fields for each project
        for i in range(num_projects):
            cols = st.columns([1, 1, 1, 1])
            with cols[0]:
                st.markdown(f"Проєкт {i+1}")
            with cols[1]:
                cost = st.number_input(f"Вартість {i+1}", 
                                       min_value=1, 
                                       value=20, 
                                       key=f"cost_{i}")
            with cols[2]:
                profit = st.number_input(f"Прибуток {i+1}", 
                                         min_value=0, 
                                         value=30, 
                                         key=f"profit_{i}")
            with cols[3]:
                expert = st.number_input(f"Експертна оцінка {i+1}", 
                                         min_value=0, 
                                         value=40, 
                                         key=f"expert_{i}")
            
            # Add project data
            projects.append([cost, profit, expert])
                
    elif input_method == "Приклад даних":
        # Provide sample project data
        sample_data = {
            "Малий приклад (4 проєкти)": [
                [2, 20, 4],  # x1
                [1, 30, 3],  # x2
                [3, 40, 2],  # x3
                [2, 20, 5]   # x4
            ]
        }
        
        selected_sample = st.selectbox("Виберіть приклад даних", list(sample_data.keys()))
        projects = sample_data[selected_sample]
        
        # Display the sample data
        project_df = pd.DataFrame(projects, 
                                 columns=["Вартість", "Прибуток", "Експертна оцінка"],
                                 index=[f"Проєкт {i+1}" for i in range(len(projects))])
        st.dataframe(project_df)
        
    else:  # CSV Upload
        st.info("Завантажте CSV файл із стовпцями: Cost, Profit, ExpertScore")
        
        uploaded_file = st.file_uploader("Виберіть CSV файл", type="csv")
        if uploaded_file is not None:
            try:
                csv_data = pd.read_csv(uploaded_file)
                required_columns = ["Cost", "Profit", "ExpertScore"]
                
                # Check if required columns exist (case-insensitive)
                has_columns = all(any(col.lower() == req.lower() for col in csv_data.columns) 
                                 for req in required_columns)
                
                if has_columns:
                    # Get the actual column names (preserving case)
                    cost_col = next(col for col in csv_data.columns 
                                   if col.lower() == "cost".lower())
                    profit_col = next(col for col in csv_data.columns 
                                     if col.lower() == "profit".lower())
                    expert_col = next(col for col in csv_data.columns 
                                     if col.lower() == "expertscore".lower())
                    
                    projects = []
                    for _, row in csv_data.iterrows():
                        projects.append([row[cost_col], row[profit_col], row[expert_col]])
                    
                    st.dataframe(csv_data)
                else:
                    st.error("CSV повинен містити стовпці Cost, Profit та ExpertScore")
                    projects = []
            except Exception as e:
                st.error(f"Помилка читання CSV: {e}")
                projects = []
        else:
            projects = []
    
    # Only proceed if we have project data
    if not projects:
        st.warning("Будь ласка, введіть дані про проєкти, щоб продовжити.")
        return
    
    # Run analysis when button is pressed
    if st.button("Виконати аналіз"):
        run_analysis(projects, budget, show_normalization, show_knapsack, 
                    show_combinations, num_top_combinations)

def run_analysis(projects, budget, show_normalization, show_knapsack, 
                show_combinations, num_top_combinations):
    """Run the complete project selection analysis"""
    
    st.header("Результати аналізу")
    
    # Step 1: Normalize data
    norm_profits, norm_expert, norm_data = normalize_data(projects)
    
    if show_normalization:
        with st.expander("Крок 1: Нормалізація даних", expanded=True):
            st.markdown("""
            На цьому кроці ми нормалізуємо значення критеріїв, використовуючи метод евклідової нормалізації. 
            Це забезпечує порівнянність обох критеріїв (прибутку та експертної оцінки).
            
            Використовується формула: $\\bar{a}_{ij} = \\frac{r_{ij}}{\\sqrt{\\sum_{i=1}^{m} r_{ij}^2}}$
            """)
            
            # Show normalization table
            norm_df = create_normalization_df(projects, norm_data)
            st.dataframe(norm_df, use_container_width=True)
            
            # Verify normalization
            verify_result = verify_normalization(norm_profits, norm_expert)
            
            st.markdown("**Перевірка:**")
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"Сума квадратів нормалізованих прибутків: {verify_result['profit_sum']}")
                st.markdown(f"Має дорівнювати 1.0: {'✅' if verify_result['profit_valid'] else '❌'}")
            with cols[1]:
                st.markdown(f"Сума квадратів нормалізованих експертних оцінок: {verify_result['expert_sum']}")
                st.markdown(f"Має дорівнювати 1.0: {'✅' if verify_result['expert_valid'] else '❌'}")
    
    # Step 2: Solve knapsack problems to find ideal points
    with st.expander("Крок 2: Пошук ідеальних точок", expanded=True):
        st.markdown("""
        Для кожного критерію ми вирішуємо задачу про рюкзак, щоб знайти максимально можливе значення 
        з урахуванням бюджетних обмежень. Ці значення представляють ідеальні (але зазвичай недосяжні) точки.
        """)
        
        # Solve for max profit
        profit_solution, max_profit, profit_dp, profit_path = solve_knapsack(
            projects, budget, 1)
        
        # Solve for max expert score
        expert_solution, max_expert, expert_dp, expert_path = solve_knapsack(
            projects, budget, 2)
        
        # Find normalized ideal points
        ideal_profit = sum([norm_profits[i] for i, x in enumerate(profit_solution) if x == 1])
        ideal_expert = sum([norm_expert[i] for i, x in enumerate(expert_solution) if x == 1])
        
        # Display ideal points
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Максимізація прибутку:**")
            selected = ", ".join([f"x{i+1}" for i, x in enumerate(profit_solution) if x == 1])
            st.markdown(f"Вибрані проєкти: {selected}")
            st.markdown(f"Максимальний прибуток: {max_profit}")
            st.markdown(f"Нормалізоване значення: {ideal_profit:.4f}")
            
            if show_knapsack:
                st.markdown("#### Рішення задачі про рюкзак для прибутку")
                st.markdown("**Таблиця динамічного програмування:**")
                dp_df = create_dp_table_df(profit_dp, budget, "Прибуток")
                st.dataframe(dp_df, hide_index=True)
        
        with cols[1]:
            st.markdown("**Максимізація експертної оцінки:**")
            selected = ", ".join([f"x{i+1}" for i, x in enumerate(expert_solution) if x == 1])
            st.markdown(f"Вибрані проєкти: {selected}")
            st.markdown(f"Максимальна експертна оцінка: {max_expert}")
            st.markdown(f"Нормалізоване значення: {ideal_expert:.4f}")
            
            if show_knapsack:
                st.markdown("#### Рішення задачі про рюкзак для експертної оцінки")
                st.markdown("**Таблиця динамічного програмування:**")
                dp_df = create_dp_table_df(expert_dp, budget, "Експертна оцінка")
                st.dataframe(dp_df, hide_index=True)
        
        st.markdown("**Ідеальна точка:**")
        st.markdown(f"(Прибуток, Експертна оцінка) = ({max_profit}, {max_expert})")
        st.markdown(f"(Нормалізований прибуток, Нормалізована експертна оцінка) = ({ideal_profit:.4f}, {ideal_expert:.4f})")
    
    # Step 3: Generate all feasible combinations
    with st.expander("Крок 3: Пошук найкращого рішення", expanded=True):
        st.markdown("""
        Ми генеруємо всі можливі комбінації проєктів, які відповідають бюджетним обмеженням.
        Для кожної комбінації обчислюємо евклідову відстань до ідеальної точки.
        Комбінація з найменшою відстанню є нашим рекомендованим рішенням.
        """)
        st.markdown("""
        Формула, яка використовується для вимірювання відстані до ідеальної точки: $D_i=\\sqrt{\\sum_{j=1}^{m}{(r_{ij}\\ -\\ r_j^+)}^2}$
        
        Де:
        - $D_i$ - відстань від рішення $i$ до ідеальної точки
        - $r_{ij}$ - нормалізоване значення рішення $i$ за критерієм $j$
        - $r_j^+$ - ідеальне значення для критерію $j$
        """)
        
        combinations = generate_combinations(projects, budget)
        
        # Calculate distances to ideal point
        distances = calculate_distances(
            combinations, norm_profits, norm_expert, ideal_profit, ideal_expert)
        
        # Display best solution
        best_combo, best_cost, best_profit, best_expert, best_norm_profit, best_norm_expert, best_distance = distances[0]
        
        st.markdown("**Найкраще рішення:**")
        selected = ", ".join([f"x{i+1}" for i, x in enumerate(best_combo) if x == 1])
        if not selected:
            selected = "Жодного"
            
        st.markdown(f"Вибрані проєкти: {selected}")
        st.markdown(f"Загальна вартість: {best_cost}")
        st.markdown(f"Загальний прибуток: {best_profit}")
        st.markdown(f"Загальна експертна оцінка: {best_expert}")
        st.markdown(f"Відстань до ідеальної точки: {best_distance:.4f}")
        
        # Show visualization of solutions
        st.markdown("**Візуалізація рішень:**")
        
        # Create dataframe for plotting
        plot_data = []
        for combo, cost, profit, expert, norm_profit, norm_expert, distance in distances:
            combo_str = ", ".join([f"x{j+1}" for j, x in enumerate(combo) if x == 1]) or "Жодного"
            is_best = (combo == best_combo)
            is_ideal_profit = (norm_profit == ideal_profit)
            is_ideal_expert = (norm_expert == ideal_expert)
            
            point_type = "Звичайна точка"
            if is_best:
                point_type = "Найкраще рішення"
            elif is_ideal_profit and is_ideal_expert:
                point_type = "Ідеальна точка"
            elif is_ideal_profit:
                point_type = "Ідеальний прибуток"
            elif is_ideal_expert:
                point_type = "Ідеальна експертна оцінка"
            
            plot_data.append({
                "Комбінація": combo_str,
                "Нормалізований прибуток": norm_profit,
                "Нормалізована експертна оцінка": norm_expert,
                "Прибуток": profit,
                "Експертна оцінка": expert,
                "Відстань": distance,
                "Тип": point_type
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create scatter plot with Plotly
        fig = px.scatter(
            plot_df, 
            x="Нормалізований прибуток", 
            y="Нормалізована експертна оцінка",
            color="Тип",
            symbol="Тип",
            hover_name="Комбінація",
            hover_data=["Прибуток", "Експертна оцінка", "Відстань"],
            title="Рішення в просторі нормалізованих критеріїв",
            color_discrete_map={
                "Найкраще рішення": "#FF5733",
                "Ідеальний прибуток": "#33A8FF",
                "Ідеальна експертна оцінка": "#33FF57",
                "Ідеальна точка": "#9E33FF",
                "Звичайна точка": "#BEBEBE"
            },
            symbol_map={
                "Найкраще рішення": "star",
                "Ідеальний прибуток": "diamond",
                "Ідеальна експертна оцінка": "diamond",
                "Ідеальна точка": "circle",
                "Звичайна точка": "circle"
            },
            size_max=15
        )
        
        # Add ideal point (if not already in the solutions)
        fig.add_scatter(
            x=[ideal_profit], 
            y=[ideal_expert],
            mode="markers",
            marker=dict(color="purple", size=15, symbol="x"),
            name="Ідеальна точка",
            hoverinfo="name"
        )
        
        # Customize layout to make the plot square
        fig.update_layout(
            xaxis_title="Нормалізований прибуток",
            yaxis_title="Нормалізована експертна оцінка",
            legend_title="Тип рішення",
            height=600,
            width=600,
            autosize=False,
            yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            )
        )
        
        st.plotly_chart(fig, use_container_width=False)
        
        # Show all combinations if requested
        if show_combinations:
            st.markdown(f"**Топ {min(num_top_combinations, len(distances))} рішень:**")
            combinations_df = create_combinations_df(distances[:num_top_combinations])
            st.dataframe(combinations_df, use_container_width=True)
            
            # Option to download full results
            csv = combinations_df.to_csv(index=False)
            st.download_button(
                label="Завантажити результати як CSV",
                data=csv,
                file_name="project_selection_results.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()