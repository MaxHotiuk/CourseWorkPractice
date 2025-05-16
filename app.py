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
from utils.sequential_concessions import (initialize_sequential_concessions, make_next_concession, 
                                        get_current_result, create_concessions_df, get_history_df)

def main():
    st.set_page_config(page_title="Вибір проєктів за кількома критеріями", 
                       page_icon="📊", 
                       layout="wide")
    
    st.title("Методи багатокритеріальної оптимізації для вибору проєктів")
    st.markdown("""
    Цей додаток допомагає особам, які приймають рішення, вибрати оптимальний портфель проєктів, 
    враховуючи кілька критеріїв та бюджетні обмеження.
    """)
    
    with st.sidebar:
        st.header("Вхідні параметри")
        
        # Select optimization method
        optimization_method = st.radio(
            "Метод оптимізації",
            ["Метод ідеальної точки", "Метод послідовних поступок"],
            index=0
        )
        
        # Budget input
        budget = st.number_input("Доступний бюджет", min_value=1, value=6)
        
        # Number of projects
        num_projects = st.number_input("Кількість проєктів", min_value=1, max_value=200, value=4)
        
        st.subheader("Опції аналізу")
        
        if optimization_method == "Метод ідеальної точки":
            show_normalization = st.checkbox("Показати деталі нормалізації", value=True)
            show_knapsack = st.checkbox("Показати рішення задачі про рюкзак", value=True)
            show_combinations = st.checkbox("Показати всі комбінації", value=True)
            num_top_combinations = st.slider("Кількість найкращих комбінацій для відображення", 
                                            min_value=1, max_value=20, value=10)
        else:  # Sequential concessions method
            st.markdown("**Параметри послідовних поступок:**")
            primary_criterion = st.radio(
                "Основний критерій",
                ["Прибуток", "Експертна оцінка"],
                index=0
            )
            primary_criterion_index = 1 if primary_criterion == "Прибуток" else 2
            secondary_criterion_index = 2 if primary_criterion == "Прибуток" else 1
            
            concession_amount = st.number_input(
                f"Величина поступки для {primary_criterion}", 
                min_value=1,
                value=10
            )
    
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
        
        # Add option to download entered project data
        if projects:
            project_df = pd.DataFrame(projects, 
                                    columns=["Вартість", "Прибуток", "Експертна оцінка"],
                                    index=[f"Проєкт {i+1}" for i in range(len(projects))])
            
            # Display the entered data as a table
            st.dataframe(project_df)
            
            # Download option for manual data
            csv = project_df.to_csv(index=True)
            st.download_button(
                label="Завантажити введені дані як CSV",
                data=csv,
                file_name="project_data.csv",
                mime="text/csv",
            )
                
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
                                 columns=["Cost", "Profit", "ExpertScore"],
                                 index=[f"Проєкт {i+1}" for i in range(len(projects))])
        st.dataframe(project_df)
        
        # Download option for sample data
        csv = project_df.to_csv(index=True)
        st.download_button(
            label="Завантажити дані прикладу як CSV",
            data=csv,
            file_name="sample_project_data.csv",
            mime="text/csv",
        )
        
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
    
    # Initialize session state for sequential concessions method
    if 'concessions_state' not in st.session_state:
        st.session_state.concessions_state = None
    if 'show_continue_button' not in st.session_state:
        st.session_state.show_continue_button = False
    if 'solution_accepted' not in st.session_state:
        st.session_state.solution_accepted = False
    
    # Run analysis based on selected method
    if optimization_method == "Метод ідеальної точки":
        if st.button("Виконати аналіз за методом ідеальної точки"):
            # Reset sequential concessions state when switching methods
            st.session_state.concessions_state = None
            st.session_state.show_continue_button = False
            st.session_state.solution_accepted = False
            
            run_ideal_point_analysis(
                projects, budget, show_normalization, show_knapsack, 
                show_combinations, num_top_combinations
            )
    else:  # Sequential concessions method
        # Initialize concessions process when button is pressed
        if st.button("Розпочати аналіз методом послідовних поступок"):
            # Initialize state
            st.session_state.concessions_state = initialize_sequential_concessions(
                projects, budget, primary_criterion_index, secondary_criterion_index
            )
            st.session_state.show_continue_button = True
            st.session_state.solution_accepted = False
            
            # Show initial solution
            display_sequential_concessions_results(
                st.session_state.concessions_state, 
                primary_criterion,
                concession_amount
            )
        
        # Continue with next concession
        if st.session_state.show_continue_button and not st.session_state.solution_accepted:
            with st.form("concession_form"):
                st.markdown("### Прийняти поточне рішення або зробити поступку?")
                
                make_concession = st.radio(
                    "Дія:",
                    ["Прийняти поточне рішення", "Зробити поступку і шукати нове рішення"],
                    index=1
                )
                
                if make_concession == "Зробити поступку і шукати нове рішення":
                    new_concession = st.number_input(
                        f"Величина поступки для {primary_criterion}", 
                        min_value=1,
                        value=concession_amount
                    )
                else:
                    new_concession = 0
                
                submit_button = st.form_submit_button("Продовжити")
                
                if submit_button:
                    if make_concession == "Прийняти поточне рішення":
                        st.success("Рішення прийнято!")
                        st.session_state.show_continue_button = False
                        st.session_state.solution_accepted = True
                        
                        # Display final solution after accepting
                        display_final_sequential_solution(
                            st.session_state.concessions_state,
                            primary_criterion
                        )
                    else:
                        # Apply next concession
                        st.session_state.concessions_state = make_next_concession(
                            st.session_state.concessions_state, 
                            new_concession
                        )
                        
                        # Check if we still have acceptable combinations
                        latest_history = st.session_state.concessions_state["history"][-1]
                        if "acceptable_combinations" in latest_history and not latest_history["acceptable_combinations"]:
                            st.warning("Немає прийнятних комбінацій з такою поступкою. Використовуємо попереднє рішення.")
                            st.session_state.show_continue_button = False
                            st.session_state.solution_accepted = True
                            
                            # Display final solution when no more combinations are available
                            display_final_sequential_solution(
                                st.session_state.concessions_state,
                                primary_criterion
                            )
                        else:
                            # Display updated results
                            display_sequential_concessions_results(
                                st.session_state.concessions_state, 
                                primary_criterion,
                                new_concession
                            )
        
        # If solution was already accepted, display the final solution
        elif st.session_state.solution_accepted and st.session_state.concessions_state:
            display_final_sequential_solution(
                st.session_state.concessions_state,
                primary_criterion
            )

def display_final_sequential_solution(state, primary_criterion):
    """Display the final solution after accepting in sequential concessions method"""
    
    st.header("Фінальне рішення методом послідовних поступок")
    
    # Determine names for criteria
    primary_name = primary_criterion
    secondary_name = "Експертна оцінка" if primary_criterion == "Прибуток" else "Прибуток"
    
    # Get current result
    current_results = get_current_result(state)
    
    # Show selected projects
    selected_projects = ", ".join([f"x{i+1}" for i, x in enumerate(current_results["final_solution"]) if x == 1]) or "Жодного"
    
    st.markdown("### Результати оптимізації")
    
    # Create a nice styled summary with columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Деталі рішення:**")
        st.markdown(f"**Вибрані проєкти:** {selected_projects}")
        st.markdown(f"**Загальна вартість:** {current_results['final_cost']}")
    
    with col2:
        st.markdown("**Значення критеріїв:**")
        st.markdown(f"**{primary_name}:** {current_results['final_primary_value']}")
        st.markdown(f"**{secondary_name}:** {current_results['final_secondary_value']}")
        st.markdown(f"**Загальна поступка для {primary_name}:** {current_results['total_concession']}")
    
    # Show history of iterations in an expander
    history_df = get_history_df(state)

    # Rename columns based on primary criterion
    column_mapping = {
        'Критерій 1': primary_name,
        'Критерій 2': secondary_name
    }
    history_df = history_df.rename(columns=column_mapping)

    with st.expander("Історія ітерацій", expanded=False):
        st.dataframe(history_df, use_container_width=True)

    csv = history_df.to_csv(index=False)
    st.download_button(
        label="Завантажити історію ітерацій як CSV",
        data=csv,
        file_name="sequential_concessions_history.csv",
        mime="text/csv",
    )

    # Visualize the final solution
    st.markdown("### Візуалізація фінального рішення")
    
    # Get all combinations from the last iteration
    latest_entry = None
    for entry in reversed(state["history"]):
        if "acceptable_combinations" in entry and entry["acceptable_combinations"]:
            latest_entry = entry
            break
    
    if latest_entry:
        final_solution = state["current_solution"]
        
        # Convert to plotting format
        plot_data = []
        for combo, cost, primary_value, secondary_value in latest_entry["acceptable_combinations"]:
            combo_str = ", ".join([f"x{j+1}" for j, x in enumerate(combo) if x == 1]) or "Жодного"
            point_type = "Фінальне рішення" if np.array_equal(combo, final_solution) else "Інші можливі рішення"
            
            plot_data.append({
                "Комбінація": combo_str,
                primary_name: primary_value,
                secondary_name: secondary_value,
                "Вартість": cost,
                "Тип": point_type
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create scatter plot with Plotly
        fig = px.scatter(
            plot_df, 
            x=primary_name, 
            y=secondary_name,
            color="Тип",
            symbol="Тип",
            hover_name="Комбінація",
            hover_data=["Вартість"],
            title=f"{primary_name} vs {secondary_name} для фінального рішення",
            color_discrete_map={
                "Фінальне рішення": "#FF5733",
                "Інші можливі рішення": "#BEBEBE"
            },
            symbol_map={
                "Фінальне рішення": "star",
                "Інші можливі рішення": "circle"
            },
            size_max=15
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title=primary_name,
            yaxis_title=secondary_name,
            legend_title="Тип рішення",
            height=500,
            width=800,
            autosize=False
        )
        
        st.plotly_chart(fig)

def display_sequential_concessions_results(state, primary_criterion, concession_amount):
    """Display the results of the sequential concessions method"""
    
    st.header("Результати методу послідовних поступок")
    
    # Determine names for criteria
    primary_name = primary_criterion
    secondary_name = "Експертна оцінка" if primary_criterion == "Прибуток" else "Прибуток"
    
    # Get current result
    current_results = get_current_result(state)
    
    # Show selected projects
    selected_projects = ", ".join([f"x{i+1}" for i, x in enumerate(current_results["final_solution"]) if x == 1]) or "Жодного"
    
    st.markdown("### Поточне рішення")
    st.markdown(f"**Вибрані проєкти:** {selected_projects}")
    st.markdown(f"**Загальна вартість:** {current_results['final_cost']}")
    st.markdown(f"**{primary_name}:** {current_results['final_primary_value']}")
    st.markdown(f"**{secondary_name}:** {current_results['final_secondary_value']}")
    st.markdown(f"**Загальна поступка для {primary_name}:** {current_results['total_concession']}")
    
    # Show history of iterations
    st.markdown("### Історія ітерацій")
    history_df = get_history_df(state)
    
    # Rename columns based on primary criterion
    column_mapping = {
        'Критерій 1': primary_name,
        'Критерій 2': secondary_name
    }
    history_df = history_df.rename(columns=column_mapping)
    
    st.dataframe(history_df, use_container_width=True)
    
    # Visualize latest iteration if available
    latest_entry = state["history"][-1]
    if "acceptable_combinations" in latest_entry and latest_entry["acceptable_combinations"]:
        st.markdown("### Прийнятні комбінації на поточній ітерації")
        
        final_solution = state["current_solution"]
        combinations_df = create_concessions_df(latest_entry["acceptable_combinations"], final_solution)
        
        # Rename columns based on primary criterion
        column_mapping = {
            'Критерій 1': primary_name,
            'Критерій 2': secondary_name
        }
        combinations_df = combinations_df.rename(columns=column_mapping)
        
        st.dataframe(combinations_df, use_container_width=True)
        
        # Create visualization
        st.markdown("### Візуалізація")
        
        # Convert to plotting format
        plot_data = []
        for combo, cost, primary_value, secondary_value in latest_entry["acceptable_combinations"]:
            combo_str = ", ".join([f"x{j+1}" for j, x in enumerate(combo) if x == 1]) or "Жодного"
            point_type = "Поточне рішення" if np.array_equal(combo, final_solution) else "Можливе рішення"
            
            plot_data.append({
                "Комбінація": combo_str,
                primary_name: primary_value,
                secondary_name: secondary_value,
                "Вартість": cost,
                "Тип": point_type
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create scatter plot with Plotly
        fig = px.scatter(
            plot_df, 
            x=primary_name, 
            y=secondary_name,
            color="Тип",
            symbol="Тип",
            hover_name="Комбінація",
            hover_data=["Вартість"],
            title=f"{primary_name} vs {secondary_name} для прийнятних комбінацій",
            color_discrete_map={
                "Поточне рішення": "#FF5733",
                "Можливе рішення": "#BEBEBE"
            },
            symbol_map={
                "Поточне рішення": "star",
                "Можливе рішення": "circle"
            },
            size_max=15
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title=primary_name,
            yaxis_title=secondary_name,
            legend_title="Тип рішення",
            height=500,
            width=800,
            autosize=False
        )
        
        st.plotly_chart(fig)
    
def run_ideal_point_analysis(projects, budget, show_normalization, show_knapsack, 
                            show_combinations, num_top_combinations):
    """Run the ideal point method analysis"""
    
    st.header("Результати аналізу за методом ідеальної точки")
    
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