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
    st.set_page_config(page_title="Ideal Point Method for Project Selection", 
                       page_icon="📊", 
                       layout="wide")
    
    st.title("Ideal Point Method for Project Selection")
    st.markdown(""""
    This application helps decision-makers select an optimal portfolio of projects 
    considering multiple criteria and budget constraints. It uses the Ideal Point Method 
    to find solutions that balance different objectives.
    """)
    
    with st.sidebar:
        st.header("Input Parameters")
        
        # Budget input
        budget = st.number_input("Available Budget", min_value=1, value=6)
        
        # Number of projects
        num_projects = st.number_input("Number of Projects", min_value=1, max_value=200, value=4)
        
        st.subheader("Analysis Options")
        show_normalization = st.checkbox("Show Normalization Details", value=True)
        show_knapsack = st.checkbox("Show Knapsack Solution", value=True)
        show_combinations = st.checkbox("Show All Combinations", value=True)
        num_top_combinations = st.slider("Number of Top Combinations to Display", 
                                         min_value=1, max_value=20, value=10)
    
    # Project data input
    st.header("Project Data")
    
    # Allow selecting input method
    input_method = st.radio("Input Method", 
                            ["Manual Entry", "Sample Data", "CSV Upload"],
                            horizontal=True)
    
    if input_method == "Manual Entry":
        col_headers = st.columns([1, 1, 1, 1])
        with col_headers[0]:
            st.markdown("**Project**")
        with col_headers[1]:
            st.markdown("**Cost**")
        with col_headers[2]:
            st.markdown("**Profit**")
        with col_headers[3]:
            st.markdown("**Expert Score**")
        
        # Initialize project data list
        projects = []
        
        # Create input fields for each project
        for i in range(num_projects):
            cols = st.columns([1, 1, 1, 1])
            with cols[0]:
                st.markdown(f"Project {i+1}")
            with cols[1]:
                cost = st.number_input(f"Cost {i+1}", 
                                       min_value=1, 
                                       value=20, 
                                       key=f"cost_{i}")
            with cols[2]:
                profit = st.number_input(f"Profit {i+1}", 
                                         min_value=0, 
                                         value=30, 
                                         key=f"profit_{i}")
            with cols[3]:
                expert = st.number_input(f"Expert Score {i+1}", 
                                         min_value=0, 
                                         value=40, 
                                         key=f"expert_{i}")
            
            # Add project data
            projects.append([cost, profit, expert])
                
    elif input_method == "Sample Data":
        # Provide sample project data
        sample_data = {
            "Small Example (4 projects)": [
                [2, 20, 4],  # x1
                [1, 30, 3],  # x2
                [3, 40, 2],  # x3
                [2, 20, 5]   # x4
            ]
        }
        
        selected_sample = st.selectbox("Select Sample Data", list(sample_data.keys()))
        projects = sample_data[selected_sample]
        
        # Display the sample data
        project_df = pd.DataFrame(projects, 
                                 columns=["Cost", "Profit", "Expert Score"],
                                 index=[f"Project {i+1}" for i in range(len(projects))])
        st.dataframe(project_df)
        
    else:  # CSV Upload
        st.info("Upload a CSV file with columns: Cost, Profit, ExpertScore")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
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
                    st.error("CSV must contain Cost, Profit, and ExpertScore columns")
                    projects = []
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                projects = []
        else:
            projects = []
    
    # Only proceed if we have project data
    if not projects:
        st.warning("Please enter project data to continue.")
        return
    
    # Run analysis when button is pressed
    if st.button("Run Analysis"):
        run_analysis(projects, budget, show_normalization, show_knapsack, 
                    show_combinations, num_top_combinations)

def run_analysis(projects, budget, show_normalization, show_knapsack, 
                show_combinations, num_top_combinations):
    """Run the complete project selection analysis"""
    
    st.header("Analysis Results")
    
    # Step 1: Normalize data
    norm_profits, norm_expert, norm_data = normalize_data(projects)
    
    if show_normalization:
        with st.expander("Step 1: Data Normalization", expanded=True):
            st.markdown("""
            In this step, we normalize the criteria values using the Euclidean normalization method. 
            This ensures that both criteria (profit and expert score) are on a comparable scale.
            
            The formula used is: $\\bar{a}_{ij} = \\frac{a_{ij}}{\\sqrt{\\sum_{i=1}^{m} a_{ij}^2}}$
            """)
            
            # Show normalization table
            norm_df = create_normalization_df(projects, norm_data)
            st.dataframe(norm_df, use_container_width=True)
            
            # Verify normalization
            verify_result = verify_normalization(norm_profits, norm_expert)
            
            st.markdown("**Verification**:")
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"Sum of squared normalized profits: {verify_result['profit_sum']}")
                st.markdown(f"Should equal 1.0: {'✅' if verify_result['profit_valid'] else '❌'}")
            with cols[1]:
                st.markdown(f"Sum of squared normalized expert scores: {verify_result['expert_sum']}")
                st.markdown(f"Should equal 1.0: {'✅' if verify_result['expert_valid'] else '❌'}")
    
    # Step 2: Solve knapsack problems to find ideal points
    with st.expander("Step 2: Finding Ideal Points", expanded=True):
        st.markdown("""
        For each criterion, we solve a knapsack problem to find the maximum possible value 
        given the budget constraint. These represent the ideal (but typically unattainable) points.
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
            st.markdown("**Maximizing Profit:**")
            selected = ", ".join([f"x{i+1}" for i, x in enumerate(profit_solution) if x == 1])
            st.markdown(f"Selected projects: {selected}")
            st.markdown(f"Maximum profit: {max_profit}")
            st.markdown(f"Normalized value: {ideal_profit:.4f}")
            
            if show_knapsack:
                # Changed from nested expander to a subheader and additional content
                st.markdown("#### Knapsack Solution for Profit")
                st.markdown("**Dynamic Programming Table:**")
                dp_df = create_dp_table_df(profit_dp, budget, "Profit")
                st.dataframe(dp_df)
        
        with cols[1]:
            st.markdown("**Maximizing Expert Score:**")
            selected = ", ".join([f"x{i+1}" for i, x in enumerate(expert_solution) if x == 1])
            st.markdown(f"Selected projects: {selected}")
            st.markdown(f"Maximum expert score: {max_expert}")
            st.markdown(f"Normalized value: {ideal_expert:.4f}")
            
            if show_knapsack:
                # Changed from nested expander to a subheader and additional content
                st.markdown("#### Knapsack Solution for Expert Score")
                st.markdown("**Dynamic Programming Table:**")
                dp_df = create_dp_table_df(expert_dp, budget, "Expert Score")
                st.dataframe(dp_df)
        
        st.markdown("**Ideal Point:**")
        st.markdown(f"(Normalized Profit, Normalized Expert Score) = ({ideal_profit:.4f}, {ideal_expert:.4f})")
    
    # Step 3: Generate all feasible combinations
    with st.expander("Step 3: Finding the Best Solution", expanded=True):
        st.markdown("""
        We generate all feasible combinations of projects that respect the budget constraints.
        For each combination, we calculate the Euclidean distance to the ideal point.
        The combination with the smallest distance is our recommended solution.
        """)
        
        combinations = generate_combinations(projects, budget)
        
        # Calculate distances to ideal point
        distances = calculate_distances(
            combinations, norm_profits, norm_expert, ideal_profit, ideal_expert)
        
        # Display best solution
        best_combo, best_cost, best_profit, best_expert, best_norm_profit, best_norm_expert, best_distance = distances[0]
        
        st.markdown("**Best Solution:**")
        selected = ", ".join([f"x{i+1}" for i, x in enumerate(best_combo) if x == 1])
        if not selected:
            selected = "None"
            
        st.markdown(f"Selected projects: {selected}")
        st.markdown(f"Total cost: {best_cost}")
        st.markdown(f"Total profit: {best_profit}")
        st.markdown(f"Total expert score: {best_expert}")
        st.markdown(f"Distance to ideal point: {best_distance:.4f}")
        
        # Show visualization of solutions
        st.markdown("**Visualization of Solutions:**")
        
        # Create dataframe for plotting
        plot_data = []
        for combo, cost, profit, expert, norm_profit, norm_expert, distance in distances:
            combo_str = ", ".join([f"x{j+1}" for j, x in enumerate(combo) if x == 1]) or "None"
            is_best = (combo == best_combo)
            is_ideal_profit = (norm_profit == ideal_profit)
            is_ideal_expert = (norm_expert == ideal_expert)
            
            point_type = "Regular"
            if is_best:
                point_type = "Best Solution"
            elif is_ideal_profit and is_ideal_expert:
                point_type = "Ideal Point"
            elif is_ideal_profit:
                point_type = "Ideal Profit"
            elif is_ideal_expert:
                point_type = "Ideal Expert"
            
            plot_data.append({
                "Combination": combo_str,
                "Normalized Profit": norm_profit,
                "Normalized Expert Score": norm_expert,
                "Profit": profit,
                "Expert Score": expert,
                "Distance": distance,
                "Type": point_type
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create scatter plot with Plotly
        fig = px.scatter(
            plot_df, 
            x="Normalized Profit", 
            y="Normalized Expert Score",
            color="Type",
            symbol="Type",
            hover_name="Combination",
            hover_data=["Profit", "Expert Score", "Distance"],
            title="Solutions in Normalized Criteria Space",
            color_discrete_map={
                "Best Solution": "#FF5733",
                "Ideal Profit": "#33A8FF",
                "Ideal Expert": "#33FF57",
                "Ideal Point": "#9E33FF",
                "Regular": "#BEBEBE"
            },
            symbol_map={
                "Best Solution": "star",
                "Ideal Profit": "diamond",
                "Ideal Expert": "diamond",
                "Ideal Point": "circle",
                "Regular": "circle"
            },
            size_max=15
        )
        
        # Add ideal point (if not already in the solutions)
        fig.add_scatter(
            x=[ideal_profit], 
            y=[ideal_expert],
            mode="markers",
            marker=dict(color="purple", size=15, symbol="x"),
            name="Ideal Point",
            hoverinfo="name"
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title="Normalized Profit",
            yaxis_title="Normalized Expert Score",
            legend_title="Solution Type",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show all combinations if requested
        if show_combinations:
            st.markdown(f"**Top {min(num_top_combinations, len(distances))} Solutions:**")
            combinations_df = create_combinations_df(distances[:num_top_combinations])
            st.dataframe(combinations_df, use_container_width=True)
            
            # Option to download full results
            csv = combinations_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="project_selection_results.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()