from utils import (
    nearest_neighbor_tsp,
    ant_colony_optimization,
    improved_aco_active_inference,
    particle_swarm_optimization,
    generate_random_graphs,
    compare_methods,
    perform_statistical_tests,
    run_experiments
)

from visualization import (
    plot_path_lengths,
    plot_computation_times,
    plot_aco_improvement,
    plot_path_length_vs_nodes,
)

import pandas as pd

if __name__ == "__main__":
    node_sizes = [50] # Size of the grap
    num_graphs = 100 # Number of graphs per node size
    results_df = run_experiments(node_sizes, num_graphs)

    print(f"\nResults summary (for {num_graphs} graphs per node size):")
    summary = results_df.groupby('num_nodes').agg({
        'NN_length': ['mean', 'std'],
        'basic_aco_length': ['mean', 'std'],
        'ai_aco_length': ['mean', 'std'],
        'aco_improvement': ['mean', 'std'],
        'nn_time': ['mean', 'std'],
        'basic_aco_time': ['mean', 'std'],
        'ai_aco_time': ['mean', 'std'],
    })

    # Set display options to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(summary)

    # Reset display options to default (optional)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')

    print("\nPerforming statistical tests:")
    perform_statistical_tests(results_df)

    # Generate plots
    plot_path_lengths(results_df)
    plot_computation_times(results_df)
    plot_aco_improvement(results_df)
    plot_path_length_vs_nodes(results_df)
