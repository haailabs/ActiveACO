import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr


def ant_colony_optimization(graph, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, pheromone_deposit=1.0):
    N = graph.shape[0]
    pheromone = np.ones((N, N))
    best_path = None
    best_path_length = float('inf')

    def calculate_path_length(path):
        return sum(graph[path[i], path[i + 1]] for i in range(len(path) - 1)) + graph[path[-1], path[0]]

    def choose_next_node(available_nodes, current_node):
        probabilities = [(pheromone[current_node, node] ** alpha) * ((1.0 / graph[current_node, node]) ** beta) for node in available_nodes]
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        return np.random.choice(available_nodes, p=probabilities)

    for _ in range(num_iterations):
        all_paths = []
        all_lengths = []
        for _ in range(num_ants):
            path = [random.randint(0, N - 1)]
            available_nodes = list(set(range(N)) - set(path))
            while available_nodes:
                next_node = choose_next_node(available_nodes, path[-1])
                path.append(next_node)
                available_nodes.remove(next_node)
            path_length = calculate_path_length(path)
            all_paths.append(path)
            all_lengths.append(path_length)
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length
        pheromone *= (1 - evaporation_rate)
        for path, length in zip(all_paths, all_lengths):
            for i in range(len(path) - 1):
                pheromone[path[i], path[i + 1]] += pheromone_deposit / length
            pheromone[path[-1], path[0]] += pheromone_deposit / length

    return best_path, best_path_length

def improved_aco_active_inference(graph, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, pheromone_deposit=1.0):
    N = graph.shape[0]
    pheromone = np.ones((N, N))
    best_path = None
    best_path_length = float('inf')

    def calculate_path_length(path):
        return sum(graph[path[i], path[i + 1]] for i in range(len(path) - 1)) + graph[path[-1], path[0]]

    def choose_next_node(available_nodes, current_node, belief):
        probabilities = [(pheromone[current_node, node] ** alpha) * ((1.0 / graph[current_node, node]) ** beta) for node in available_nodes]
        probabilities = np.array(probabilities)
        probabilities *= belief  # Adjust probabilities based on belief
        probabilities /= probabilities.sum()
        return np.random.choice(available_nodes, p=probabilities)

    def free_energy(belief_in_tour, path_length):
        if 0 < belief_in_tour < 1:
            uncertainty = -belief_in_tour * np.log(belief_in_tour) - (1 - belief_in_tour) * np.log(1 - belief_in_tour)
        else:
            uncertainty = 0
        expected_energy = path_length  # Use actual path length as energy
        return expected_energy + uncertainty

    for iteration in range(num_iterations):
        all_paths = []
        all_lengths = []
        for _ in range(num_ants):
            path = [random.randint(0, N - 1)]
            available_nodes = list(set(range(N)) - set(path))
            belief_in_tour = 0.5
            current_path_length = 0
            while available_nodes:
                next_node = choose_next_node(available_nodes, path[-1], belief_in_tour)
                path.append(next_node)
                available_nodes.remove(next_node)
                current_path_length += graph[path[-2], path[-1]]
                
                if best_path_length != float('inf'):
                    belief_in_tour = 1 - (current_path_length / best_path_length)
                    belief_in_tour = max(0.1, min(0.9, belief_in_tour))
            
            # Ensure the path is a complete tour by returning to the start
            path.append(path[0])
            path_length = calculate_path_length(path)
            all_paths.append(path)
            all_lengths.append(path_length)
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

        pheromone *= (1 - evaporation_rate)
        for path, length in zip(all_paths, all_lengths):
            deposit = pheromone_deposit / length
            for i in range(len(path) - 1):
                pheromone[path[i], path[i + 1]] += deposit

        # Elitist strategy
        best_deposit = pheromone_deposit * 2 / best_path_length
        for i in range(len(best_path) - 1):
            pheromone[best_path[i], best_path[i + 1]] += best_deposit

    return best_path, best_path_length
def generate_random_graphs(num_graphs, num_nodes):
    graphs = []
    for _ in range(num_graphs):
        # Choose a random distribution type
        dist_type = np.random.choice(['uniform', 'normal', 'exponential', 'lognormal'])
        
        if dist_type == 'uniform':
            low = np.random.uniform(0.1, 1)
            high = np.random.uniform(low + 0.5, low + 5)
            graph = np.random.uniform(low, high, size=(num_nodes, num_nodes))
        elif dist_type == 'normal':
            mean = np.random.uniform(1, 5)
            std = np.random.uniform(0.1, 2)
            graph = np.abs(np.random.normal(mean, std, size=(num_nodes, num_nodes)))
        elif dist_type == 'exponential':
            scale = np.random.uniform(0.5, 2)
            graph = np.random.exponential(scale, size=(num_nodes, num_nodes))
        else:  # lognormal
            mean = np.random.uniform(0, 2)
            sigma = np.random.uniform(0.1, 1)
            graph = np.random.lognormal(mean, sigma, size=(num_nodes, num_nodes))
        
        # Ensure symmetry
        graph = (graph + graph.T) / 2
        
        # Set diagonal to zero (no self-loops)
        np.fill_diagonal(graph, 0)
        
        # Randomly set some edges to a small non-zero value (to vary density)
        mask = np.random.random(graph.shape) < np.random.uniform(0.3, 1)
        graph = np.where(mask, graph, 0.01)  # Use 0.01 instead of 0
        
        # Scale the graph to have weights mostly between 0.1 and 10
        scale_factor = 9.9 / np.percentile(graph[graph > 0], 95)
        graph = graph * scale_factor + 0.1
        
        graphs.append(graph)
    
    return graphs
    graphs = []
    for _ in range(num_graphs):
        # Choose a random distribution type
        dist_type = np.random.choice(['uniform', 'normal', 'exponential', 'lognormal'])
        
        if dist_type == 'uniform':
            low = np.random.uniform(0, 1)
            high = np.random.uniform(low + 0.5, low + 5)
            graph = np.random.uniform(low, high, size=(num_nodes, num_nodes))
        elif dist_type == 'normal':
            mean = np.random.uniform(0, 5)
            std = np.random.uniform(0.1, 2)
            graph = np.abs(np.random.normal(mean, std, size=(num_nodes, num_nodes)))
        elif dist_type == 'exponential':
            scale = np.random.uniform(0.1, 2)
            graph = np.random.exponential(scale, size=(num_nodes, num_nodes))
        else:  # lognormal
            mean = np.random.uniform(0, 2)
            sigma = np.random.uniform(0.1, 1)
            graph = np.random.lognormal(mean, sigma, size=(num_nodes, num_nodes))
        
        # Ensure symmetry
        graph = (graph + graph.T) / 2
        
        # Set diagonal to zero (no self-loops)
        np.fill_diagonal(graph, 0)
        
        # Randomly set some edges to zero (to vary density)
        mask = np.random.random(graph.shape) < np.random.uniform(0.3, 1)
        graph *= mask
        
        # Scale the graph to have weights mostly between 0 and 10
        scale_factor = 10 / np.percentile(graph[graph > 0], 95)
        graph *= scale_factor
        
        graphs.append(graph)
    
    return graphs

def compare_methods(graphs):
    results = []
    for i, graph in enumerate(graphs):
        print(f"Processing graph {i+1}/{len(graphs)}")
        
        best_path_basic, best_path_length_basic = ant_colony_optimization(graph)
        best_path_ai, best_path_length_ai = improved_aco_active_inference(graph)

        results.append({
            'graph_id': i,
            'basic_path_length': best_path_length_basic,
            'ai_path_length': best_path_length_ai,
            'improvement': (best_path_length_basic - best_path_length_ai) / best_path_length_basic * 100
        })
    return results

def analyze_graph_characteristics(graphs, results):
    characteristics = []
    
    for i, graph in enumerate(graphs):
        # Calculate graph characteristics
        total_edges = np.sum(graph > 0)
        total_possible_edges = graph.shape[0] * (graph.shape[0] - 1)
        density = total_edges / total_possible_edges if total_possible_edges > 0 else 0
        
        avg_edge_weight = np.mean(graph[graph > 0]) if total_edges > 0 else 0
        std_edge_weight = np.std(graph[graph > 0]) if total_edges > 0 else 0
        
        # Calculate the coefficient of variation of edge weights
        cv_edge_weight = std_edge_weight / avg_edge_weight if avg_edge_weight > 0 else 0
        
        # Calculate the range of edge weights
        edge_weight_range = np.max(graph) - np.min(graph[graph > 0]) if total_edges > 0 else 0
        
        characteristics.append({
            'graph_id': i,
            'avg_edge_weight': avg_edge_weight,
            'std_edge_weight': std_edge_weight,
            'cv_edge_weight': cv_edge_weight,
            'density': density,
            'edge_weight_range': edge_weight_range,
            'improvement': results[i]['improvement']
        })
    
    df = pd.DataFrame(characteristics)
    
    # Print debugging information
    print("\nDebugging Information:")
    print(f"Number of graphs: {len(graphs)}")
    print(f"Density statistics: min={df['density'].min()}, max={df['density'].max()}, mean={df['density'].mean()}, std={df['density'].std()}")
    print(f"Improvement statistics: min={df['improvement'].min()}, max={df['improvement'].max()}, mean={df['improvement'].mean()}, std={df['improvement'].std()}")
    
    # Calculate correlations
    correlations = {}
    for column in df.columns:
        if column not in ['graph_id', 'improvement']:
            correlation, p_value = spearmanr(df[column], df['improvement'])
            correlations[column] = {'correlation': correlation, 'p_value': p_value}
    
    # Print correlations
    print("\nCorrelations with improvement:")
    for char, values in correlations.items():
        print(f"{char}: correlation = {values['correlation']:.4f}, p-value = {values['p_value']:.4f}")
    
    # Plotting
    num_chars = len(correlations)
    fig, axes = plt.subplots((num_chars + 1) // 2, 2, figsize=(15, 5 * ((num_chars + 1) // 2)))
    axes = axes.ravel()
    
    for i, (column, values) in enumerate(correlations.items()):
        axes[i].scatter(df[column], df['improvement'])
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Improvement (%)')
        axes[i].set_title(f'{column} vs Improvement\nr={values["correlation"]:.2f}, p={values["p_value"]:.4f}')
    
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    
    return df, correlations
def perform_statistical_tests(results_df):
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(results_df['basic_path_length'], results_df['ai_path_length'])
    print(f"Paired T-test: T-statistic = {t_stat}, P-value = {p_value}")

    # Wilcoxon signed-rank test
    w_stat, w_p_value = stats.wilcoxon(results_df['basic_path_length'], results_df['ai_path_length'])
    print(f"Wilcoxon Signed-Rank Test: W-statistic = {w_stat}, P-value = {w_p_value}")

    # Mann-Whitney U test (if treating as independent samples)
    u_stat, u_p_value = stats.mannwhitneyu(results_df['basic_path_length'], results_df['ai_path_length'])
    print(f"Mann-Whitney U Test: U-statistic = {u_stat}, P-value = {u_p_value}")

if __name__ == "__main__":
    num_graphs = 50
    num_nodes = 100

    print("Generating random graphs...")
    graphs = generate_random_graphs(num_graphs, num_nodes)

    print("Comparing methods...")
    results = compare_methods(graphs)

    results_df = pd.DataFrame(results)
    print("\nResults summary:")
    print(results_df.describe())

    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['basic_path_length'], results_df['ai_path_length'])
    plt.plot([results_df['basic_path_length'].min(), results_df['basic_path_length'].max()], 
             [results_df['basic_path_length'].min(), results_df['basic_path_length'].max()], 
             'r--', label='y=x')
    plt.xlabel('Basic ACO Path Length')
    plt.ylabel('Improved ACO Path Length')
    plt.title('Comparison of Basic ACO vs Improved ACO with Active Inference')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(results_df['basic_path_length'], label='Basic ACO Path Length', marker='o')
    plt.plot(results_df['ai_path_length'], label='Improved ACO Path Length', marker='x')
    plt.title('Comparison of ACO and Improved ACO with Active Inference')
    plt.xlabel('Graph Index')
    plt.ylabel('Path Length')
    plt.legend()
    plt.grid(True)
    plt.show()

    avg_improvement = results_df['improvement'].mean()
    print(f"\nAverage improvement: {avg_improvement:.2f}%")
    print("\nPerforming statistical tests:")
    perform_statistical_tests(results_df)
    graph_analysis_df, correlations = analyze_graph_characteristics(graphs, results)
