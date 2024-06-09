import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
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

# ACO with active inference
def ant_colony_optimization_active_inference(graph, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, pheromone_deposit=1.0, free_energy_threshold=0.1):
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

    def free_energy(belief_in_cycle):
        if belief_in_cycle > 0 and belief_in_cycle < 1:
            uncertainty = -belief_in_cycle * np.log(belief_in_cycle) - (1 - belief_in_cycle) * np.log(1 - belief_in_cycle)
        else:
            uncertainty = 0
        expected_energy = -np.log(belief_in_cycle) if belief_in_cycle > 0 else float('inf')
        return expected_energy + uncertainty

    for _ in range(num_iterations):
        all_paths = []
        all_lengths = []
        for _ in range(num_ants):
            path = [random.randint(0, N - 1)]
            available_nodes = list(set(range(N)) - set(path))
            belief_in_cycle = 0.5
            while available_nodes:
                next_node = choose_next_node(available_nodes, path[-1])
                path.append(next_node)
                available_nodes.remove(next_node)
                if free_energy(belief_in_cycle) < free_energy_threshold:
                    break
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



# Generate multiple random symmetric graphs
def generate_random_graphs(num_graphs, num_nodes):
    graphs = []
    for _ in range(num_graphs):
        graph = np.random.rand(num_nodes, num_nodes)
        graph = (graph + graph.T) / 2  # Make the graph symmetric
        np.fill_diagonal(graph, 0)  # No self-loops
        graphs.append(graph)
    return graphs

# Function to compare ACO and ACO with active inference
def compare_methods(graphs):
    results = []
    for graph in graphs:
        # Run standard ACO
        best_path_basic, best_path_length_basic = ant_colony_optimization(graph)

        # Run ACO with active inference
        best_path_ai, best_path_length_ai = ant_colony_optimization_active_inference(graph)

        results.append({
            'basic_path_length': best_path_length_basic,
            'ai_path_length': best_path_length_ai
        })
    return results

# Specify number of graphs and nodes per graph
num_graphs = 10
num_nodes = 100

# Generate the graphs
graphs = generate_random_graphs(num_graphs, num_nodes)

# Compare the methods
results = compare_methods(graphs)
print(results)




# After obtaining results
results_df = pd.DataFrame(results)
print(results_df)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(results_df['basic_path_length'], label='Basic ACO Path Length', marker='o')
plt.plot(results_df['ai_path_length'], label='ACO with Active Inference Path Length', marker='x')
plt.title('Comparison of ACO and ACO with Active Inference')
plt.xlabel('Graph Index')
plt.ylabel('Path Length')
plt.legend()
plt.grid(True)
plt.show()
