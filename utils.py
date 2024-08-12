import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr
import time

def nearest_neighbor_tsp(graph):
    """
    Nearest Neighbor TSP algorithm.
    
    Args:
        graph (np.ndarray): Adjacency matrix representing the graph.
        
    Returns:
        tuple: A tuple containing the best tour and its distance.
    """
    n = len(graph)
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        last = tour[-1]
        next_node = min(unvisited, key=lambda x: graph[last][x])
        tour.append(next_node)
        unvisited.remove(next_node)
    tour.append(tour[0])
    
    def calculate_tour_distance(graph, tour):
        return sum(graph[tour[i]][tour[i+1]] for i in range(len(tour)-1))
    
    return tour, calculate_tour_distance(graph, tour)

def ant_colony_optimization(graph, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, pheromone_deposit=1.0):
    """
    Ant Colony Optimization algorithm for TSP.
    
    Args:
        graph (np.ndarray): Adjacency matrix representing the graph.
        num_ants (int): Number of ants in the colony.
        num_iterations (int): Number of iterations to run the algorithm.
        alpha (float): Pheromone importance factor.
        beta (float): Heuristic information importance factor.
        evaporation_rate (float): Pheromone evaporation rate.
        pheromone_deposit (float): Amount of pheromone deposited by each ant.
        
    Returns:
        tuple: A tuple containing the best path and its length.
    """
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
    """
    Improved Ant Colony Optimization algorithm with Active Inference for TSP.
    
    Args:
        graph (np.ndarray): Adjacency matrix representing the graph.
        num_ants (int): Number of ants in the colony.
        num_iterations (int): Number of iterations to run the algorithm.
        alpha (float): Pheromone importance factor.
        beta (float): Heuristic information importance factor.
        evaporation_rate (float): Pheromone evaporation rate.
        pheromone_deposit (float): Amount of pheromone deposited by each ant.
        
    Returns:
        tuple: A tuple containing the best path and its length.
    """
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

def particle_swarm_optimization(graph, num_particles=30, num_iterations=100, w=0.5, c1=1, c2=2):
    """
    Particle Swarm Optimization algorithm for TSP.
    
    Args:
        graph (np.ndarray): Adjacency matrix representing the graph.
        num_particles (int): Number of particles in the swarm.
        num_iterations (int): Number of iterations to run the algorithm.
        w (float): Inertia weight.
        c1 (float): Cognitive parameter.
        c2 (float): Social parameter.
        
    Returns:
        tuple: A tuple containing the best solution and its fitness value.
    """
    def create_solution():
        return np.random.permutation(len(graph))
    
    def calculate_fitness(solution):
        return sum(graph[solution[i], solution[(i+1) % len(solution)]] for i in range(len(solution)))
    
    particles = [create_solution() for _ in range(num_particles)]
    velocities = [np.zeros_like(particles[0]) for _ in range(num_particles)]
    personal_best = particles.copy()
    global_best = min(particles, key=calculate_fitness) 
    
    for _ in range(num_iterations):
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] + 
                            c1 * r1 * (personal_best[i] - particles[i]) + 
                            c2 * r2 * (global_best - particles[i]))
            
            new_position = particles[i] + velocities[i]
            new_position = np.argsort(new_position)
            
            if calculate_fitness(new_position) < calculate_fitness(particles[i]):
                particles[i] = new_position
                if calculate_fitness(new_position) < calculate_fitness(personal_best[i]):
                    personal_best[i] = new_position
                    if calculate_fitness(new_position) < calculate_fitness(global_best):
                        global_best = new_position
    
    return global_best, calculate_fitness(global_best)

def generate_random_graphs(num_graphs, num_nodes):
    """
    Generate random graphs with different weight distributions.
    
    Args:
        num_graphs (int): Number of graphs to generate.
        num_nodes (int): Number of nodes in each graph.
        
    Returns:
        list: A list of generated graphs.
    """
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

def compare_methods(graphs):
    """
    Compare different optimization methods on the given graphs.
    
    Args:
        graphs (list): A list of graphs to compare the methods on.
        
    Returns:
        list: A list of dictionaries containing the comparison results.
    """
    results = []
    for i, graph in enumerate(graphs):
        print(f"Processing graph {i+1}/{len(graphs)}")
        
        start_time = time.time()
        best_path_NN, best_path_length_NN = nearest_neighbor_tsp(graph)
        nn_time = time.time() - start_time
        
        start_time = time.time()
        best_path_basic, best_path_length_basic = ant_colony_optimization(graph)
        basic_aco_time = time.time() - start_time
        
        start_time = time.time()
        best_path_ai_aco, best_path_length_ai_aco = improved_aco_active_inference(graph)
        ai_aco_time = time.time() - start_time
        results.append({
            'graph_id': i,
            'num_nodes': graph.shape[0],
            'NN_length': best_path_length_NN,
            'basic_aco_length': best_path_length_basic,
            'ai_aco_length': best_path_length_ai_aco,
            'aco_improvement': (best_path_length_basic - best_path_length_ai_aco) / best_path_length_basic * 100,
            'nn_time': nn_time,
            'basic_aco_time': basic_aco_time,
            'ai_aco_time': ai_aco_time,
        })
    return results

def perform_statistical_tests(results_df):
    """
    Perform statistical tests on the comparison results.
    
    Args:
        results_df (pd.DataFrame): A DataFrame containing the comparison results.
    """
    print(f"Average NN path length: {results_df['NN_length'].mean()}")
    
    # ACO tests
    t_stat_aco, p_value_aco = stats.ttest_rel(results_df['basic_aco_length'], results_df['ai_aco_length'])
    print(f"ACO Paired T-test: T-statistic = {t_stat_aco}, P-value = {p_value_aco}")

    w_stat_aco, w_p_value_aco = stats.wilcoxon(results_df['basic_aco_length'], results_df['ai_aco_length'])
    print(f"ACO Wilcoxon Signed-Rank Test: W-statistic = {w_stat_aco}, P-value = {w_p_value_aco}")
  
    # Time comparisons
    t_stat_time_aco, p_value_time_aco = stats.ttest_rel(results_df['basic_aco_time'], results_df['ai_aco_time'])
    print(f"ACO Computation Time Paired T-test: T-statistic = {t_stat_time_aco}, P-value = {p_value_time_aco}")

def run_experiments(node_sizes, num_graphs):
    """
    Run experiments with different graph sizes and number of graphs.
    
    Args:
        node_sizes (list): A list of node sizes to generate graphs for.
        num_graphs (int): The number of graphs to generate for each node size.
        
    Returns:
        pd.DataFrame: A DataFrame containing the results for all experiments.
    """
    all_results = []
    for num_nodes in node_sizes:
        print(f"\nRunning experiments for {num_nodes} nodes")
        graphs = generate_random_graphs(num_graphs, num_nodes)
        results = compare_methods(graphs)
        all_results.extend(results)
    
    return pd.DataFrame(all_results)

