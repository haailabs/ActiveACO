import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

# Ensure the Figures/ folder exists
if not os.path.exists('Figures'):
    os.makedirs('Figures')

def plot_path_lengths(results_df):
    """
    Plot the path lengths for different methods using violin plots.
    
    Args:
        results_df (pd.DataFrame): A DataFrame containing the comparison results.
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=results_df[['NN_length', 'basic_aco_length', 'ai_aco_length']])
    plt.title('Path Lengths for Different Methods')
    plt.ylabel('Path Length')
    plt.xlabel('Methods')
    plt.legend(title='Methods', labels=['NN', 'Basic ACO', 'AI ACO'])
    
    # Add dashed lines for means and medians
    for method in ['NN_length', 'basic_aco_length', 'ai_aco_length']:
        mean_val = results_df[method].mean()
        median_val = results_df[method].median()
        plt.axhline(mean_val, color='r', linestyle='--', label=f'{method} Mean')
        plt.axhline(median_val, color='b', linestyle='-.', label=f'{method} Median')
    
    plt.savefig('Figures/path_lengths.png')
    plt.close()

def plot_computation_times(results_df):
    """
    Plot the computation times for different methods using violin plots.
    
    Args:
        results_df (pd.DataFrame): A DataFrame containing the comparison results.
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=results_df[['nn_time', 'basic_aco_time', 'ai_aco_time']])
    plt.title('Computation Times for Different Methods')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Methods')
    plt.legend(title='Methods', labels=['NN', 'Basic ACO', 'AI ACO'])
    
    # Add dashed lines for means and medians
    for method in ['nn_time', 'basic_aco_time', 'ai_aco_time']:
        mean_val = results_df[method].mean()
        median_val = results_df[method].median()
        plt.axhline(mean_val, color='r', linestyle='--', label=f'{method} Mean')
        plt.axhline(median_val, color='b', linestyle='-.', label=f'{method} Median')
    
    plt.savefig('Figures/computation_times.png')
    plt.close()

def plot_aco_improvement(results_df):
    """
    Plot the improvement of AI ACO over basic ACO.
    
    Args:
        results_df (pd.DataFrame): A DataFrame containing the comparison results.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x='aco_improvement', kde=True)
    plt.title('Improvement of AI ACO over Basic ACO')
    plt.xlabel('Improvement (%)')
    plt.ylabel('Frequency')
    
    # Add dashed lines for mean and median
    mean_val = results_df['aco_improvement'].mean()
    median_val = results_df['aco_improvement'].median()
    plt.axvline(mean_val, color='r', linestyle='--', label='Mean')
    plt.axvline(median_val, color='b', linestyle='-.', label='Median')
    
    plt.savefig('Figures/aco_improvement.png')
    plt.close()

def plot_path_length_vs_nodes(results_df):
    """
    Plot the path lengths vs the number of nodes using a scatter plot with error bars.
    
    Args:
        results_df (pd.DataFrame): A DataFrame containing the comparison results.
    """
    # Group by number of nodes and calculate mean and std for each method
    grouped = results_df.groupby('num_nodes').agg({
        'NN_length': ['mean', 'std'],
        'basic_aco_length': ['mean', 'std'],
        'ai_aco_length': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    grouped.columns = ['num_nodes', 'NN_mean', 'NN_std', 'Basic_ACO_mean', 'Basic_ACO_std', 'AI_ACO_mean', 'AI_ACO_std']

    plt.figure(figsize=(10, 6))

    # Plot scatter points with error bars for each method
    plt.errorbar(grouped['num_nodes'], grouped['NN_mean'], yerr=grouped['NN_std'], 
                 fmt='o-', capsize=5, label='NN', alpha=0.7)
    plt.errorbar(grouped['num_nodes'], grouped['Basic_ACO_mean'], yerr=grouped['Basic_ACO_std'], 
                 fmt='s-', capsize=5, label='Basic ACO', alpha=0.7)
    plt.errorbar(grouped['num_nodes'], grouped['AI_ACO_mean'], yerr=grouped['AI_ACO_std'], 
                 fmt='^-', capsize=5, label='AI ACO', alpha=0.7)

    plt.title('Path Lengths vs Number of Nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Path Length')
    plt.legend(title='Methods')
    
    # Add jitter to x-axis to avoid overlapping points
    for line in plt.gca().lines:
        line.set_xdata(line.get_xdata() + np.random.normal(0, 0.1, len(line.get_xdata())))
    
    # Add dashed lines for means
    plt.axhline(grouped['NN_mean'].mean(), color='r', linestyle='--', label='NN Mean')
    plt.axhline(grouped['Basic_ACO_mean'].mean(), color='g', linestyle='--', label='Basic ACO Mean')
    plt.axhline(grouped['AI_ACO_mean'].mean(), color='b', linestyle='--', label='AI ACO Mean')
    
    plt.savefig('Figures/path_length_vs_nodes.png')
    plt.close()