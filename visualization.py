import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway

def plot_path_lengths(results_df, figures_dir, num_nodes, num_graphs, markdown_file):
    """
    Plot the path lengths for different methods using violin plots and add statistical tests.
    
    Args:
        results_df (pd.DataFrame): A DataFrame containing the comparison results.
        figures_dir (str): Directory to save the figures.
        num_nodes (int): Number of nodes in the graph.
        num_graphs (int): Number of graphs.
        markdown_file (str): Path to the markdown file to save the results.
    """
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=results_df[['NN_length', 'basic_aco_length', 'ai_aco_length']])
    plt.title(f'Path Lengths Comparison (Nodes: {num_nodes}, Graphs: {num_graphs})', fontsize=16)
    plt.ylabel('Path Length', fontsize=14)
    plt.xlabel('Methods', fontsize=14)
    plt.xticks([0, 1, 2], ['Nearest Neighbor', 'Basic ACO', 'AI-enhanced ACO'], fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add dashed lines for means and medians
    colors = ['red', 'blue', 'green']
    linestyles = ['--', '-.']
    labels = ['Mean', 'Median']
    for i, method in enumerate(['NN_length', 'basic_aco_length', 'ai_aco_length']):
        mean_val = results_df[method].mean()
        median_val = results_df[method].median()
        plt.axhline(mean_val, color=colors[i], linestyle=linestyles[0], alpha=0.7)
        plt.axhline(median_val, color=colors[i], linestyle=linestyles[1], alpha=0.7)
    
    # Create custom legend
    custom_lines = [plt.Line2D([0], [0], color=c, lw=2, linestyle=ls) for c in colors for ls in linestyles]
    custom_labels = [f'{m} {l}' for m in ['NN', 'Basic ACO', 'AI ACO'] for l in labels]
    plt.legend(custom_lines, custom_labels, title='Statistics', title_fontsize=12, fontsize=10, loc='upper right')
    
    # Perform ANOVA test
    f_val, p_val = f_oneway(results_df['NN_length'], results_df['basic_aco_length'], results_df['ai_aco_length'])
    with open(markdown_file, 'a') as f:
        f.write(f'# Path Lengths Statistical Test\n\n')
        f.write(f'**ANOVA p-value:** {p_val:.3e}\n')
        f.write(f'**Effect Size (F-value):** {f_val:.3f}\n\n')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'path_lengths_nodes_{num_nodes}_graphs_{num_graphs}.png'), dpi=300)
    plt.close()

def plot_computation_times(results_df, figures_dir, num_nodes, num_graphs, markdown_file):
    """
    Plot the computation times for different methods using violin plots and add statistical tests.
    
    Args:
        results_df (pd.DataFrame): A DataFrame containing the comparison results.
        figures_dir (str): Directory to save the figures.
        num_nodes (int): Number of nodes in the graph.
        num_graphs (int): Number of graphs.
        markdown_file (str): Path to the markdown file to save the results.
    """
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=results_df[['nn_time', 'basic_aco_time', 'ai_aco_time']])
    plt.title(f'Computation Times Comparison (Nodes: {num_nodes}, Graphs: {num_graphs})', fontsize=16)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.xlabel('Methods', fontsize=14)
    plt.xticks([0, 1, 2], ['Nearest Neighbor', 'Basic ACO', 'AI-enhanced ACO'], fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add dashed lines for means and medians
    colors = ['red', 'blue', 'green']
    linestyles = ['--', '-.']
    labels = ['Mean', 'Median']
    for i, method in enumerate(['nn_time', 'basic_aco_time', 'ai_aco_time']):
        mean_val = results_df[method].mean()
        median_val = results_df[method].median()
        plt.axhline(mean_val, color=colors[i], linestyle=linestyles[0], alpha=0.7)
        plt.axhline(median_val, color=colors[i], linestyle=linestyles[1], alpha=0.7)
    
    # Create custom legend
    custom_lines = [plt.Line2D([0], [0], color=c, lw=2, linestyle=ls) for c in colors for ls in linestyles]
    custom_labels = [f'{m} {l}' for m in ['NN', 'Basic ACO', 'AI ACO'] for l in labels]
    plt.legend(custom_lines, custom_labels, title='Statistics', title_fontsize=12, fontsize=10, loc='upper right')
    
    # Perform ANOVA test
    f_val, p_val = f_oneway(results_df['nn_time'], results_df['basic_aco_time'], results_df['ai_aco_time'])
    with open(markdown_file, 'a') as f:
        f.write(f'# Computation Times Statistical Test\n\n')
        f.write(f'**ANOVA p-value:** {p_val:.3e}\n')
        f.write(f'**Effect Size (F-value):** {f_val:.3f}\n\n')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'computation_times_nodes_{num_nodes}_graphs_{num_graphs}.png'), dpi=300)
    plt.close()

def plot_aco_improvement(results_df, figures_dir, num_nodes, num_graphs, markdown_file):
    """
    Plot the improvement of AI ACO over basic ACO and add statistical tests.
    
    Args:
        results_df (pd.DataFrame): A DataFrame containing the comparison results.
        figures_dir (str): Directory to save the figures.
        num_nodes (int): Number of nodes in the graph.
        num_graphs (int): Number of graphs.
        markdown_file (str): Path to the markdown file to save the results.
    """
    plt.figure(figsize=(12, 8))
    sns.histplot(data=results_df, x='aco_improvement', kde=True, color='skyblue', edgecolor='black')
    plt.title(f'AI-enhanced ACO Improvement over Basic ACO (Nodes: {num_nodes}, Graphs: {num_graphs})', fontsize=16)
    plt.xlabel('Improvement (%)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add dashed lines for mean and median
    mean_val = results_df['aco_improvement'].mean()
    median_val = results_df['aco_improvement'].median()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}%')
    plt.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}%')
    
    plt.legend(title='Statistics', title_fontsize=12, fontsize=10, loc='upper right')
    
    # Perform t-test
    t_val, p_val = ttest_ind(results_df['basic_aco_length'], results_df['ai_aco_length'])
    with open(markdown_file, 'a') as f:
        f.write(f'# ACO Improvement Statistical Test\n\n')
        f.write(f'**T-test p-value:** {p_val:.3e}\n')
        f.write(f'**Effect Size (t-value):** {t_val:.3f}\n\n')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'aco_improvement_nodes_{num_nodes}_graphs_{num_graphs}.png'), dpi=300)
    plt.close()

def plot_path_length_vs_nodes(results_df, figures_dir, num_nodes, num_graphs, markdown_file):
    """
    Plot the path lengths vs the number of nodes using a scatter plot with error bars and add statistical tests.
    
    Args:
        results_df (pd.DataFrame): A DataFrame containing the comparison results.
        figures_dir (str): Directory to save the figures.
        num_nodes (int): Number of nodes in the graph.
        num_graphs (int): Number of graphs.
        markdown_file (str): Path to the markdown file to save the results.
    """
    # Group by number of nodes and calculate mean and std for each method
    grouped = results_df.groupby('num_nodes').agg({
        'NN_length': ['mean', 'std'],
        'basic_aco_length': ['mean', 'std'],
        'ai_aco_length': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    grouped.columns = ['num_nodes', 'NN_mean', 'NN_std', 'Basic_ACO_mean', 'Basic_ACO_std', 'AI_ACO_mean', 'AI_ACO_std']

    plt.figure(figsize=(12, 8))

    # Plot scatter points with error bars for each method
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    methods = ['Nearest Neighbor', 'Basic ACO', 'AI-enhanced ACO']
    for i, method in enumerate(['NN', 'Basic_ACO', 'AI_ACO']):
        plt.errorbar(grouped['num_nodes'], grouped[f'{method}_mean'], yerr=grouped[f'{method}_std'], 
                     fmt=f'{markers[i]}-', capsize=5, label=methods[i], color=colors[i], alpha=0.7)

    plt.title(f'Path Lengths vs Number of Nodes (Nodes: {num_nodes}, Graphs: {num_graphs})', fontsize=16)
    plt.xlabel('Number of Nodes', fontsize=14)
    plt.ylabel('Path Length', fontsize=14)
    plt.legend(title='Methods', title_fontsize=12, fontsize=10, loc='upper left')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add jitter to x-axis to avoid overlapping points
    for line in plt.gca().lines:
        line.set_xdata(line.get_xdata() + np.random.normal(0, 0.1, len(line.get_xdata())))
    
    # Add dashed lines for overall means
    for i, method in enumerate(['NN', 'Basic_ACO', 'AI_ACO']):
        plt.axhline(grouped[f'{method}_mean'].mean(), color=colors[i], linestyle='--', 
                    label=f'{methods[i]} Overall Mean', alpha=0.5)
    
    plt.legend(title='Methods and Statistics', title_fontsize=12, fontsize=10, loc='upper left')
    
    # Perform ANOVA test
    f_val, p_val = f_oneway(grouped['NN_mean'], grouped['Basic_ACO_mean'], grouped['AI_ACO_mean'])
    with open(markdown_file, 'a') as f:
        f.write(f'# Path Lengths vs Number of Nodes Statistical Test\n\n')
        f.write(f'**ANOVA p-value:** {p_val:.3e}\n')
        f.write(f'**Effect Size (F-value):** {f_val:.3f}\n\n')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'path_length_vs_nodes_nodes_{num_nodes}_graphs_{num_graphs}.png'), dpi=300)
    plt.close()

def plot_anova_results(results_df, figures_dir, num_nodes, num_graphs, markdown_file):
    """
    Plot the ANOVA results for different methods using violin plots and add statistical tests.
    
    Args:
        results_df (pd.DataFrame): A DataFrame containing the comparison results.
        figures_dir (str): Directory to save the figures.
        num_nodes (int): Number of nodes in the graph.
        num_graphs (int): Number of graphs.
        markdown_file (str): Path to the markdown file to save the results.
    """
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=results_df[['NN_length', 'basic_aco_length', 'ai_aco_length']])
    plt.title(f'ANOVA Results for Different Methods (Nodes: {num_nodes}, Graphs: {num_graphs})', fontsize=16)
    plt.ylabel('Path Length', fontsize=14)
    plt.xlabel('Methods', fontsize=14)
    plt.xticks([0, 1, 2], ['Nearest Neighbor', 'Basic ACO', 'AI-enhanced ACO'], fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add dashed lines for means and medians
    colors = ['red', 'blue', 'green']
    linestyles = ['--', '-.']
    labels = ['Mean', 'Median']
    for i, method in enumerate(['NN_length', 'basic_aco_length', 'ai_aco_length']):
        mean_val = results_df[method].mean()
        median_val = results_df[method].median()
        plt.axhline(mean_val, color=colors[i], linestyle=linestyles[0], alpha=0.7)
        plt.axhline(median_val, color=colors[i], linestyle=linestyles[1], alpha=0.7)
    
    # Create custom legend
    custom_lines = [plt.Line2D([0], [0], color=c, lw=2, linestyle=ls) for c in colors for ls in linestyles]
    custom_labels = [f'{m} {l}' for m in ['NN', 'Basic ACO', 'AI ACO'] for l in labels]
    plt.legend(custom_lines, custom_labels, title='Statistics', title_fontsize=12, fontsize=10, loc='upper right')
    
    # Perform ANOVA test
    f_val, p_val = f_oneway(results_df['NN_length'], results_df['basic_aco_length'], results_df['ai_aco_length'])
    with open(markdown_file, 'a') as f:
        f.write(f'# ANOVA Results Statistical Test\n\n')
        f.write(f'**ANOVA p-value:** {p_val:.3e}\n')
        f.write(f'**Effect Size (F-value):** {f_val:.3f}\n\n')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'anova_results_nodes_{num_nodes}_graphs_{num_graphs}.png'))
    plt.close()

def plot_statistics_summary(summary_df, figures_dir, markdown_file):
    """
    Plot a summary of statistics.
    
    Args:
        summary_df (pd.DataFrame): A DataFrame containing the summary statistics.
        figures_dir (str): Directory to save the figures.
        markdown_file (str): Path to the markdown file to save the results.
    """
    plt.figure(figsize=(16, 10))  
    sns.heatmap(summary_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title('Summary Statistics')
    plt.savefig(os.path.join(figures_dir, 'statistics.png'))
    plt.close()

    # Save the summary statistics as a text file in Markdown format
    summary_text = summary_df.to_markdown()
    with open(markdown_file, 'a') as f:
        f.write('# Summary Statistics\n\n')
        f.write(summary_text)
        f.write('\n\n')