# Active ACO

## Overview
This repository includes implementations of various optimization algorithms for the Traveling Salesman Problem (TSP), including:
- Nearest Neighbor TSP
- Ant Colony Optimization (ACO)
- Improved ACO with Active Inference
- Particle Swarm Optimization (PSO)

Active inference is based on the concept that perception is an active process, involving the minimization of free energy and continuous updating of beliefs based on environmental interactions. This project demonstrates how such concepts can improve traditional algorithms like ACO when applied to randomly generated graphs.

## Files
- `main.py`: Script to run experiments, compare methods, and generate plots.
- `utils.py`: Contains the implementations of the optimization algorithms and utility functions.
- `visualization.py`: Functions for generating plots to visualize the results.

## Usage
Run `main.py` to generate graphs and compare the algorithms. Results are presented both numerically in tables and visually in plots.

## Requirements
Ensure all dependencies are installed:
- numpy
- pandas
- matplotlib
- seaborn
- scipy
