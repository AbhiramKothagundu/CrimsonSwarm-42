import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from datetime import datetime
import pytz

indian_timezone = pytz.timezone('Asia/Kolkata')
current_time = datetime.now(indian_timezone)

os.makedirs(os.path.expanduser('~/plots'), exist_ok=True)
save_path = os.path.join(os.path.expanduser('~/plots'), f'pareto_{current_time.strftime("%d%m%H%M%S")}.png')

def fill_coordinates(df):
    df['Fx'] = [random.randint(0, 200) for _ in range(len(df))]
    df['Fy'] = [random.randint(0, 200) for _ in range(len(df))]

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def compute_distance_fog_devices(fog_device, fog_devices):
    distances = []
    for index, fog in fog_devices.iterrows():
        if index != fog_device['Fog Device'] - 1:
            dist = calculate_distance(fog_device['Fx'], fog_device['Fy'], fog['Fx'], fog['Fy'])
            distances.append(dist)
    return sum(distances) / len(distances)

def compute_distance_eeg_devices(fog_device, eeg_devices):
    distances = []
    for index, eeg_device in eeg_devices.iterrows():
        dist = calculate_distance(fog_device['Fx'], fog_device['Fy'], eeg_device['Fx'], eeg_device['Fy'])
        distances.append(dist)
    return sum(distances) / len(distances)

def compute_transmission_delay(task_size, bandwidth):
    return task_size / bandwidth

def compute_propagation_delay(fog_device, fog_devices, eeg_devices):
    return (compute_distance_fog_devices(fog_device, fog_devices) + compute_distance_eeg_devices(fog_device, eeg_devices)) / (fog_device['SS'])

def compute_processing_delay(fog_device, task_size):
    return task_size / (fog_device['C avg'] * math.pow(10, 9))

def compute_average_queuing_delay(fog_device, no_tasks, task_size):
    total_processing_delay = 0
    for _ in range(1, no_tasks):
        total_processing_delay += compute_processing_delay(fog_device, task_size)
    return total_processing_delay / no_tasks

def compute_processor_performance_index(fog_device, task_size):
    ppi = (1 / 2) * (((fog_device['RAM'] * 1024 * 1024 * 1024 * 8) - task_size) / (fog_device['RAM'] * 1024 * 1024 * 1024 * 8)) + (1 / 2) * (fog_device['MIPS'] * math.pow(10, 6) / (fog_device['C avg'] * math.pow(10, 9)))
    return ppi

def compute_final_parameters(fog_parameters, fog_devices, eeg_devices, l, r):
    for index, fd in fog_devices.iterrows():
        propagation_delay = compute_propagation_delay(fd, fog_devices, eeg_devices)
        processing_delay = compute_processing_delay(fd, l)
        avg_queuing_delay = compute_average_queuing_delay(fd, 3, l)

        total_delay = propagation_delay + processing_delay + avg_queuing_delay
        ppi = compute_processor_performance_index(fd, l)

        fog_parameters.at[index, 'Fog Delay Index'] = total_delay * math.pow(10, 6)
        fog_parameters.at[index, 'Fog Performance Index'] = ppi
    return fog_parameters

def pareto_optimisation_2d(fog_parameters):
    dominated_solutions = []
    non_dominated_solutions = []
    fog_parameters['Fog Delay Index'] = pd.to_numeric(fog_parameters['Fog Delay Index'], errors='coerce')
    fog_parameters['Fog Performance Index'] = pd.to_numeric(fog_parameters['Fog Performance Index'], errors='coerce')

    for i, node1 in fog_parameters.iterrows():
        is_dominating = False
        for j, node2 in fog_parameters.iterrows():
            if i != j:
                if node1['Fog Delay Index'] <= node2['Fog Delay Index'] and node1['Fog Performance Index'] >= node2['Fog Performance Index']:
                    is_dominating = True
                    break
        if is_dominating:
            non_dominated_solutions.append(int(node1['Fog Device']))
        else:
            dominated_solutions.append(int(node1['Fog Device']))

    dom_x = fog_parameters.loc[fog_parameters['Fog Device'].isin(dominated_solutions), 'Fog Delay Index']
    dom_y = fog_parameters.loc[fog_parameters['Fog Device'].isin(dominated_solutions), 'Fog Performance Index']
    non_dom_x = fog_parameters.loc[fog_parameters['Fog Device'].isin(non_dominated_solutions), 'Fog Delay Index']
    non_dom_y = fog_parameters.loc[fog_parameters['Fog Device'].isin(non_dominated_solutions), 'Fog Performance Index']

    non_dom_sorted = fog_parameters.loc[fog_parameters['Fog Device'].isin(non_dominated_solutions)].sort_values(by='Fog Delay Index')

    min_delay_point = non_dom_sorted.loc[non_dom_sorted['Fog Delay Index'].idxmin()]
    max_perf_point = non_dom_sorted.loc[non_dom_sorted['Fog Performance Index'].idxmax()]

    utopia_point_min_delay = non_dom_sorted['Fog Delay Index'].min()
    utopia_point_max_perf = non_dom_sorted['Fog Performance Index'].max()

    plt.scatter(dom_x, dom_y, s=60, c='navy', marker='x', label='DS')
    plt.scatter(non_dom_x, non_dom_y, s=90, c='red', marker='*', label='NDS')
    plt.plot(non_dom_sorted['Fog Delay Index'], non_dom_sorted['Fog Performance Index'], linestyle='-', color='green', label='PF')
    plt.scatter(min_delay_point['Fog Delay Index'], min_delay_point['Fog Performance Index'], c='red', marker='X', s=100, label='RP')
    plt.scatter(max_perf_point['Fog Delay Index'], max_perf_point['Fog Performance Index'], c='red', marker='X', s=100)

    plt.xlabel('Fog Delay Index', weight='bold')
    plt.ylabel('Fog Performance Index', weight='bold')
    plt.legend(loc='best')
    plt.grid()

    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=1200)
    plt.show()

    utopia_point = np.array([utopia_point_min_delay, utopia_point_max_perf])
    non_dom_points = non_dom_sorted[['Fog Delay Index', 'Fog Performance Index']].values
    distances = [euclidean(utopia_point, point) for point in non_dom_points]

    min_distance_index = np.argmin(distances)
    optimal_fog_head = non_dom_sorted.iloc[min_distance_index]

    return dominated_solutions, non_dominated_solutions, utopia_point, optimal_fog_head

class AntColonyOptimizer:
    def __init__(self, num_fog_nodes, num_tasks, pheromone_influence=1.0, heuristic_influence=1.0):
        self.num_fog_nodes = num_fog_nodes
        self.num_tasks = num_tasks
        self.pheromone_levels = np.ones((num_fog_nodes, num_tasks))  # Initialize pheromone levels
        self.pheromone_influence = pheromone_influence
        self.heuristic_influence = heuristic_influence

        # Initialize heuristic_info with random values (can be modified for actual heuristics)
        self.heuristic_info = np.random.rand(num_fog_nodes, num_tasks)  # Shape (num_fog_nodes, num_tasks)

        # Store pheromone and heuristic history for plotting
        self.phero_history = []
        self.heuristic_history = []
        self.allocations = []  # Store the allocation history

    def compute_selection_probabilities(self, task):
        pheromone = self.pheromone_levels[:, task]
        heuristic = self.heuristic_info[:, task]
        total = (pheromone ** self.pheromone_influence) * (heuristic ** self.heuristic_influence)
        probabilities = total / total.sum()  # Normalize probabilities
        return probabilities

    def update_pheromones(self, task, selected_node, decay_rate=0.1):
        # Pheromone evaporation
        self.pheromone_levels *= (1 - decay_rate)

        # Add pheromones based on the selected node and task
        self.pheromone_levels[selected_node, task] += 1  # Example: increment pheromone for selected path

        # Store pheromone levels for plotting
        self.phero_history.append(self.pheromone_levels.copy())
        self.heuristic_history.append(self.heuristic_info.copy())
        self.allocations.append(selected_node)

    def allocate_tasks(self, fog_devices, fog_parameters):
        allocations = []
        for task in range(self.num_tasks):
            probabilities = self.compute_selection_probabilities(task)
            selected_node = np.random.choice(self.num_fog_nodes, p=probabilities)  # Select fog node based on probabilities

            # Store the allocation
            allocations.append(selected_node)

            # Retrieve fog device details
            fog_device = fog_parameters.iloc[selected_node]
            fdi = fog_device['Fog Delay Index']
            fpi = fog_device['Fog Performance Index']

            # Print pheromone and heuristic values
            print(f"Task {task + 1} allocated to Fog Node {selected_node + 1} (FDI: {fdi}, FPI: {fpi})")
            print(f"Current pheromone levels for task {task + 1}: {self.pheromone_levels[:, task]}")
            print(f"Current heuristic values for task {task + 1}: {self.heuristic_info[:, task]}")
            print(f"Reason: Selected based on pheromone levels and heuristic information.\n")

            # Update pheromones after allocation
            self.update_pheromones(task, selected_node)

        return allocations

    def plot_pheromone_and_heuristic(self):
        # Plot pheromone levels
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        for task in range(self.num_tasks):
            pheromone_values = [self.phero_history[i][selected_node, task] for i, selected_node in enumerate(self.allocations)]
            plt.plot(pheromone_values, label=f'Task {task + 1}')
        plt.title('Pheromone Levels Over Tasks')
        plt.xlabel('Iterations')
        plt.ylabel('Pheromone Level')
        plt.legend()
        plt.grid()

        # Plot heuristic levels
        plt.subplot(1, 2, 2)
        for task in range(self.num_tasks):
            heuristic_values = [self.heuristic_history[i][selected_node, task] for i, selected_node in enumerate(self.allocations)]
            plt.plot(heuristic_values, label=f'Task {task + 1}')
        plt.title('Heuristic Values Over Tasks')
        plt.xlabel('Iterations')
        plt.ylabel('Heuristic Value')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

def main():
    # Load datasets
    fog_devices = pd.read_csv('fog_parameters.csv')
    eeg_devices = pd.read_csv('eeg_devices.csv')

    # Fill coordinates for fog and EEG devices
    fill_coordinates(fog_devices)
    fill_coordinates(eeg_devices)

    # Initialize fog parameters DataFrame
    fog_parameters = pd.DataFrame({
        'Fog Device': fog_devices['Fog Device'],
        'Fog Delay Index': np.zeros(len(fog_devices)),
        'Fog Performance Index': np.zeros(len(fog_devices))
    })

    # Compute final parameters
    fog_parameters = compute_final_parameters(fog_parameters, fog_devices, eeg_devices, l=5000, r=1000)

    # Perform Pareto optimization
    dominated_solutions, non_dominated_solutions, utopia_point, optimal_fog_head = pareto_optimisation_2d(fog_parameters)

    # Initialize ACO
    aco = AntColonyOptimizer(num_fog_nodes=len(fog_devices), num_tasks=15)  # Adjust number of tasks as needed
    allocations = aco.allocate_tasks(fog_devices, fog_parameters)

    # Print optimal fog head
    print(f"\nOptimal Fog Head: Fog Device {optimal_fog_head['Fog Device']} (FDI: {optimal_fog_head['Fog Delay Index']}, FPI: {optimal_fog_head['Fog Performance Index']})")

    # Plot pheromone and heuristic values
    aco.plot_pheromone_and_heuristic()

if __name__ == "__main__":
    main()
