import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.spatial.distance import euclidean
from datetime import datetime
import requests
import pytz


# Define the URLs for each fog node
urls = {
    "F1": "http://192.168.49.2:30001/get_status",
    "F2": "http://192.168.49.2:30011/get_status",
    "F3": "http://192.168.49.2:30021/get_status"
}

# Initialize an empty list to hold data from each fog node
data = []

# Fetch data from each fog node
for device, url in urls.items():
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the request fails
        status_data = response.json()  # Parse JSON response
        data.append(status_data)  # Append the data to our list
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {device}: {e}")

# Convert the list of dictionaries into a DataFrame
fog_devices = pd.DataFrame(data)

# Display the DataFrame
print(fog_devices)

fog_devices.to_csv('Results1.csv', index=False)

print("Results1.csv has been created.")


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
    return (compute_distance_fog_devices(fog_device, fog_devices) + compute_distance_eeg_devices(fog_device, eeg_devices)) / (fog_device['SS (m/s)'])

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

    non_dom_indices = fog_parameters[fog_parameters['Fog Device'].isin(non_dominated_solutions)].index

    non_dom_sorted = fog_parameters.loc[fog_parameters['Fog Device'].isin(non_dominated_solutions)].sort_values(by='Fog Delay Index')

    min_delay_point = non_dom_sorted.loc[non_dom_sorted['Fog Delay Index'].idxmin()]
    max_perf_point = non_dom_sorted.loc[non_dom_sorted['Fog Performance Index'].idxmax()]

    utopia_point_min_delay = non_dom_sorted['Fog Delay Index'].min()
    utopia_point_max_perf = non_dom_sorted['Fog Performance Index'].max()

    sizes = [60]
    sizes2 = [90]

    
    plt.scatter(dom_x, dom_y, sizes, c='navy', marker='x', label='DS')
    plt.scatter(non_dom_x, non_dom_y, sizes2, c='red', marker='*', label='NDS')
    plt.plot(non_dom_sorted['Fog Delay Index'], non_dom_sorted['Fog Performance Index'], linestyle='-', color='green', label='PF')
    plt.scatter(min_delay_point['Fog Delay Index'], min_delay_point['Fog Performance Index'], c='red', marker='X', s=100, label='RP')
    plt.scatter(max_perf_point['Fog Delay Index'], max_perf_point['Fog Performance Index'], c='red', marker='X', s=100)
    plt.scatter([utopia_point_min_delay], [utopia_point_max_perf], c='black', edgecolors='cyan', marker='P', s=100, label='UP')

    for i, label in enumerate(non_dominated_solutions):
        plt.text(non_dom_x.iloc[i], non_dom_y.iloc[i] + 0.05, f'NDS {int(label)}',
                 bbox=dict(boxstyle="round", alpha=0.1), fontsize=8, ha='center', va='bottom', color='blue', weight='bold')

    for i, label in enumerate(dominated_solutions):
        plt.text(dom_x.iloc[i], dom_y.iloc[i] + 0.05, f'DS {int(label)}',
                 bbox=dict(boxstyle="round", alpha=0.1), fontsize=8, ha='center', va='bottom', color='blue')

    plt.xlabel('Fog Delay Index', weight='bold')
    plt.ylabel('Fog Performance Index', weight='bold')
    plt.legend(loc='best')
    plt.grid()

    timestamp = current_time.strftime("%d%m%H%M%S")
    plt.show()
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=1200)
    

    utopia_point = np.array([utopia_point_min_delay, utopia_point_max_perf])
    non_dom_points = non_dom_sorted[['Fog Delay Index', 'Fog Performance Index']].values
    distances = [euclidean(utopia_point, point) for point in non_dom_points]

    min_distance_index = np.argmin(distances)
    optimal_fog_head = non_dom_sorted.iloc[min_distance_index]

    if len(non_dom_sorted) > 1:
        non_dom_sorted_without_optimal = non_dom_sorted.drop(index=optimal_fog_head.name)
        utopia_point_alternate = np.array([non_dom_sorted_without_optimal['Fog Delay Index'].min(),
                                           non_dom_sorted_without_optimal['Fog Performance Index'].max()])
        distances_alternate = [euclidean(utopia_point_alternate, point) for point in non_dom_sorted_without_optimal[['Fog Delay Index', 'Fog Performance Index']].values]
        min_distance_index_alternate = np.argmin(distances_alternate)
        alternate_fog_head = non_dom_sorted_without_optimal.iloc[min_distance_index_alternate]
    else:
        alternate_fog_head = None

    return dominated_solutions, non_dominated_solutions, utopia_point, optimal_fog_head, alternate_fog_head

# Load your data
fog_devices = pd.read_csv('Results1.csv')
eeg_devices = pd.read_csv('Results1.csv')

fill_coordinates(fog_devices)
fill_coordinates(eeg_devices)

fog_parameters = pd.DataFrame(columns=['Fog Device', 'Fog Delay Index', 'Fog Performance Index'])
fog_parameters['Fog Device'] = fog_devices['Fog Device']

fog_parameters = compute_final_parameters(fog_parameters, fog_devices, eeg_devices, 1, 2)
dominated, non_dominated, utopia_point, fog_head, alternate_fog_head = pareto_optimisation_2d(fog_parameters)

print("Dominated Solutions:", dominated)
print("Non-Dominated Solutions:", non_dominated)
print("Optimal Fog Head:", fog_head)
print("Alternate Fog Head:", alternate_fog_head)

# After computing final parameters
fog_parameters = compute_final_parameters(fog_parameters, fog_devices, eeg_devices, 1, 2)

# Print FPI and FDI for all devices
for index, row in fog_parameters.iterrows():
    print(f"Fog Device {row['Fog Device']}: FPI = {row['Fog Performance Index']}, FDI = {row['Fog Delay Index']}")

