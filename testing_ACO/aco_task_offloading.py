import numpy as np
import matplotlib.pyplot as plt

# Parameters
BWc = 12 * 1e6  # 12 MB/s in bytes per second
BW1 = 300 * 1e6  # 300 MB/s in bytes per second
Msensors = np.arange(200, 2001, 200)  # Varies from 200 to 2000 sensors
C = 1  # Application class
Mants = 3  # Number of ants
Niter = 10  # Number of iterations
ρ = 0.1  # Pheromone evaporation rate
ρg = 0.2  # Global evaporation ratea
α = 0.2  # Heuristic parameter for pheromone effect
β = 2  # Heuristic parameter for task offloading quality

np.random.seed(42)

# Initialize result containers
average_response_times_per_nodes = {}

# Iterate over different numbers of fog nodes
for Nnodes in range(10, 61, 10):  # Nnodes from 10 to 60, incrementing by 10
    average_response_times = []
    
    for Ms in Msensors:
        λi = np.random.uniform(1, 3, Ms)  # Data rate produced by sensors
        μij = np.random.uniform(50, 300, Nnodes)  # Service rate of fog nodes
        Dsizei = np.random.uniform(250 * 1e3, 1 * 1e6, len(Msensors))  # Data size generated from sensor in bytes
        Lij = np.random.uniform(2, 20, (len(Msensors), Nnodes)) * 1e-3  # Network latency in seconds
        Ljc = 30 * 1e-3  # Cloud latency in seconds
        Rj = np.zeros(Nnodes)
        Raverage = 0

        # Ant Colony Optimization (ACO) Algorithm
        pheromone_matrix = np.ones((Ms, Nnodes))
        best_solution = None
        best_cost = float('inf')

        for _ in range(Niter):
            tabu_tables = [set() for _ in range(Mants)]  # Taboo tables for each ant
            solutions = []
            
            for ant in range(Mants):
                current_solution = []
                for sensor in range(Ms):
                    # Compute probabilities
                    probabilities = np.zeros(Nnodes)
                    for j in range(Nnodes):
                        if Rj[j] != 0:
                            loadj = 1 - (Rj[j] - Raverage) / Rj[j]
                            ηij = loadj / Rj[j]
                        else:
                            ηij = 1

                        probabilities[j] = (pheromone_matrix[sensor, j] * α) * (ηij * β)

                    # Normalize probabilities
                    sum_probabilities = np.sum(probabilities)
                    if sum_probabilities > 0:
                        probabilities /= sum_probabilities
                    else:
                        probabilities.fill(1 / Nnodes)

                    # Roulette wheel selection
                    selected_node = np.random.choice(Nnodes, p=probabilities)
                    current_solution.append(selected_node)
                    tabu_tables[ant].add(selected_node)  # Add to taboo table
                
                solutions.append(current_solution)

            # Update pheromone matrix
            pheromone_updates = np.zeros_like(pheromone_matrix)
            for ant in range(Mants):
                for sensor in range(Ms):
                    selected_node = solutions[ant][sensor]
                    if Rj[selected_node] != 0:
                        pheromone_updates[sensor, selected_node] += 1 / Rj[selected_node]
            
            pheromone_matrix = (1 - ρ) * pheromone_matrix + ρ * pheromone_updates

            # Global pheromone update
            for j in range(Nnodes):
                if Rj[j] != 0:
                    pheromone_matrix[:, j] = (1 - ρg) * pheromone_matrix[:, j] + ρg * (1 / Rj[j])

            # Calculate response times
            for j in range(Nnodes):
                CommCostij = Lij[:, j] + Dsizei / BW1
                Rj[j] = np.sum(CommCostij) + np.sum(1 / (μij[j] - λi))

            Raverage = np.mean(Rj)

            # Compare with the best solution
            if Raverage < best_cost:
                best_cost = Raverage
                best_solution = pheromone_matrix.copy()

        average_response_times.append(best_cost)
    
    average_response_times_per_nodes[Nnodes] = average_response_times

# Plotting the results
plt.figure(figsize=(12, 8))
for Nnodes, response_times in average_response_times_per_nodes.items():
    plt.plot(Msensors, response_times, marker='o', label=f'Nnodes = {Nnodes}')

plt.title('Average Response Time vs. Number of IoT Nodes for Different Fog Nodes')
plt.xlabel('Number of IoT Nodes')
plt.ylabel('Average Response Time (s)')
plt.grid(True)
plt.legend()
plt.show()