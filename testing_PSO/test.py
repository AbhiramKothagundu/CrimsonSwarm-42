import numpy as np
import matplotlib.pyplot as plt

# Parameters
BWc = 12 * 1e6  # 12 MB/s in bytes per second
BW1 = 300 * 1e6  # 300 MB/s in bytes per second
Msensors = np.arange(200, 2001, 200)  # Varies from 200 to 2000 sensors
C = 1  # Application class
Mparticles = 20  # Number of particles
Niter = 10  # Number of iterations
ω = 0.5  # Inertia weight
c1 = 1.5  # Cognitive parameter
c2 = 1.5  # Social parameter

# Initialize result containers
average_response_times_per_nodes = {}

# Iterate over different numbers of fog nodes
for Nnodes in range(10, 61, 10):  # Nnodes from 10 to 60, incrementing by 10
    average_response_times = []

    for Ms in Msensors:
        λi = np.random.uniform(1, 3, Ms)  # Data rate produced by sensors
        μij = np.random.uniform(50, 300, Nnodes)  # Service rate of fog nodes
        Dsizei = np.random.uniform(250 * 1e3, 1 * 1e6, Ms)  # Data size generated from sensor in bytes
        Lij = np.random.uniform(2, 20, (Ms, Nnodes)) * 1e-3  # Network latency in seconds
        Ljc = 30 * 1e-3  # Cloud latency in seconds
        Rj = np.zeros(Nnodes)
        Raverage = 0

        # PSO Algorithm
        particles = np.random.randint(0, Nnodes, size=(Mparticles, Ms))  # Initialize particle positions
        velocities = np.zeros_like(particles)  # Initialize velocities
        personal_best_positions = particles.copy()  # Best position for each particle
        personal_best_costs = np.full(Mparticles, np.inf)  # Costs associated with personal best positions
        global_best_position = None
        global_best_cost = np.inf

        for _ in range(Niter):
            for particle in range(Mparticles):
                # Calculate response time for the current particle's solution
                for j in range(Nnodes):
                    CommCostij = Lij[:, j] + Dsizei / BW1
                    Rj[j] = np.sum(CommCostij) + np.sum(1 / (μij[j] - λi))

                Raverage = np.mean(Rj)
                
                # Update personal best if the current cost is better
                if Raverage < personal_best_costs[particle]:
                    personal_best_costs[particle] = Raverage
                    personal_best_positions[particle] = particles[particle].copy()

                # Update global best
                if Raverage < global_best_cost:
                    global_best_cost = Raverage
                    global_best_position = particles[particle].copy()

            # Update particle velocities and positions
            for particle in range(Mparticles):
                r1 = np.random.rand(Ms)
                r2 = np.random.rand(Ms)

                velocities[particle] = (ω * velocities[particle] +
                                        c1 * r1 * (personal_best_positions[particle] - particles[particle]) +
                                        c2 * r2 * (global_best_position - particles[particle]))

                # Update positions and ensure they remain within bounds
                particles[particle] += velocities[particle]
                particles[particle] = np.clip(particles[particle], 0, Nnodes - 1).astype(int)  # Keep within bounds

        average_response_times.append(global_best_cost)
    
    average_response_times_per_nodes[Nnodes] = average_response_times

# Plotting the results
plt.figure(figsize=(12, 8))
for Nnodes, response_times in average_response_times_per_nodes.items():
    plt.plot(Msensors, response_times, marker='o', label=f'Nnodes = {Nnodes}')

plt.title('Average Response Time vs. Number of IoT Nodes for Different Fog Nodes (PSO)')
plt.xlabel('Number of IoT Nodes')
plt.ylabel('Average Response Time (s)')
plt.grid(True)
plt.legend()
plt.show()
