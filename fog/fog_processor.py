import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import requests
import psutil
import uuid
from datetime import datetime

CLOUD_NODE_URL = "http://10.110.253.80:5002/store"
EDGE_NODE_URL = "http://10.96.137.169:5000/receive"  # Update to your Edge node URL

app = Flask(__name__)

tasks = []  # List to store task information
isHead = False

FOG_NODE_URLS = [
    # "http://10.96.137.169:5001/process",  # Fog node 1
    "http://10.109.110.20:5011/process",   # Fog node 2
    "http://10.104.122.124:5021/process"    # Fog node 3
]


# Get status from fog nodes
def get_fog_status():
    status_data = []
    for url in FOG_NODE_URLS:
        try:
            response = requests.get(f"{url}/get_status")
            if response.status_code == 200:
                status_data.append(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching status from {url}: {e}")
    return status_data

# ACO parameters
BWc = 12 * 1e6  # 12 MB/s in bytes per second
BW1 = 300 * 1e6  # 300 MB/s in bytes per second
Mants = 3  # Number of ants
Niter = 10  # Number of iterations
ρ = 0.1  # Pheromone evaporation rate
ρg = 0.2  # Global evaporation rate
α = 0.2  # Heuristic parameter for pheromone effect
β = 2  # Heuristic parameter for task offloading quality

# Task offloading using ACO
def aco_task_offloading(Ms, Nnodes, λi, μij, Dsizei, Lij):
    # Initialize pheromone matrix
    pheromone_matrix = np.ones((Ms, Nnodes))
    best_solution = None
    best_cost = float('inf')

    Rj = np.zeros(Nnodes)
    Raverage = 0

    for _ in range(Niter):
        tabu_tables = [set() for _ in range(Mants)]  # Taboo tables for each ant
        solutions = []

        for ant in range(Mants):
            current_solution = []
            for sensor in range(Ms):
                probabilities = np.zeros(Nnodes)
                for j in range(Nnodes):
                    loadj = 1 - (Rj[j] - Raverage) / Rj[j] if Rj[j] != 0 else 1
                    ηij = loadj / Rj[j] if Rj[j] != 0 else 1
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
                tabu_tables[ant].add(selected_node)

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

    return best_solution, best_cost

# List to store task info (task_id and the target fog node)
sent_tasks = []

@app.route('/head', methods=['POST'])
def head():
    """
    Receives tasks from the edge node and sends them to the best fog node based on ACO optimization.
    """
    task_id = request.args.get("task_id", str(uuid.uuid4()))  # Get task_id from query parameter (if any)
    img_data = request.data  # Get the raw image byte data (as received)

    if not img_data:
        return jsonify({"error": "No image data received"}), 400

    # Fetch the fog status
    fog_status = get_fog_status()
    
    if not fog_status:
        return jsonify({"error": "No valid fog node status received"}), 500

    # Prepare data for ACO
    Ms = len(fog_status)  # Number of sensors (same as the number of fog nodes)
    Nnodes = len(fog_status)
    
    λi = np.random.uniform(1, 3, Ms)  # Data rate produced by sensors
    μij = np.random.uniform(50, 300, Nnodes)  # Service rate of fog nodes
    Dsizei = np.random.uniform(250 * 1e3, 1 * 1e6, Ms)  # Data size generated from sensor in bytes
    Lij = np.random.uniform(2, 20, (Ms, Nnodes)) * 1e-3  # Network latency in seconds

    # Perform ACO to find the best fog node for task offloading
    best_solution, best_cost = aco_task_offloading(Ms, Nnodes, λi, μij, Dsizei, Lij)

    # Find the best fog node (with the lowest cost)
    best_node_idx = np.argmin(best_solution.sum(axis=0))  # Select the node with the lowest pheromone cost
    fog_node_url = FOG_NODE_URLS[best_node_idx]

    # Forward the received image data and task ID to the selected fog node
    try:
        response = requests.post(fog_node_url, data=img_data, headers={'Content-Type': 'application/octet-stream'}, params={'task_id': task_id})
        
        if response.status_code == 200:
            print(f"Task {task_id} sent successfully to {fog_node_url}")
            
            # Store the task and the fog node it was sent to
            sent_tasks.append({
                "task_id": task_id,
                "fog_node": fog_node_url
            })
            
            return jsonify({"message": "Task received and forwarded to Fog Node."})
        else:
            print(f"Failed to forward task {task_id} to {fog_node_url}: {response.text}")
            return jsonify({"error": f"Failed to forward task to Fog Node: {response.text}"}), 500
    
    except Exception as e:
        print(f"Error forwarding task {task_id} to {fog_node_url}: {str(e)}")
        return jsonify({"error": f"Error forwarding task to Fog Node: {str(e)}"}), 500


@app.route('/get_status', methods=['GET'])
def get_status():
    """
    Returns the hardcoded status for Fog Node 1.
    """
    # Hardcoded status data for Fog Node 1
    status_data = {
        'Fog Device': 'F1',
        'Fx': '',  # Leave empty or add value if needed
        'Fy': '',  # Leave empty or add value if needed
        'SS (m/s)': 299792458,
        'B/W': 100,
        'SNR (dB)': 20,
        'Init Energy (J)': 335700,
        'Idle (W/H)': 1.25,
        'Idle (J)': 4500,
        'Cons (W/H)': 10,
        'Cons (J)': 36000,
        'C max': 1.43,
        'C min': 1.43,
        'C avg': 1.43,
        'RAM': 4,
        'MIPS': 9000
    }

    return jsonify(status_data)


@app.route('/process', methods=['POST'])
def process_frame():
    print("Received a frame for processing.")
    # Get task ID from query parameters
    task_id = request.args.get("task_id", str(uuid.uuid4()))  # Use request.args to get query parameter
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize task details
    task_info = {
        "task_id": task_id,
        "status": "Processing",
        "progress": 0,
        "timestamp": timestamp,
        "detection_status": None
    }
    
    # Append the task info to tasks list
    tasks.append(task_info)

    # Read the image data from the request body
    img_data = request.data  # Use request.data to get the raw byte data
    img_data_np = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(img_data_np, cv2.IMREAD_COLOR)
    
    if frame is None:
        print("Failed to decode frame.")
        return jsonify({"error": "Failed to decode frame."}), 400

    # Red ball detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours of the detected red areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coordinates = []  # List to store coordinates of detected red objects

    # Extract coordinates of the detected contours
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:  # To avoid division by zero
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coordinates.append((cX, cY))
            print(f"Detected red object at: ({cX}, {cY})")  # Log detected coordinates

            # Draw a circle on the original frame (optional for visualization)
            cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)  # Marking the position

    # Update detection status
    if coordinates:  # If coordinates are found
        task_info["detection_status"] = "DETECTED RED"
    else:
        task_info["detection_status"] = "NO RED DETECTED"

    # Update progress
    task_info["progress"] = 100  # Mark progress as complete
    print(f"Detection status: {task_info['detection_status']}")  # Log detection status

    requests.post(EDGE_NODE_URL, json={
        "detection": {
            "task_id": task_id,
            "coordinates": coordinates,
            "detection_status": task_info["detection_status"],
            "timestamp": timestamp
        }
    })

    # Send the detection result to the cloud node
    requests.post(CLOUD_NODE_URL, json={
        "detection": {
            "task_id": task_id,
            "coordinates": coordinates,
            "detection_status": task_info["detection_status"],
            "timestamp": timestamp
        }
    })

    return jsonify({"coordinates": coordinates, "detection_status": task_info["detection_status"]})

@app.route('/tasks', methods=['GET'])
def get_tasks():
    return jsonify(tasks)

@app.route('/data')
def data():
    if isHead == True:
        return render_template('head_org.html', sent_tasks=sent_tasks)
    return render_template('fog_node_data.html', coordinates=[])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)