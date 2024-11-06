import cv2
import numpy as np
import pandas as pd
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


fog_nodes = [
    {"name": "Fog Node 3", "url": "http://10.104.122.124:5021/process"},
    {"name": "Fog Node 1", "url": "http://10.96.137.169:5001/process"}
]

fog_parameters = pd.DataFrame({
    'Fog Delay Index': [2, 3],  # Example values
    'Fog Performance Index': [8, 7]  # Example values
})

@app.route('/get_status', methods=['GET'])
def get_status():
    """
    Returns the hardcoded status for Fog Node 1.
    """
    # Hardcoded status data for Fog Node 1
    status_data = {
        'Fog Device': 2,
        'Fog Processor': 'fog node 2',
        'Fx': '0',
        'Fy': '20',
        'SS (m/s)': 299792458,
        'B/W': 100,
        'SNR (dB)': 20,
        'Init Energy (J)': 90000,
        'Idle (W/H)': 2.7,
        'Idle (J)': 9720,
        'Cons (W/H)': 6.4,
        'Cons (J)': 23040,
        'C max': 1.5,
        'C min': 1.5,
        'C avg': 1.5,
        'RAM': 4,
        'MIPS': 12000
    }

    return jsonify(status_data)

@app.route('/get_node_status', methods=['GET'])
def get_node_status():
    """
    Returns the current status of this fog node.
    """
    # Gather service rate and latency (for demonstration; adjust as needed)
    service_rate = 10  # Service rate could be a dynamic calculation
    latency = 10  # Replace with actual latency measurements if available

    # Gather additional metrics
    cpu_usage = psutil.cpu_percent(interval=1)  # Current CPU usage in percentage
    memory_info = psutil.virtual_memory()  # Memory details
    memory_usage = memory_info.percent  # Memory usage in percentage
    network_stats = psutil.net_io_counters()  # Network details
    
    # Data dictionary for status information
    status_data = {
        'service_rate': service_rate,
        'latency': latency,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'sent_bytes': network_stats.bytes_sent,
        'recv_bytes': network_stats.bytes_recv,
        'available_memory': memory_info.available,
        'total_memory': memory_info.total
    }
    
    return jsonify(status_data)


class AntColonyOptimizer:
    def __init__(self, num_fog_nodes, num_tasks, pheromone_influence=1.0, heuristic_influence=1.0, fog_head_index=None):
        self.num_fog_nodes = num_fog_nodes
        self.num_tasks = num_tasks
        self.pheromone_levels = np.ones((num_fog_nodes, num_tasks))  # Initialize pheromone levels
        self.pheromone_influence = pheromone_influence
        self.heuristic_influence = heuristic_influence
        self.fog_head_index = fog_head_index

        # Initialize heuristic_info with random values (can be modified for actual heuristics)
        self.heuristic_info = np.random.rand(num_fog_nodes, num_tasks)  # Shape (num_fog_nodes, num_tasks)

    def compute_selection_probabilities(self, task):
        pheromone = self.pheromone_levels[:, task]
        heuristic = self.heuristic_info[:, task]
        total = (pheromone ** self.pheromone_influence) * (heuristic ** self.heuristic_influence)
        probabilities = total / total.sum()  # Normalize probabilities
        return probabilities

    def resize_pheromone_array(self, task):
    # Resize pheromone levels array if the task exceeds the current size
        if task >= self.pheromone_levels.shape[1]:
            new_size = task + 1  # Ensure the array is large enough for the new task
            new_pheromone_levels = np.ones((self.num_fog_nodes, new_size))
            self.pheromone_levels = new_pheromone_levels


    def update_pheromones(self, task, selected_node, decay_rate=0.1):
        # Pheromone evaporation
        self.pheromone_levels *= (1 - decay_rate)

        # Add pheromones based on the selected node and task
        self.pheromone_levels[selected_node, task] += 1  # Example: increment pheromone for selected path

    def allocate_task(self, task, fog_devices, fog_parameters):
        self.resize_pheromone_array(task)
        probabilities = self.compute_selection_probabilities(task)
        if self.fog_head_index is not None:
            probabilities[self.fog_head_index] = 0  # Set the pheromone for the fog head to 0
            probabilities /= probabilities.sum()  # Normalize probabilities again

        # Select fog node based on probabilities
        selected_node = np.random.choice(self.num_fog_nodes, p=probabilities)

        # Retrieve fog device details
        fog_device = fog_parameters.iloc[selected_node]
        fdi = fog_device['Fog Delay Index']
        fpi = fog_device['Fog Performance Index']

        # Update pheromones after allocation
        self.update_pheromones(task, selected_node)

        # Return the selected node and details
        return selected_node, fdi, fpi

# Initialize Ant Colony Optimizer
aco = AntColonyOptimizer(num_fog_nodes=len(fog_nodes), num_tasks=15, pheromone_influence=1.0, heuristic_influence=1.0)



# List to store task info (task_id and the target fog node)
sent_tasks = []
# current_fog_index = 0

@app.route('/head', methods=['POST'])
def head():
    """
    Receives tasks from the edge node and sends them directly to a specified Fog Node.
    """
    # global current_fog_index

    global isHead
    isHead = True
    task_id = request.args.get("task_id", str(uuid.uuid4()))  # Get task_id from query parameter (if any)
    img_data = request.data  # Get the raw image byte data (as received)

    if not img_data:
        return jsonify({"error": "No image data received"}), 400

    selected_node, fdi, fpi = aco.allocate_task(task=len(sent_tasks), fog_devices=fog_nodes, fog_parameters=fog_parameters)

    fog_node = fog_nodes[selected_node]
    fog_node_name = fog_node["name"]
    fog_node_url = fog_node["url"]
    
    # Update the counter to alternate between fog nodes
    # current_fog_index = (current_fog_index + 1) % len(fog_nodes)

    try:
        response = requests.post(fog_node_url, data=img_data, headers={'Content-Type': 'application/octet-stream'}, params={'task_id': task_id})
        
        if response.status_code == 200:
            print(f"Task {task_id} sent successfully to {fog_node_name}")
            
            # Store the task and the fog node name it was sent to
            sent_tasks.append({
                "task_id": task_id,
                "fog_node": fog_node_name
            })
            
            return jsonify({"message": "Task received and forwarded to Fog Node."})
        else:
            print(f"Failed to forward task {task_id} to {fog_node_url}: {response.text}")
            return jsonify({"error": f"Failed to forward task to Fog Node: {response.text}"}), 500
    
    except Exception as e:
        print(f"Error forwarding task {task_id} to {fog_node_url}: {str(e)}")
        return jsonify({"error": f"Error forwarding task to Fog Node: {str(e)}"}), 500


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
    if isHead:
        return render_template('head_org.html', sent_tasks=sent_tasks)
    return render_template('fog_node_data.html', coordinates=[])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5011)