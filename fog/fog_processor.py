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
    {"name": "Fog Node 2", "url": "http://10.109.110.20:5011/process"}
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

    cpu_freq = psutil.cpu_freq()
    cpu_freq_min = cpu_freq.min
    cpu_freq_max = cpu_freq.max
    cpu_freq_current = cpu_freq.current

    cpu_freq_avg = cpu_freq_current

    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024 ** 3) 
    used_memory = memory_info.used / (1024 ** 3) 
    memory_usage_percent = memory_info.percent 

    status_data = {
        'Fog Device': 1,
        'Fog Processor': 'fog node 1',
        'Fx': '10', 
        'Fy': '20', 
        'SS (m/s)': 299792458,
        'B/W': 100,
        'SNR (dB)': 20,
        'Init Energy (J)': 335700,
        'Idle (W/H)': 1.25,
        'Idle (J)': 4500,
        'Cons (W/H)': 10,
        'Cons (J)': 36000,
        'C max': cpu_freq_max, 
        'C min': cpu_freq_min,  
        'C avg': cpu_freq_avg, 
        'RAM': total_memory, 
        'MIPS': 9000
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
    def __init__(self, num_fog_nodes, pheromone_influence=1.0, heuristic_influence=1.0, fog_head_index=None):
        self.num_fog_nodes = num_fog_nodes
        self.pheromone_influence = pheromone_influence
        self.heuristic_influence = heuristic_influence
        self.fog_head_index = fog_head_index

        # Initialize pheromone levels with 1 column (for the first task)
        self.pheromone_levels = np.ones((num_fog_nodes, 1))
        
        # Initialize heuristic information, can be set to meaningful values based on node parameters
        self.heuristic_info = np.random.rand(num_fog_nodes, 1)

    def compute_selection_probabilities(self, task):
        pheromone = self.pheromone_levels[:, task]
        heuristic = self.heuristic_info[:, task]
        total = (pheromone ** self.pheromone_influence) * (heuristic ** self.heuristic_influence)
        probabilities = total / total.sum()  # Normalize probabilities
        return probabilities

    def resize_pheromone_array(self, task):
        # Resize pheromone and heuristic arrays to accommodate new tasks if necessary
        if task >= self.pheromone_levels.shape[1]:
            new_size = task + 1  # Ensure the array is large enough for the new task
            self.pheromone_levels = np.hstack((self.pheromone_levels, np.ones((self.num_fog_nodes, new_size - self.pheromone_levels.shape[1]))))
            self.heuristic_info = np.hstack((self.heuristic_info, np.random.rand(self.num_fog_nodes, new_size - self.heuristic_info.shape[1])))

    def update_pheromones(self, task, selected_node, decay_rate=0.1):
        # Apply pheromone decay and update for selected node
        self.pheromone_levels *= (1 - decay_rate)
        self.pheromone_levels[selected_node, task] += 1

    def allocate_task(self, task, fog_parameters):
        self.resize_pheromone_array(task)  # Resize to handle new task
        probabilities = self.compute_selection_probabilities(task)
        if self.fog_head_index is not None:
            probabilities[self.fog_head_index] = 0  # Exclude the fog head if specified
            probabilities /= probabilities.sum()  # Normalize again

        selected_node = np.random.choice(self.num_fog_nodes, p=probabilities)

        # Retrieve fog device details
        fog_device = fog_parameters.iloc[selected_node]
        fdi = fog_device['Fog Delay Index']
        fpi = fog_device['Fog Performance Index']

        self.update_pheromones(task, selected_node)

        return selected_node, fdi, fpi


aco = AntColonyOptimizer(num_fog_nodes=len(fog_nodes), pheromone_influence=1.0, heuristic_influence=1.0)

# List to store task info (task_id and the target fog node)
sent_tasks = []


@app.route('/head', methods=['POST'])
def head():
    """
    Receives tasks from the edge node and dynamically assigns them to a Fog Node using ACO.
    """
    global isHead
    isHead = True
    task_id = request.args.get("task_id", str(uuid.uuid4()))
    img_data = request.data

    if not img_data:
        return jsonify({"error": "No image data received"}), 400

    # Dynamically assign task using ACO
    task_index = len(sent_tasks)  # Use current number of tasks as the new task index
    selected_node, fdi, fpi = aco.allocate_task(task=task_index, fog_parameters=fog_parameters)

    fog_node = fog_nodes[selected_node]
    fog_node_name = fog_node["name"]
    fog_node_url = fog_node["url"]

    try:
        response = requests.post(fog_node_url, data=img_data, headers={'Content-Type': 'application/octet-stream'}, params={'task_id': task_id})
        
        if response.status_code == 200:
            print(f"Task {task_id} sent successfully to {fog_node_name}")
            
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
    task_info["progress"] = 100
    task_info["status"] = "Completed"
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

@app.route('/get_tasks', methods=['GET'])
def get_sent_tasks():
    return jsonify({"sent_tasks": sent_tasks})

@app.route('/tasks', methods=['GET'])
def get_tasks():
    return jsonify(tasks)

@app.route('/data')
def data():
    if isHead:
        return render_template('head_org.html', sent_tasks=sent_tasks)
    return render_template('fog_node_data.html', coordinates=[])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)