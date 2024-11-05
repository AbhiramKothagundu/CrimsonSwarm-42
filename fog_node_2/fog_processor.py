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

FOG_NODE_URLS = [
    # "http://10.96.137.169:5001/process",  # Fog node 1
    "http://10.109.110.20:5011/process",   # Fog node 2
    # "http://10.104.122.124:5021/process"    # Fog node 3
]


@app.route('/get_status', methods=['GET'])
def get_status():
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



@app.route('/head', methods=['POST'])
def head():
    """
    Receives tasks from the edge node and distributes them to available fog nodes.
    """
    data = request.get_json()
    # Assume data is a list of tasks
    tasks_to_process = data.get("tasks", [])

    if not tasks_to_process:
        return jsonify({"error": "No tasks received"}), 400

    for task in tasks_to_process:
        task_id = task.get("task_id", str(uuid.uuid4()))  # Use provided task_id or generate a new one
        img_data = task.get("image")  # Assuming the image data is sent in base64 or binary format
        
        # Distributing the task to fog nodes
        for fog_node_url in FOG_NODE_URLS:
            try:
                response = requests.post(fog_node_url, json={"task_id": task_id, "image": img_data})
                if response.status_code == 200:
                    print(f"Task {task_id} distributed successfully to {fog_node_url}")
                else:
                    print(f"Failed to distribute task {task_id} to {fog_node_url}: {response.text}")
            except Exception as e:
                print(f"Error distributing task {task_id} to {fog_node_url}: {str(e)}")

    return jsonify({"message": "Tasks received and distributed."})


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
    return render_template('fog_node_data.html', coordinates=[])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5011)