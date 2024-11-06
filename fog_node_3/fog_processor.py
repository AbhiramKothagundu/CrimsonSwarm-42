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

# Dictionary mapping fog node URLs to their names
fog_nodes = [
    {"name": "Fog Node 1", "url": "http://10.96.137.169:5001/process"},
    {"name": "Fog Node 2", "url": "http://10.109.110.20:5011/process"}
]

@app.route('/get_status', methods=['GET'])
def get_status():
    """
    Returns the hardcoded status for Fog Node 1.
    """
    # Hardcoded status data for Fog Node 1
    status_data = {
        'Fog Device': 3,
        'Fog Processor': 'fog node 3',
        'Fx': '20',
        'Fy': '20',
        'SS (m/s)': 299792458,
        'B/W': 100,
        'SNR (dB)': 20,
        'Init Energy (J)': 335700,
        'Idle (W/H)': 5,
        'Idle (J)': 18000,
        'Cons (W/H)': 30,
        'Cons (J)': 108000,
        'C max': 1.4,
        'C min': 1.9,
        'C avg': 1.65,
        'RAM': 8,
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

sent_tasks = []
current_fog_index = 0

@app.route('/head', methods=['POST'])
def head():
    """
    Receives tasks from the edge node and sends them directly to a specified Fog Node.
    """
    global current_fog_index

    global isHead
    isHead = True
    task_id = request.args.get("task_id", str(uuid.uuid4()))  # Get task_id from query parameter (if any)
    img_data = request.data  # Get the raw image byte data (as received)

    if not img_data:
        return jsonify({"error": "No image data received"}), 400

    fog_node = fog_nodes[current_fog_index]
    fog_node_name = fog_node["name"]
    fog_node_url = fog_node["url"]
    
    # Update the counter to alternate between fog nodes
    current_fog_index = (current_fog_index + 1) % len(fog_nodes)

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
    app.run(host="0.0.0.0", port=5021)