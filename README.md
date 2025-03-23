# Cloud-Fog-Edge Computing System for Real-Time Video Processing

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [Architecture](#3-architecture)
    - [Components](#components)
    - [Data Flow](#data-flow)
    - [Technologies Used](#technologies-used)
    - [Algorithms](#algorithms)
4. [Repository Structure](#4-repository-structure)
5. [Prerequisites](#5-prerequisites)
6. [Build and Push Docker Images](#6-build-and-push-docker-images)
7. [K3s Deployment](#7-k3s-deployment)
    - [Label Your Nodes](#71-label-your-nodes)
    - [Apply the Deployments and Services](#72-apply-the-deployments-and-services)
8. [Running the System](#8-running-the-system)
9. [Monitoring and Verification](#9-monitoring-and-verification)
10. [System Verification](#10-system-verification)
11. [Troubleshooting](#11-troubleshooting)
12. [Future Enhancements](#12-future-enhancements)
13. [Cleanup](#13-cleanup)
14. [Additional Notes and Commands](#14-additional-notes-and-commands)
15. [Conclusion](#15-conclusion)
16. [License](#16-license)

---

## 1. Project Overview

This project implements a complete **Cloud-Fog-Edge computing architecture** for **real-time video processing** with a focus on **red color detection**. The system captures video through edge devices, processes it via a network of fog nodes, and stores results in the cloud.

By leveraging **Docker** and **K3s (lightweight Kubernetes)**, we create a realistic distributed environment where tasks (video frames) can be offloaded intelligently. This solution demonstrates how to integrate **OpenCV**, **Pareto Optimization**, and **Ant Colony Optimization (ACO)** for efficient task distribution across a fog computing network.

---

## 2. Key Features

1. **Real-Time Video Processing**

    - Captures and processes video frames using OpenCV to detect red objects.

2. **Distributed Computing Architecture**

    - Implements edge devices, multiple fog nodes, and a cloud component.

3. **Intelligent Task Offloading**

    - Uses **Ant Colony Optimization (ACO)** to distribute tasks across fog nodes efficiently.

4. **Dynamic Fog Head Selection**

    - Applies **Pareto optimization** to select the optimal fog head based on system parameters (e.g., computation capacity, latency, energy usage).

5. **Containerized Deployment**

    - Uses **Docker** for containerization and **Kubernetes (K3s)** for orchestration.

6. **Performance Monitoring**
    - Collects and displays processing metrics (CPU, memory usage, etc.) for system analysis.

---

## 3. Architecture

```
Edge (OpenCV) ----> Fog Node(s) ----> Cloud
       |                ^
       |                |
       \----> Fog Head -/ (Pareto + ACO)
```

### Components

1. **Edge Layer**

    - Captures video frames using OpenCV.
    - Sends each frame as a task (byte stream) to the fog head node.
    - Implemented in `edge_camera.py`.

2. **Fog Layer**

    - **Fog Head Node**: Receives tasks from edge devices and distributes them to other fog nodes using ACO.
    - **Fog Worker Nodes**: Process video frames to detect red objects.
    - Collects performance metrics and provides status endpoints.
    - Implemented in `fog_processor.py`.

3. **Cloud Layer**
    - Stores processed results (detection data).
    - Provides a simple visualization or monitoring interface.
    - Implemented in `cloud_storage.py` (or similar).

### Data Flow

1. **Edge Device** captures video frames.
2. Frames are sent to the **Fog Head** node.
3. Fog Head uses **ACO** to select the optimal fog node for processing.
4. Selected fog node processes the frame to detect red objects.
5. Processing results are sent to the **Cloud** for storage.
6. Results can be visualized or logged via the Cloud interface.

### Technologies Used

-   **OpenCV**: Video capture and image processing.
-   **Flask**: RESTful APIs for communication between nodes.
-   **Docker**: Containerization of all components.
-   **Kubernetes (K3s)**: Lightweight orchestration for the distributed system.
-   **NumPy/Pandas**: Data processing and optimization algorithms.
-   **Python**: Primary programming language.

### Algorithms

1. **Ant Colony Optimization (ACO)**

    - Efficiently offloads tasks across fog nodes.
    - Dynamically adjusts based on node performance and load.
    - Implemented in the `AntColonyOptimizer` class in fog nodes.

2. **Pareto Optimization**

    - Selects the **Fog Head** based on multiple parameters (e.g., CPU capacity, data travel time, energy usage).

3. **Red Object Detection**
    - Uses HSV color space for robust color detection.
    - Contour detection for object localization.
    - Returns coordinates of detected red objects.

---

## 4. Repository Structure

```
/
├── cloud/
│   ├── templates/
│   ├── cloud_deployment.yaml
│   ├── cloud_service.yaml
│   ├── cloud_storage.py
│   └── Dockerfile
├── edge/
│   ├── templates/
│   ├── edge_deployment.yaml
│   ├── edge_service.yaml
│   ├── edge_camera.py
│   └── Dockerfile
├── fog/
│   ├── templates/
│   ├── fog_deployment.yaml
│   ├── fog_service.yaml
│   ├── fog_processor.py
│   └── Dockerfile
├── fog_node_2/
│   ├── templates/
│   ├── fog_deployment.yaml
│   ├── fog_service.yaml
│   ├── fog_processor.py
│   └── Dockerfile
└── fog_node_3/
    ├── templates/
    ├── fog_deployment.yaml
    ├── fog_service.yaml
    ├── fog_processor.py
    └── Dockerfile
```

Each folder contains:

-   A `Dockerfile` for building the container image.
-   YAML files (`deployment.yaml`, `service.yaml`) for deploying and exposing the services in K3s.
-   The Python application code (`*processor.py`, `*camera.py`, etc.) for each role (Cloud, Fog, Edge).

---

## 5. Prerequisites

-   **Docker** and a Docker Hub (or similar registry) account.
-   **K3s** (lightweight Kubernetes) cluster installed and running.
-   **kubectl** configured to communicate with your K3s cluster.
-   **Python 3.x** with required libraries (OpenCV, Flask, NumPy, Pandas, Requests, psutil).
-   **Camera** (webcam) or video input for the Edge device.
-   **Minikube** (optional, for local Kubernetes cluster):
    ```bash
    minikube start --driver=docker
    ```

---

## 6. Build and Push Docker Images

1. **Build the Edge image**:

    ```bash
    docker build -t <your-dockerhub-username>/edge_image ./edge
    docker push <your-dockerhub-username>/edge_image
    ```

2. **Build the Fog image**:

    ```bash
    docker build -t <your-dockerhub-username>/fog_image ./fog
    docker push <your-dockerhub-username>/fog_image
    ```

3. **Build the Cloud image**:

    ```bash
    docker build -t <your-dockerhub-username>/cloud_image ./cloud
    docker push <your-dockerhub-username>/cloud_image
    ```

4. **Build Fog Node 2 image**:

    ```bash
    docker build -t <your-dockerhub-username>/fog_image_2 ./fog_node_2
    docker push <your-dockerhub-username>/fog_image_2
    ```

5. **Build Fog Node 3 image**:
    ```bash
    docker build -t <your-dockerhub-username>/fog_image_3 ./fog_node_3
    docker push <your-dockerhub-username>/fog_image_3
    ```

_(Replace `<your-dockerhub-username>` with your actual Docker Hub username or another registry.)_

---

## 7. K3s Deployment

### 7.1 Label Your Nodes

(Optional) If you have multiple nodes, label them according to their role:

```bash
kubectl label node <node-name> role=cloud
kubectl label node <node-name> role=edge
kubectl label node <node-name> role=fog
```

### 7.2 Apply the Deployments and Services

Apply each deployment and service to the cluster:

```bash
# Deploy Edge
kubectl apply -f edge/edge_deployment.yaml
kubectl apply -f edge/edge_service.yaml

# Deploy Fog (Fog Node 1)
kubectl apply -f fog/fog_deployment.yaml
kubectl apply -f fog/fog_service.yaml

# Deploy Fog Node 2
kubectl apply -f fog_node_2/fog_deployment.yaml
kubectl apply -f fog_node_2/fog_service.yaml

# Deploy Fog Node 3
kubectl apply -f fog_node_3/fog_deployment.yaml
kubectl apply -f fog_node_3/fog_service.yaml

# Deploy Cloud
kubectl apply -f cloud/cloud_deployment.yaml
kubectl apply -f cloud/cloud_service.yaml
```

Check the status of your pods and services:

```bash
kubectl get pods -o wide
kubectl get services -o wide
```

---

## 8. Running the System

1. **Ensure your camera is accessible** on the machine running the Edge container.
2. The **Edge** container (`edge_camera.py`) captures frames, encodes them as byte streams, and sends them to the Fog layer (at the Fog Head endpoint).
3. **Fog Nodes** run the script (`fog_processor.py`) that can act as either:
    - A **normal fog node** (receiving tasks from the Fog Head), or
    - A **Fog Head** (selected via Pareto optimization).
4. The **Fog Head** uses the **Ant Colony Optimization (ACO)** logic to offload tasks to other Fog nodes.
5. The **Cloud** receives detection logs and stores them. Access it via its exposed service port (e.g., port `5002`) to view detection results (in `index.html` or logs).

---

## 9. Monitoring and Verification

-   **Edge Node**

    -   Visit the `/video_feed` route (if exposed) or check logs to confirm frames are being captured and sent.

-   **Fog Node**

    -   Check logs to see if tasks are being received and processed.
    -   Endpoints like `/get_node_status` or `/get_status` (if exposed) show CPU usage, memory usage, etc.
    -   Confirm that the ACO logs show tasks being offloaded among fog nodes.

-   **Cloud Node**
    -   Access the root (`/`) or `/status` endpoint to see stored detections.
    -   Logs should show detection entries being received.

When a red object appears in front of the camera, the logs or output should show **"DETECTED RED"** (or **"NO RED DETECTED"** if none is present).

---

## 10. System Verification

1. **Accessing the Interfaces**

    - **Edge Node** UI: `http://<EDGE_IP>:5000`
    - **Fog Node** UI: `http://<FOG_IP>:5001/data`
    - **Cloud Node** UI: `http://<CLOUD_IP>:5002`

2. **Testing the System**

    - Place a red object in front of the camera.
    - The system should detect the object and display its coordinates.
    - Results should be visible in the cloud node interface (or logs).

3. **Accessing the Services via Minikube IP**

    After running the following commands to get the status of services and pods:

    ```bash
    kubectl get services -o wide
    kubectl get pods -o wide
    ```

    Use the Minikube IP to access the services. Run:

    ```bash
    minikube ip
    ```

    This will give you an IP address, for example, `192.168.49.2`.

    Use this IP address to access the services in your browser:

    - **Edge Node** (Video Feed): `http://192.168.49.2:30000/`
    - **Fog Node 1**: `http://192.168.49.2:30001/data`
    - **Fog Node 2**: `http://192.168.49.2:30011/data`
    - **Fog Node 3**: `http://192.168.49.2:30021/data`
    - **Cloud Node** (Detection Results): `http://192.168.49.2:30002/`

---

## 11. Troubleshooting

If you encounter issues, consider the following steps:

1. **Check Pod Status**

    ```bash
    kubectl get pods
    kubectl describe pod <pod-name>
    kubectl logs <pod-name>
    ```

2. **Verify Network Connectivity**

    - Ensure all nodes can communicate with each other.
    - Check if services are properly exposed (e.g., NodePort, ClusterIP).

3. **Restart Deployments**
    ```bash
    kubectl delete deployments --all
    kubectl delete pods --all
    kubectl delete services --all
    # Then redeploy using the steps above
    ```

---

## 12. Future Enhancements

-   Add support for detecting multiple colors or objects.
-   Implement advanced machine learning models for more complex object detection.
-   Add authentication and security features for production environments.
-   Optimize resource allocation algorithms and implement load balancing.
-   Introduce fault tolerance and recovery mechanisms across fog nodes.

---

## 13. Cleanup

If you want to remove all deployments and services from your cluster:

```bash
# Delete all deployments
kubectl delete deployments --all

# Delete all pods (if any remain)
kubectl delete pods --all

# Delete all services
kubectl delete services --all

# Delete all HPA (Horizontal Pod Autoscalers), if used
kubectl delete hpa --all
```

---

## 14. Additional Notes and Commands

-   **View Node Token for K3s**

    ```bash
    sudo cat /var/lib/rancher/k3s/server/node-token
    ```

-   **Join Another Node to K3s**

    ```bash
    curl -sfL https://get.k3s.io | K3S_URL=https://<YOUR_K3S_SERVER_IP>:6443 K3S_TOKEN=<TOKEN> sh -
    ```

-   **Update kubeconfig** (to run `kubectl` from your local machine)

    ```bash
    sudo chmod 644 /etc/rancher/k3s/k3s.yaml
    export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
    ```

-   **Port Forwarding** (example iptables rules if needed)

    ```bash
    sudo iptables -t nat -A PREROUTING -p tcp --dport 30000 -j DNAT --to-destination 192.168.49.2:30000
    sudo iptables -t nat -A POSTROUTING -j MASQUERADE
    ```

-   **Check Pod and Service Status**
    ```bash
    kubectl get pods -o wide
    kubectl get services -o wide
    ```

Adjust these commands as needed for your specific environment.

---

## 15. Conclusion

This project demonstrates a complete **Cloud-Fog-Edge** computing architecture for real-time video processing. By leveraging **Docker**, **K3s**, **OpenCV**, and **optimization algorithms** (Pareto + ACO), the system provides an efficient, scalable solution for distributed computing. This proof-of-concept can be extended to handle more complex scenarios, additional optimization metrics, and real-world edge devices.

---

## 16. License

This project is licensed under the terms of the [MIT License](https://github.com/AbhiramKothagundu/CrimsonSwarm-42/blob/main/LICENSE).
