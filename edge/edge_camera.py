from flask import Flask, Response, render_template
import cv2
import requests
import uuid

app = Flask(__name__)

FOG_NODE_URL = "http://10.109.110.20:5011/head"

def capture_and_send_video():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Encode frame to JPEG
        _, img_encoded = cv2.imencode('.jpg', frame)
        
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Prepare the request payload with the image and task ID
        img_data = img_encoded.tobytes()
        headers = {'Content-Type': 'application/octet-stream'}
        data = {'task_id': task_id}

        # Send frame to fog node for processing along with the task ID
        try:
            response = requests.post(
                FOG_NODE_URL,
                data=img_data,  # Send the image data
                headers=headers,  # Set the appropriate headers
                params=data  # Send task ID as a query parameter
            )

            if response.status_code == 200:
                fog_data = response.json()
                coordinates = fog_data.get("coordinates", [])
                detection_status = fog_data.get("detection_status", "UNKNOWN")
                print(f"Task ID: {task_id}, Coordinates: {coordinates}, Detection Status: {detection_status}")
            else:
                print(f"Error from fog node: {response.status_code} - {response.text}")
        except Exception as e:
            print("Error sending frame to fog node:", e)
        
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(capture_and_send_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
