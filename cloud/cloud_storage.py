from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

detections = []

@app.route('/')
def index():
    # Pass the detections to the index template
    return render_template('index.html', detections=detections)


@app.route('/store', methods=['POST'])
def store_detection():
    detection = request.json.get("detection")
    detections.append(detection)
    print("Stored detection:", detection)
    return jsonify({"status": "success", "detections": detections})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
