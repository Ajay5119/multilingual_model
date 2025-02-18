from flask import Flask, jsonify
import cv2
from ultralytics import YOLO
from pyngrok import ngrok  # Import pyngrok to handle ngrok tunneling

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with the desired YOLOv8 model

# Function to count people in a single camera frame
def count_people():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if not ret:
        cap.release()
        return {"success": False, "message": "Failed to read from the camera."}

    # Perform detection
    results = model(frame)
    total_people = 0  # Counter for the number of people detected

    # Extract and process results
    for result in results:
        boxes = result.boxes.xyxy.numpy()
        class_ids = result.boxes.cls.numpy()

        # Count people based on the class ID (assuming '0' is 'person')
        total_people += sum(1 for class_id in class_ids if int(class_id) == 0)

    cap.release()
    return {"success": True, "total_people": total_people}

# Define Flask route for detecting people
@app.route('/count_people', methods=['GET'])
def count_people_endpoint():
    response = count_people()
    return jsonify(response)

# Run the Flask app with ngrok tunnel
if __name__ == "__main__":
    # Set up the ngrok tunnel for the Flask app
    public_url = ngrok.connect(5000)
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000\"")

    # Run Flask app
    app.run(host='0.0.0.0', port=5000)
