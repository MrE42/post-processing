from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import os
import logging
import cv2
from ultralytics import YOLO
import supervision as sv

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained YOLO model
MODEL_PATH = "datasets/football-players-detection-12/yolov8n.pt"

# Route to serve the main HTML page
@app.route('/')
def index():
    return render_template('temp.html')

# Endpoint to receive gaze data
@app.route('/submit_gaze_data', methods=['POST'])
def submit_gaze_data():
    gaze_data = request.get_json()
    logging.info(f"Received {len(gaze_data)} gaze points.")

    # Paths for video and data
    original_video_path = "static/soccer.mp4"
    detection_video_path = "static/result.mp4"
    csv_path = "static/data.csv"
    output_video_path = "static/visualization.mp4"
    unprocessed_gaze_csv_path = "static/unprocessed_gaze_data.csv"

    # Save unprocessed gaze data to CSV
    pd.DataFrame(gaze_data).to_csv(unprocessed_gaze_csv_path, index=False)

    # Check if the detection CSV exists; if not, run object detection
    if not os.path.exists(csv_path) or not os.path.exists(detection_video_path):
        logging.info(f"Detection data or video not found. Running object detection.")
        detection_data = run_object_detection(original_video_path, csv_path, detection_video_path)
    else:
        logging.info(f"Loading existing detection data and video.")
        detection_data = pd.read_csv(csv_path)

    # Process gaze data to classify fixations, saccades, and smooth pursuit
    fixations, saccades, smooth_pursuits, metrics = process_gaze_data(gaze_data, original_video_path)

    # Generate visualization video using the detection video as the base
    generate_visualization(fixations, saccades, smooth_pursuits, detection_data, detection_video_path, output_video_path)

    # Save processed gaze data to CSV
    processed_data = fixations + saccades + smooth_pursuits
    pd.DataFrame(processed_data).to_csv("static/processed_gaze_data.csv", index=False)

    # Return the metrics as JSON
    return jsonify({"status": "success", "metrics": metrics, "video_url": output_video_path})

def run_object_detection(video_path: str, csv_output_path: str, output_video_path: str):
    # Initialize lists to store detection data
    detection_records = []

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter to save the detection video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    model = YOLO(MODEL_PATH)
    tracker = sv.ByteTrack()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections from YOLO model
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)


        # Add class_name to the detections
        class_names = [results.names[class_id] for class_id in detections.class_id]

        # Draw detections on the frame
        for i in range(len(detections)):
            x_min, y_min, x_max, y_max = map(int, detections.xyxy[i])
            class_name = class_names[i]
            confidence = detections.confidence[i]
            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame with detections to the output video
        out.write(frame)

        # Prepare data for CSV output
        for i in range(len(detections)):
            record = {
                "frame_index": frame_index,
                "x_min": detections.xyxy[i][0],
                "y_min": detections.xyxy[i][1],
                "x_max": detections.xyxy[i][2],
                "y_max": detections.xyxy[i][3],
                "class_id": detections.class_id[i],
                "confidence": detections.confidence[i],
                "tracker_id": detections.tracker_id[i],
                "class_name": class_names[i]
            }
            detection_records.append(record)

        frame_index += 1

    cap.release()
    out.release()
    logging.info(f"Object detection data saved to '{csv_output_path}'.")
    logging.info(f"Detection video saved to '{output_video_path}'.")

    # Save detections to CSV
    detection_df = pd.DataFrame(detection_records)
    detection_df.to_csv(csv_output_path, index=False)

    return detection_df

def process_gaze_data(gaze_data, video_path):
    # Extract x, y, and timestamp data
    x = np.array([point['x'] for point in gaze_data])
    y = np.array([point['y'] for point in gaze_data])
    t = np.array([point['timestamp'] for point in gaze_data])

    # Correct artifacts if necessary
    x, y = correct_artifacts(x, y, video_path)

    # Smooth the data using a moving average filter
    window_size = 5
    x_smooth = running_mean(x, window_size)
    y_smooth = running_mean(y, window_size)

    # Compute velocities
    velocities = cartesian_velocity(x_smooth, y_smooth, t)

    # Define thresholds for fixation, saccade, and smooth pursuit
    VELOCITY_THRESHOLD_FIXATION = 30
    VELOCITY_THRESHOLD_SACCADE = 150

    fixations = []
    saccades = []
    smooth_pursuits = []

    for i in range(len(velocities)):
        gaze_point = gaze_data[i]
        if velocities[i] < VELOCITY_THRESHOLD_FIXATION:
            fixations.append(gaze_point)
        elif velocities[i] > VELOCITY_THRESHOLD_SACCADE:
            saccades.append(gaze_point)
        else:
            smooth_pursuits.append(gaze_point)

    # Calculate metrics
    metrics = {
        'num_fixations': len(fixations),
        'num_saccades': len(saccades),
        'num_smooth_pursuits': len(smooth_pursuits),
        'average_fixation_duration': calculate_average_duration(fixations),
        'average_saccade_duration': calculate_average_duration(saccades),
        'average_smooth_pursuit_duration': calculate_average_duration(smooth_pursuits)
    }

    return fixations, saccades, smooth_pursuits, metrics

def correct_artifacts(x, y, video_path):
    # Get video dimensions
    cap = cv2.VideoCapture(video_path)
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    # Define video boundaries
    x_min, x_max = 0, video_width
    y_min, y_max = 0, video_height

    # Clip values to be within the video boundaries
    x_clipped = np.clip(x, x_min, x_max)
    y_clipped = np.clip(y, y_min, y_max)

    return x_clipped, y_clipped

def running_mean(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    result = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    # Pad the result to match the original data length
    pad_size = len(data) - len(result)
    result = np.pad(result, (pad_size, 0), mode='edge')
    return result

def cartesian_velocity(x, y, t):
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t) / 1000.0  # Convert milliseconds to seconds
    dt[dt == 0] = np.finfo(float).eps  # Avoid division by zero
    vx = dx / dt
    vy = dy / dt
    velocities = np.sqrt(vx**2 + vy**2)
    # Pad to match the length of the original data
    velocities = np.append(velocities, velocities[-1])
    return velocities

def calculate_average_duration(events):
    if not events:
        return 0
    durations = [events[i]['timestamp'] - events[i - 1]['timestamp'] for i in range(1, len(events))]
    average_duration = np.mean(durations) if durations else 0
    return average_duration

def generate_visualization(fixations, saccades, smooth_pursuits, detection_data, base_video_path, output_video_path):
    cap = cv2.VideoCapture(base_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw fixations as green circles
        for fixation in fixations:
            if int(fixation['videoTime'] * fps) == frame_index:
                x = int(fixation['x'])
                y = int(fixation['y'])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw saccades as red lines
        for i in range(1, len(saccades)):
            if int(saccades[i]['videoTime'] * fps) == frame_index:
                x1, y1 = int(saccades[i - 1]['x']), int(saccades[i - 1]['y'])
                x2, y2 = int(saccades[i]['x']), int(saccades[i]['y'])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw smooth pursuits as blue lines
        for i in range(1, len(smooth_pursuits)):
            if int(smooth_pursuits[i]['videoTime'] * fps) == frame_index:
                x1, y1 = int(smooth_pursuits[i - 1]['x']), int(smooth_pursuits[i - 1]['y'])
                x2, y2 = int(smooth_pursuits[i]['x']), int(smooth_pursuits[i]['y'])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    logging.info("Visualization video generated successfully.")

if __name__ == "__main__":
    app.run(debug=True)