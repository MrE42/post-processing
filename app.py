from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import pandas as pd
import os
import logging
import cv2
import supervision as sv
from ultralytics import YOLO

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
    detection_video_path = "static/result.mp4"  # Video generated during object detection
    csv_path = "static/data.csv"
    output_video_path = "static/visualization.mp4"  # New visualization video

    # Check if the detection CSV exists; if not, run object detection
    if not os.path.exists(csv_path) or not os.path.exists(detection_video_path):
        logging.info(f"Detection data or video not found. Running object detection.")
        detection_data = run_object_detection(original_video_path, csv_path, detection_video_path)
    else:
        logging.info(f"Loading existing detection data and video.")
        detection_data = pd.read_csv(csv_path)

        # Ensure numeric columns are correctly typed
        numeric_columns = ['x_min', 'y_min', 'x_max', 'y_max']
        detection_data[numeric_columns] = detection_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Check for NaN values and handle them
        if detection_data[numeric_columns].isnull().values.any():
            logging.warning("NaN values found in numeric columns after conversion. Dropping rows with NaN values.")
            detection_data = detection_data.dropna(subset=numeric_columns)

    # Process gaze data
    fixations, saccades, metrics = process_gaze_data(gaze_data, original_video_path)

    # Generate visualization video using the detection video as the base
    generate_visualization(fixations, saccades, detection_data, detection_video_path, output_video_path)

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

    # Correct artifacts if necessary (e.g., out-of-bounds values)
    x, y = correct_artifacts(x, y, video_path)

    # Smooth the data using a moving average filter
    window_size = 5  # Adjust window size as needed
    x_smooth = running_mean(x, window_size)
    y_smooth = running_mean(y, window_size)

    # Compute velocities using the I-VT algorithm
    velocities = cartesian_velocity(x_smooth, y_smooth, t)

    # Identify fixations and saccades based on velocity threshold
    VELOCITY_THRESHOLD = 100  # Adjust based on data (pixels per second)
    indices_saccades = np.where(velocities > VELOCITY_THRESHOLD)[0]
    indices_fixations = np.where(velocities <= VELOCITY_THRESHOLD)[0]

    # Extract fixation and saccade data
    fixations = [gaze_data[i] for i in indices_fixations]
    saccades = [gaze_data[i] for i in indices_saccades]

    # Calculate metrics
    metrics = {
        'num_fixations': len(fixations),
        'num_saccades': len(saccades),
        'average_fixation_duration': calculate_average_duration(fixations),
        'average_saccade_duration': calculate_average_duration(saccades)
    }

    return fixations, saccades, metrics

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
    durations = [events[i]['timestamp'] - events[i-1]['timestamp'] for i in range(1, len(events))]
    average_duration = np.mean(durations) if durations else 0
    return average_duration

def generate_visualization(fixations, saccades, detection_data, base_video_path, output_video_path):
    # Open video capture and get properties
    cap = cv2.VideoCapture(base_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))

    # Group gaze data by frame
    fixations_df = pd.DataFrame(fixations)
    saccades_df = pd.DataFrame(saccades)

    fixations_df['frame_index'] = (fixations_df['videoTime'] * fps).astype(int)
    saccades_df['frame_index'] = (saccades_df['videoTime'] * fps).astype(int)

    fixation_groups = fixations_df.groupby('frame_index')
    saccade_groups = saccades_df.groupby('frame_index')

    # Convert detection data to a dict for faster access
    detection_dict = detection_data.groupby('frame_index')

    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(f"Generating visualization video: {output_video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw fixations on the frame
        if frame_index in fixation_groups.groups:
            fixation_points = fixation_groups.get_group(frame_index)
            for _, point in fixation_points.iterrows():
                x = int(point['x'])
                y = int(point['y'])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green circles for fixations

        # Draw saccades on the frame
        if frame_index in saccade_groups.groups:
            saccade_points = saccade_groups.get_group(frame_index)
            saccade_points = saccade_points.sort_values('timestamp')  # Ensure correct order
            for idx in range(len(saccade_points) - 1):
                x1 = int(saccade_points.iloc[idx]['x'])
                y1 = int(saccade_points.iloc[idx]['y'])
                x2 = int(saccade_points.iloc[idx+1]['x'])
                y2 = int(saccade_points.iloc[idx+1]['y'])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines for saccades

        # Determine the object being observed
        observed_object = None
        if frame_index in fixation_groups.groups:
            fixation_points = fixation_groups.get_group(frame_index)
            frame_detections = detection_dict.get_group(frame_index) if frame_index in detection_dict.groups else None
            if frame_detections is not None:
                for _, fixation_point in fixation_points.iterrows():
                    for _, detection in frame_detections.iterrows():
                        if is_point_inside_bbox(fixation_point['x'], fixation_point['y'], detection):
                            observed_object = detection['class_name']
                            break
                    if observed_object:
                        break

        # Display the observed object on the frame
        if observed_object:
            cv2.putText(frame, f"Observing: {observed_object}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the frame to the output video
        out.write(frame)

        frame_index += 1

    cap.release()
    out.release()
    logging.info("Visualization video generated successfully.")

def is_point_inside_bbox(x, y, bbox):
    # Ensure coordinates are floats
    x = float(x)
    y = float(y)
    x_min = float(bbox['x_min'])
    y_min = float(bbox['y_min'])
    x_max = float(bbox['x_max'])
    y_max = float(bbox['y_max'])

    return x_min <= x <= x_max and y_min <= y <= y_max

if __name__ == "__main__":
    app.run(debug=True)
