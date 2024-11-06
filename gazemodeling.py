from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import os
import logging
import cv2
from ultralytics import YOLO
import supervision as sv
import subprocess

from deep_em import gazeprocess

gaze = True
testing = True

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained YOLO model
MODEL_PATH = "datasets/football-players-detection-12/yolo11n.pt"


# project_path = os.path.expanduser('~')
current_path = os.getcwd()

current_directory = current_path.replace('\\', '/')+"/"


original_video_path = "static/soccer.mp4"
detection_video_path = "static/result.mp4"
object_csv_path = "static/data.csv"
output_video_path = "static/visualization.mp4"
unprocessed_gaze_csv_path = current_directory+"deep_em/unprocessed_gaze_data.csv"
processed_gaze_csv_path = current_directory+"deep_em/processed_gaze_data.csv"

# Route to serve the main HTML page
@app.route('/')
def index():
    return render_template('temp.html')

# Endpoint to receive gaze data
@app.route('/submit_gaze_data', methods=['POST'])
def submit_gaze_data():
    gaze_data = request.get_json()
    logging.info(f"Received {len(gaze_data)} gaze points.")

    # Save unprocessed gaze data to CSV
    pd.DataFrame(gaze_data).to_csv(unprocessed_gaze_csv_path, index=False)

    return main()

def run_object_detection(video_path: str, csv_output_path: str, output_video_path: str):
    # Initialize lists to store detection data
    detection_records = []
    d2 = []

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



def process_gaze_data(data_in, data_out, process):
    fixations = []
    saccades = []
    smooth_pursuits = []
    other = []


    '''
    if process:
        env = os.environ.copy()

        os.chdir(current_path+'\\deep_em')
        subprocess.run([current_path+'\\venv\\Scripts\\python.exe','gazeprocess.py', data_in, data_out, '1'],
                                 shell=True, env=env)
        os.chdir(current_path)
    '''

    gazeprocess.main(data_in, data_out, '1', current_path)

    gazed = pd.read_csv(data_out)

    for i in range(len(gazed.index)):
        point = gazed.iloc[i]  # This returns the entire row as a pandas Series

        if point['classification'] == 'FIX':
            fixations.append(point[['x', 'y', 'videoTime', 'classification']])
        elif point['classification'] == 'SACCADES':
            saccades.append(point[['x', 'y', 'videoTime', 'classification']])
        elif point['classification'] == 'SP':
            smooth_pursuits.append(point[['x', 'y', 'videoTime', 'classification']])
        else:
            other.append(point[['x', 'y', 'videoTime', 'classification']])

    # Calculate metrics for fixations, saccades, smooth pursuits
    metrics = {
        'num_fixations': len(fixations),
        'num_saccades': len(saccades),
        'num_smooth_pursuits': len(smooth_pursuits),
        'average_fixation_duration': calculate_average_duration(fixations),
        'average_saccade_duration': calculate_average_duration(saccades),
        'average_smooth_pursuit_duration': calculate_average_duration(smooth_pursuits)
    }

    return fixations, saccades, smooth_pursuits, other, metrics


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
    durations = [events[i]['videoTime'] - events[i - 1]['videoTime'] for i in range(1, len(events))]
    average_duration = np.mean(durations) if durations else 0
    return average_duration

def list_adjust(total, buffer, fps, frame_index):
    past = 0
    current = []
    for t in total:
        if (t['videoTime'] + buffer) * fps >= frame_index:
            if (t['videoTime'] - buffer) * fps <= frame_index:
                current.append(t)
            else:
                break
        else:
            past += 1

    while past > 1:
        total.pop(0)
        past -= 1
    return total, current

def generate_visualization(fixations, saccades, smooth_pursuits, other, detection_data, base_video_path, output_video_path):
    logging.info("Generating visualization video...")

    cap = cv2.VideoCapture(base_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    total = fixations + saccades + smooth_pursuits + other
    total.sort(key=lambda x:x['videoTime'])

    frame_index = 0
    annotator = sv.BoxCornerAnnotator()
    annotator.color = sv.Color(r=0, g=0, b=255)
    annotator.thickness = 4

    annotator2 = sv.CircleAnnotator()
    annotator2.color = sv.Color(r=255, g=0, b=0)
    annotator2.thickness = 6


    if len(total) >= 3:
        gps=float(1/float(total[2]['videoTime']-total[1]['videoTime']))
    else:
        gps=30

    fpg = float(fps/gps) #frames per gaze (around 2 usually)

    buffer = fpg/gps
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Filter detections for the current frame
        current_detections = detection_data[detection_data['frame_index'] == frame_index]

        if not current_detections.empty:
            v = False
            detections_list = []
            viewing_list = []

            total, current = list_adjust(total, buffer, fps, frame_index)

            for _, detection in current_detections.iterrows():
                x_min, y_min, x_max, y_max = int(detection['x_min']), int(detection['y_min']), int(detection['x_max']), int(detection['y_max'])
                class_id = detection['class_id']
                confidence = detection['confidence']
                tracker_id = detection['tracker_id']

                detection_entry = {
                    'xyxy': [x_min, y_min, x_max, y_max],
                    'class_id': class_id,
                    'confidence': confidence,
                    'tracker_id': tracker_id
                }

                #For precision
                b = 5

                for c in current:
                    if x_min-b <= c['x'] <= x_max+b and y_min-b <= c['y'] <= y_max+b:
                        viewing_list.append(detection_entry)
                        v = True
                        break
                        
                if not v:
                    detections_list.append(detection_entry)

            if not detections_list == []:
                # Create sv.Detections object
                detections = sv.Detections(
                    xyxy=np.array([d['xyxy'] for d in detections_list]),
                    class_id=np.array([d['class_id'] for d in detections_list]),
                    confidence=np.array([d['confidence'] for d in detections_list]),
                    tracker_id=np.array([d['tracker_id'] for d in detections_list])
                )

                # Annotate the frame with the detections
                frame = annotator.annotate(scene=frame, detections=detections)

            if not viewing_list == []:
                detections2 = sv.Detections(
                    xyxy=np.array([d['xyxy'] for d in viewing_list]),
                    class_id=np.array([d['class_id'] for d in viewing_list]),
                    confidence=np.array([d['confidence'] for d in viewing_list]),
                    tracker_id=np.array([d['tracker_id'] for d in viewing_list])
                )
                frame = annotator2.annotate(scene=frame, detections=detections2)


        # Draw fixations as green circles
        fixations, current_f = list_adjust(fixations, buffer, fps, frame_index)
        if not current_f == []:
            fixation = current_f[int((len(current_f) - 1)/2)]
            x = int(fixation['x'])
            y = int(fixation['y'])
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

        # Draw saccades as red lines
        saccades, current_s = list_adjust(saccades, buffer * 2, fps, frame_index)
        if len(current_s) >= 2:
            m1 = int((len(current_s))/2)-1
            m2 = int((len(current_s))/2)
            x1, y1 = int(current_s[m1]['x']), int(current_s[m1]['y'])
            x2, y2 = int(current_s[m2]['x']), int(current_s[m2]['y'])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw smooth pursuits as blue lines
        smooth_pursuits, current_p = list_adjust(smooth_pursuits, buffer * 2, fps, frame_index)
        if len(current_p) >= 2:
            m1 = int((len(current_p)) / 2) - 1
            m2 = int((len(current_p)) / 2)
            x1, y1 = int(current_p[m1]['x']), int(current_p[m1]['y'])
            x2, y2 = int(current_p[m2]['x']), int(current_p[m2]['y'])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if current_f == [] and current_s == [] and current_p == []:
            # Draw fixations from 'other' as black circles
            other, current_o = list_adjust(other, buffer, fps, frame_index)
            if not current_o == []:
                o = current_o[int((len(current_f) - 1) / 2)]
                x = int(o['x'])
                y = int(o['y'])
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    logging.info("Visualization video generated successfully.")



def main():

    # Save unprocessed gaze data to CSV
    # pd.DataFrame(gaze_data).to_csv(unprocessed_gaze_csv_path, index=False)
    logging.info("Beginning with gaze CSV")
    # Check if the detection CSV exists; if not, run object detection
    if not os.path.exists(object_csv_path) or not os.path.exists(detection_video_path):
        logging.info(f"Detection data or video not found. Running object detection.")
        detection_data = run_object_detection(original_video_path, object_csv_path, detection_video_path)
    else:
        logging.info(f"Loading existing detection data and video.")
        detection_data = pd.read_csv(object_csv_path)

    # Process gaze data to classify fixations, saccades, and smooth pursuit
    fixations, saccades, smooth_pursuits, other, metrics = process_gaze_data(unprocessed_gaze_csv_path,
                                                                             processed_gaze_csv_path, gaze)

    # Generate visualization video using the detection video as the base
    generate_visualization(fixations, saccades, smooth_pursuits, other, detection_data, original_video_path,
                           output_video_path)

    # Return the metrics as JSON
    if not testing:
        return jsonify({"status": "success", "metrics": metrics, "video_url": output_video_path})

if __name__ == "__main__":
    if not testing:
        app.run(debug=True)
    else:
        result = main()
        logging.info("Done")


