import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# Load the YOLOv8 Nano model
try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# DBSCAN Configuration
EPS_DISTANCE = 100 
MIN_SAMPLES = 3

def get_group_density_info(member_count):
    """Returns (Label, Color) for a specific group based on size."""
    if member_count >= 8:
        return "HIGH DENSITY", (0, 0, 255)    # Red
    elif member_count >= 5:
        return "MEDIUM DENSITY", (0, 255, 255) # Yellow
    return "LOW DENSITY", (0, 255, 0)         # Green


# Initialize video capture from the default camera
video="http://10.77.67.61:5000"
cap = cv2.VideoCapture(video)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Camera started. Point your camera at the crowd/video. Press 'q' to exit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Perform detection
    results = model.predict(frame, classes=[0], conf=0.5, verbose=False)
    
    detections = results[0].boxes
    person_centers = []
    bboxes = []

    for box in detections:
        # Cast coordinates to standard int
        coords = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, coords)
        bboxes.append((x1, y1, x2, y2))
        
        # Center point for DBSCAN
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        person_centers.append([cx, cy])

    if len(person_centers) >= MIN_SAMPLES:
        clustering = DBSCAN(eps=EPS_DISTANCE, min_samples=MIN_SAMPLES).fit(person_centers)
        cluster_labels = clustering.labels_
    else:
        cluster_labels = [-1] * len(person_centers)

    # Dictionary to store group boundaries: {cluster_id: [min_x, min_y, max_x, max_y, count]}
    group_bounds = {}

    for i, (box, label) in enumerate(zip(bboxes, cluster_labels)):
        x1, y1, x2, y2 = box
        
        if label == -1:
            # White for individual persons
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        else:
            # Generate unique color per person box within a cluster
            b = int((label * 100 + 50) % 255)
            g = int((label * 50 + 100) % 255)
            r = int((label * 150 + 150) % 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), 1)

            # Update the bounding box for the entire cluster group
            if label not in group_bounds:
                group_bounds[label] = [x1, y1, x2, y2, 1]
            else:
                group_bounds[label][0] = min(group_bounds[label][0], x1)
                group_bounds[label][1] = min(group_bounds[label][1], y1)
                group_bounds[label][2] = max(group_bounds[label][2], x2)
                group_bounds[label][3] = max(group_bounds[label][3], y2)
                group_bounds[label][4] += 1

    for label, bounds in group_bounds.items():
        gx1, gy1, gx2, gy2, count = bounds
        
        # Get density level for this specific group
        density_text, status_color = get_group_density_info(count)
        
        # Draw a thicker dashed-style boundary for the group
        cv2.rectangle(frame, (gx1 - 10, gy1 - 10), (gx2 + 10, gy2 + 10), status_color, 3)
        
        # Draw Label Plate for the group
        label_size = cv2.getTextSize(density_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (gx1 - 10, gy1 - 35), (gx1 - 10 + label_size[0] + 10, gy1 - 10), status_color, -1)
        cv2.putText(frame, density_text, (gx1 - 5, gy1 - 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Global Info Panel
    total_people = len(bboxes)
    num_clusters = len(group_bounds)
    cv2.rectangle(frame, (10, 10), (380, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"LIVE SURVEILLANCE | People: {total_people} | Groups: {num_clusters}", 
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Crowd Surveillance AI - Cluster Density", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()