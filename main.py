import cv2
from datetime import datetime, timedelta
from ultralytics import YOLO
import requests
import time


def send_face_to_api(face_image, api_url="https://10.130.64.245:7059/Image"):
    try:
        _, buffer = cv2.imencode('.png', face_image)
        files = {'file': ('face.png', buffer.tobytes(), 'image/png')}
        
        headers = {
            'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1lIjoidGVzdCIsImh0dHA6Ly9zY2hlbWFzLm1pY3Jvc29mdC5jb20vd3MvMjAwOC8wNi9pZGVudGl0eS9jbGFpbXMvcm9sZSI6IkFkbWluIiwiZXhwIjoxNzU4MjczNTE4LCJpc3MiOiJwaGl6YXBpIiwiYXVkIjoicGhpemFwaV91c2VycyJ9.mzjsrN_52c7rGLAqKcl8pGGSHERi5keSGIDzuC9iVNI'
        }

        data = {
            'timestamp': datetime.now().isoformat(),
            'source': 'webcam'
        }
        
        response = requests.post(api_url, files=files, data=data, headers=headers, timeout=10, verify=False)
        if response.status_code == 200:
            print(f"Successfully sent face to API: {response.json()}")
        else:
            print(f"API error: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Failed to send face to API: {e}")
    except Exception as e:
        print(f"Error processing face image: {e}")

def main():
    cap = cv2.VideoCapture(0)
    time.sleep(5)    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    model = YOLO('runs/detect/train/weights/best.pt')
    
    last_detection_time = datetime.now() - timedelta(minutes=1)
    detection_interval = timedelta(minutes=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        current_time = datetime.now()
        should_detect = current_time - last_detection_time >= detection_interval
        
        if should_detect:
            results = model(frame[..., ::-1], verbose=False)
            last_detection_time = current_time
            
            if len(results[0].boxes) > 0:
                x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
                
                width = x2 - x1
                height = y2 - y1
                zoom_factor = 0.2
                
                new_x1 = max(0, x1 - int(width * zoom_factor))
                new_y1 = max(0, y1 - int(height * zoom_factor))
                new_x2 = min(frame.shape[1], x2 + int(width * zoom_factor))
                new_y2 = min(frame.shape[0], y2 + int(height * zoom_factor))
                
                zoomed_face = frame[new_y1:new_y2, new_x1:new_x2]
                
                if zoomed_face.size > 0: 
                    zoomed_face_resized = cv2.resize(zoomed_face, (224, 224))
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"detected_face_{timestamp}.jpg", zoomed_face_resized)
                    
                    send_face_to_api(zoomed_face_resized)
                    
                    print(f"Face detected and sent to API at {current_time}")
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = "Unknown Face" if int(box.cls[0]) == 0 else model.names[int(box.cls[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, "PhizRecon", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, now, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        time_until_next = detection_interval - (current_time - last_detection_time)
        if time_until_next.total_seconds() > 0:
            seconds_left = int(time_until_next.total_seconds())
            cv2.putText(frame, f"Next detection in: {seconds_left}s", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Webcam", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()