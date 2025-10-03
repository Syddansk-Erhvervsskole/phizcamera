from models.api_worker import APIWorker
from models.face_detection_engine import FaceDetectionEngine
from models.authentication_engine import AuthenticationEngine
from models.communication_engine import CommunicationEngine
import threading
import cv2
from datetime import datetime
from ultralytics import YOLO
import argparse
import socket
import platform


def main():
    parser = argparse.ArgumentParser(description="PhizRecon Camera Client")
    parser.add_argument("--comm-server", help="Communication server URL (e.g., ws://localhost:3000)")
    parser.add_argument("--camera-id", type=int, help="Camera ID for server identification")
    parser.add_argument("--camera-name", help="Camera name for identification")
    parser.add_argument("--device-index", type=int, default=0, help="Camera device index")
    args = parser.parse_args()
    
    base_url = "https://10.130.64.245:7059"
    api_endpoint = f"{base_url}/Image"
    model_path = 'ml_model/best.pt'

    api_worker = APIWorker(base_url=base_url)
    face_detection_engine = FaceDetectionEngine(api_url=base_url)
    
    comm_engine = None
    if args.comm_server:
        comm_engine = CommunicationEngine(
            server_url=args.comm_server,
            camera_id=args.camera_id,
            camera_name=args.camera_name,
            device_index=args.device_index
        )
        print(f"Communication server configured: {args.comm_server}")
        comm_engine.start_communication_thread()
    else:
        print("No communication server specified - running in standalone mode")
    
    api_thread = threading.Thread(target=api_worker.api_worker, args=(api_endpoint,), daemon=True)
    api_thread.start()
    
    device_idx = args.device_index if args.device_index is not None else 0
    cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        print(f"Error: Could not open webcam device {device_idx}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)            
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     
    
    model = YOLO(model_path)
    model.to('cpu')
    
    frame_skip = 3 
    frame_count = 0
    last_results = None
    display_frame_skip = 2
    display_count = 0
    comm_frame_skip = 5
    
    print("Starting optimized face detection for Raspberry Pi...")
    if comm_engine:
        print(f"Communication enabled - will stream to {args.comm_server}")

    last_image_time = datetime.min
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        frame_count += 1
        display_count += 1

        if comm_engine and frame_count % comm_frame_skip == 0:
            comm_engine.queue_frame(frame)

        if frame_count % frame_skip == 0:
            process_frame = cv2.resize(frame, (320, 240))
            results = model(process_frame, verbose=False, imgsz=320)
            last_results = results

            now = datetime.now()
            if len(results[0].boxes) > 0 and (now - last_image_time).total_seconds() >= 5:
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 240
                x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                width = x2 - x1
                height = y2 - y1
                zoom_factor = 0.2
                new_x1 = max(0, x1 - int(width * zoom_factor))
                new_y1 = max(0, y1 - int(height * zoom_factor))
                new_x2 = min(frame.shape[1], x2 + int(width * zoom_factor))
                new_y2 = min(frame.shape[0], y2 + int(height * zoom_factor))
                
                zoomed_face = frame[new_y1:new_y2, new_x1:new_x2]
                if zoomed_face.size > 0:
                    zoomed_face_resized = cv2.resize(zoomed_face, (160, 160))
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    
                    api_worker.queue.put(zoomed_face_resized)
                    print(f"Face detected - queued for API authentication at {timestamp}")
                    last_image_time = now
        
        if last_results is not None and len(last_results[0].boxes) > 0:
            for box in last_results[0].boxes:
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 240
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                conf = float(box.conf[0])
                label = "Face" if int(box.cls[0]) == 0 else model.names[int(box.cls[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if display_count % display_frame_skip == 0:
            now = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, "PhizRecon", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, now, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if comm_engine:
                status_text = f"Comm: {comm_engine.camera_name}"
                cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Webcam", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    print("Shutting down...")
    
    api_worker.queue.put(None)
    
    if comm_engine:
        comm_engine.stop()
        print("Communication engine stopped")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Application stopped.")

if __name__ == "__main__":
    main()
