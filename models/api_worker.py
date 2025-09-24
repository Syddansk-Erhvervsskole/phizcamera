import requests
from requests.exceptions import RequestException
from queue import Queue
from models.face_detection_engine import FaceDetectionEngine

class APIWorker:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.queue = Queue()
        self.face_detection_engine = FaceDetectionEngine(api_url=base_url)

    def fetch_data(self, endpoint):
        url = f"{self.base_url}/{endpoint}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            return None
        
    def api_worker(self, endpoint):
        print(f"API Worker started - will send faces to {endpoint}")
        while True:
            try:
                face_image = self.queue.get()
                if face_image is None:
                    print("API Worker received shutdown signal")
                    break
                    
                result = self.face_detection_engine.send_face_to_api(face_image)
                if result:
                    print(f"Face processed successfully by API")
                else:
                    print("Failed to process face via API")
                    
                self.queue.task_done()
            except Exception as e:
                print(f"API worker error: {e}")
        print("API Worker stopped")
