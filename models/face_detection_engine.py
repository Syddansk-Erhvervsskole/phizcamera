import cv2
from datetime import datetime, timedelta
import requests
import numpy as np
from models.authentication_engine import AuthenticationEngine

class FaceDetectionEngine:
    def __init__(self, api_url):
        self.api_url = api_url
        self.auth_engine = AuthenticationEngine(auth_url="https://10.130.64.245:7059/User/Login")

    def send_face_to_api(self, face_image):
        try:
            token = self.auth_engine.call_login_token("test", "test")
            if not token:
                print("Failed to authenticate - cannot send face to API")
                return None
                
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', face_image, encode_param)
            files = {'file': ('face.jpg', buffer.tobytes(), 'image/jpeg')}
            headers = {
                'Authorization': f'Bearer {token}'
            }

            data = {
                'timestamp': datetime.now().isoformat(),
                'source': 'webcam'
            }

            api_url = f"{self.api_url}/Image"
            response = requests.post(api_url, files=files, data=data, headers=headers, verify=False, timeout=10)
            
            if response.status_code == 200:
                print(f"Successfully sent face to API")
                return response.json()
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Network error sending image to API: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during image processing: {e}")
            return None
