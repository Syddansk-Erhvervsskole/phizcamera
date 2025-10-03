import asyncio
import websockets
import json
import base64
import time
import socket
import platform
import cv2
from threading import Thread
import queue


class CommunicationEngine:
    def __init__(self, server_url, camera_id=None, camera_name=None, device_index=0):
        self.server_url = server_url
        self.device_index = device_index
        self.camera_id = camera_id or self._generate_camera_id()
        self.camera_name = camera_name or self._generate_camera_name()
        self.ws = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10) 
        self.loop = None
        self.thread = None
        
    def _generate_camera_id(self):
        hostname = socket.gethostname()
        return abs(hash(hostname)) % 1000
    
    def _generate_camera_name(self):
        hostname = socket.gethostname()
        system = platform.system()
        return f"{hostname}_{system}_Cam{self.camera_id}"
    
    async def connect_server(self):
        try:
            print(f"Connecting to communication server: {self.server_url}")
            self.ws = await websockets.connect(self.server_url)
            
            await self.ws.send(json.dumps({
                'type': 'identify',
                'role': 'producer', 
                'client_type': 'camera',
                'camera_id': str(self.camera_id),
                'camera_name': self.camera_name
            }))
            
            print(f"✓ Connected to server as {self.camera_name} (ID: {self.camera_id})")
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to communication server: {e}")
            return False
    
    def _is_connected(self):
        try:
            return self.ws is not None and hasattr(self.ws, 'open') and self.ws.open
        except:
            return False
    
    async def send_frame(self, frame):
        try:
            if not self._is_connected():
                return False
                
            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480))
            
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            await self.ws.send(json.dumps({
                'type': 'frame',
                'frameData': frame_b64,
                'camera_id': str(self.camera_id),
                'timestamp': time.time()
            }))
            
            return True
            
        except websockets.exceptions.ConnectionClosed:
            print("Communication server disconnected")
            return False
        except Exception as e:
            print(f"Error sending frame to communication server: {e}")
            return False
    
    async def handle_server_messages(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get('type')
                
                if msg_type == 'identified':
                    print(f"Server acknowledged: {data.get('message', 'Camera identified')}")
                elif msg_type == 'command':
                    print(f"Received command from server: {data.get('command')}")
                else:
                    print(f"Received message from server: {msg_type}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("Communication server closed connection")
        except Exception as e:
            print(f"Error handling server messages: {e}")
    
    async def frame_sender_worker(self):
        frame_count = 0
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    success = await self.send_frame(frame)
                    
                    if success:
                        frame_count += 1
                        if frame_count % 100 == 0:
                            print(f"Sent {frame_count} frames to communication server")
                    else:
                        if not await self.connect_server():
                            print("Failed to reconnect to communication server")
                            await asyncio.sleep(5) 
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Error in frame sender worker: {e}")
                await asyncio.sleep(1)
    
    async def async_main_loop(self):
        try:
            if not await self.connect_server():
                return False
            
            self.running = True
            
            sender_task = asyncio.create_task(self.frame_sender_worker())
            receiver_task = asyncio.create_task(self.handle_server_messages())
            
            done, pending = await asyncio.wait(
                [sender_task, receiver_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in pending:
                task.cancel()
            
            return True
            
        except Exception as e:
            print(f"Error in communication engine main loop: {e}")
            return False
        finally:
            await self.cleanup()
    
    def start_communication_thread(self):
        if self.thread and self.thread.is_alive():
            print("Communication thread already running")
            return
            
        self.thread = Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        print("Communication engine thread started")
    
    def _run_async_loop(self):
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.async_main_loop())
        except Exception as e:
            print(f"Error in communication thread: {e}")
        finally:
            if self.loop:
                self.loop.close()
    
    def queue_frame(self, frame):
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put_nowait(frame.copy()) 
            return True
        except Exception as e:
            print(f"Error queuing frame: {e}")
            return False
    
    async def cleanup(self):
        self.running = False
        
        if self.ws:
            await self.ws.close()
            
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        print("✓ Communication engine cleanup complete")
    
    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        print("Communication engine stopped")
