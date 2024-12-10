import cv2
from concurrent.futures import ThreadPoolExecutor
import time

def check_camera(index):
    """카메라 인덱스를 확인하여 사용할 수 있는지 반환"""
    cap = cv2.VideoCapture(index)
    time.sleep(1)  # 시뮬레이션용 지연 시간
    if cap.isOpened():
        cap.release()
        return index
    return None

def list_cameras():
    """사용 가능한 카메라 리스트 반환"""
    available_cameras = []
    max_camera_index = 10  # 최대 확인할 카메라 인덱스
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(check_camera, range(max_camera_index))
    for result in results:
        if result is not None:
            available_cameras.append(result)
    return available_cameras

class Camera:
    def __init__(self, index=0):
        """카메라 초기화"""
        self.camera = cv2.VideoCapture(index)
        if not self.camera.isOpened():
            raise RuntimeError(f"카메라 {index}를 열 수 없습니다.")
        self.index = index
        print(f"카메라 {index} 연결 성공!")

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

    def get_frame(self):
        """현재 카메라 프레임 반환"""
        success, frame = self.camera.read()
        if not success:
            raise RuntimeError("프레임을 읽을 수 없습니다.")
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
