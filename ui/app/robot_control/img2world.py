import os
import numpy as np
import cv2

class CalibBoard():
    def __init__(self):
        self.checkerboard=(4,3)
        self.mtx = np.array([[638.82090741,  0.0, 320.40519741],
                            [0.0,    639.05019247, 240.20585631],
                            [0.0,    0.0,               1.0]])
            
        self.dist=np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        self.z_cam=0.418
        self.T_rb = np.array([[ 0, 1,  0,  -0.17],
                            [   1, 0,  0,  0.06],
                            [   0, 0, -1,    0.0],
                            [   0, 0,  0,   1.0]])
            
        self.square_size=0.03
        self.axis = np.float32([[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]]).reshape(-1, 3)
        self.image="./ui/app/robot_control/captured_image.jpg"

    def draw(self, img, corners, imgpts):  ### Rearrange from BGR to RGB order ###
        c = corners[0].ravel()
        corner = (int(c[0]), int(c[1]))
        img = cv2.line(img, corner, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (0, 0, 255), 3)
        img = cv2.line(img, corner, (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0, 255, 0), 3)
        img = cv2.line(img, corner, (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (255, 0, 0), 3)

        return img    

    def calculate_T_rc(self):

        # Prepare object points for the checkerboard
        objp = np.zeros((self.checkerboard[0] * self.checkerboard[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard[0], 0:self.checkerboard[1]].T.reshape(-1, 2)
        objp *= self.square_size

        # Load image
        img = cv2.imread(self.image)
        if img is None:
            raise ValueError(f"Image not found at {self.image}")
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)
        if not ret:
            raise ValueError("Checkerboard corners could not be found in the image.")

        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Solve PnP to get rotation and translation vectors
        cv2.drawChessboardCorners(img, self.checkerboard, corners2, ret)
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, self.mtx, self.dist)

        # Calculate Transformation Matrix (Camera to Checkerboard)
        R, _ = cv2.Rodrigues(rvecs)  # Convert rotation vector to rotation matrix
        T_cb = np.hstack((R, tvecs))  # Combine R and t
        T_cb = np.vstack((T_cb, [0, 0, 0, 1]))  # Make it a 4x4 homogeneous matrix
        print("Transformation Matrix (Camera to Board):\n", T_cb)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(self.axis, rvecs, tvecs, self.mtx, self.dist)
        img = self.draw(img, corners2, imgpts)
        #cv2.circle(img, (pixel_x, pixel_y), 5, (0,0,255), 3)

        #cv2.imshow('img', img)
        #cv2.waitKey(0)

        # Calculate Transformation Matrix (Robot to Camera)
        T_rc = self.T_rb @ np.linalg.inv(T_cb)
        print("Transformation Matrix (Robot to Camera):\n", T_rc)
    
        return T_rc
    
    def find_available_devices(max_devices=3):
        available_devices = []
 
        # cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        cv2.setLogLevel(0)
        for device_id in range(max_devices):
            cap = cv2.VideoCapture(device_id)
            print('----------------')
            if cap.isOpened():  # 장치가 열렸다면, 사용 가능
                available_devices.append(device_id)
                cap.release()  # 장치를 닫아줍니다.
 
        # cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        cv2.setLogLevel(3)
 
        return available_devices

    def take_picture(self):

        camera_id=self.find_available_devices()

        for cam in camera_id:

            # 카메라 초기화
            cap = cv2.VideoCapture(cam)

            if not cap.isOpened():
                print("카메라를 열 수 없습니다.")
            else:
                print("Open camera")
                break

        # 카메라 해상도 설정 (1280x720 예시)
        desired_width = 640  # 원하는 해상도의 가로 크기
        desired_height = 480 # 원하는 해상도의 세로 크기
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

        # 설정된 해상도 확인
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"카메라 해상도 설정: {int(actual_width)}x{int(actual_height)}")

        print("사진을 찍으려면 's' 키를 누르고, 종료하려면 'q' 키를 누르세요.")

        # 프레임 캡처 루프
        image_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("프레임을 읽을 수 없습니다. 종료합니다.")
                    break

                # 프레임 표시
                cv2.imshow("USB Camera", frame)

                # 키 입력 대기
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):  # 's' 키를 누르면 사진 저장
                    image_count += 1
                    image_path = "captured_image.jpg"
                    cv2.imwrite(image_path, frame)
                    print(f"사진이 저장되었습니다: {image_path}")

                elif key == ord('q'):  # 'q' 키를 누르면 종료
                    print("종료합니다.")
                    break
        finally:
            # 자원 해제
            cap.release()
            cv2.destroyAllWindows()

    def init_board(self):
        self.take_picture()
        T_rc=self.calculate_T_rc()
        self.T_rc=T_rc

    
    def cam2robot(self,pixel_x, pixel_y):
        mtx=self.mtx
        #T_rc = self.calculate_T_rc(pixel_x, pixel_y)
        x_cam = (pixel_x - mtx[0, 2]) * self.z_cam / mtx[0, 0]
        y_cam = (pixel_y - mtx[1, 2]) * self.z_cam / mtx[1, 1]
        point_cam = np.array([x_cam, y_cam, self.z_cam, 1])
        point_robot = self.T_rc @ point_cam

        return point_robot