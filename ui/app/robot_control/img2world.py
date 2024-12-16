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
        self.image="captured_image.jpg"

    def draw(self, img, corners, imgpts):  ### Rearrange from BGR to RGB order ###
        c = corners[0].ravel()
        corner = (int(c[0]), int(c[1]))
        img = cv2.line(img, corner, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (0, 0, 255), 3)
        img = cv2.line(img, corner, (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0, 255, 0), 3)
        img = cv2.line(img, corner, (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (255, 0, 0), 3)

        return img    

    def calculate_T_rc(self, pixel_x, pixel_y):

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
        cv2.circle(img, (pixel_x, pixel_y), 5, (0,0,255), 3)

        cv2.imshow('img', img)
        cv2.waitKey(0)

        # Calculate Transformation Matrix (Robot to Camera)
        T_rc = self.T_rb @ np.linalg.inv(T_cb)
        print("Transformation Matrix (Robot to Camera):\n", T_rc)
    
        return T_rc
    
    
    def cam2robot(self,pixel_x, pixel_y):
        mtx=self.mtx
        T_rc = self.calculate_T_rc(pixel_x, pixel_y)
        x_cam = (pixel_x - mtx[0, 2]) * self.z_cam / mtx[0, 0]
        y_cam = (pixel_y - mtx[1, 2]) * self.z_cam / mtx[1, 1]
        point_cam = np.array([x_cam, y_cam, self.z_cam, 1])
        point_robot = T_rc @ point_cam

        return point_robot
    
# board=CalibBoard()
# print(board.cam2robot(389,170))