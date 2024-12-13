import cv2
import numpy as np
import glob

images = "./test.jpg"  # 폴더, 파일이름, 확장자 변경 가능

mtx = np.array([[607.0,  0.0, 320.0],
                [0.0,  607.0, 240.0],
                [0.0,    0.0,   1.0]])
dist = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])  # distortion matrix


T_rb = np.array([[ -1, 0,  0,  0.0],
                [   0, 1,  0,  0.0],
                [   0, 0, -1,  0.0],
                [   0, 0,  0,   1.0]]) # 기준점: 체커보드 오른쪽 아래 좌표


CHECKERBOARD = (5, 4) # # of chessboard corners
square_size = 0.02  # 체스보드 한 정사각형의 크기 (m 단위 -> 2cm)
pixel_x, pixel_y = 320, 240 # target pixels(example for center)
z_cam = 0.25                # z-coord length between camera and target object(m)


axis = np.float32([[0.03, 0, 0], [0, 0.03, 0], [0, 0, -0.03]]).reshape(-1, 3)  # for axis

def draw(img, corners, imgpts):  ### Rearrange from BGR to RGB order ###
    c = corners[0].ravel()
    corner = (int(c[0]), int(c[1]))
    img = cv2.line(img, corner, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (0, 0, 255), 3)
    img = cv2.line(img, corner, (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0, 255, 0), 3)
    img = cv2.line(img, corner, (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (255, 0, 0), 3)
    return img


def calculate_T_rc(images, checkerboard_size, square_size, mtx, dist, T_rb):
    # Prepare object points for the checkerboard
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Load image
    img = cv2.imread(images)
    if img is None:
        raise ValueError(f"Image not found at {images}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    if not ret:
        raise ValueError("Checkerboard corners could not be found in the image.")

    # Refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Solve PnP to get rotation and translation vectors
    cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

    # Calculate Transformation Matrix (Camera to Checkerboard)
    R, _ = cv2.Rodrigues(rvecs)  # Convert rotation vector to rotation matrix
    T_cb = np.hstack((R, tvecs))  # Combine R and t
    T_cb = np.vstack((T_cb, [0, 0, 0, 1]))  # Make it a 4x4 homogeneous matrix
    print("Transformation Matrix (Camera to Board):\n", T_cb)

    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    img = draw(img, corners2, imgpts)

    cv2.imshow('img', img)
    cv2.waitKey(0)

    # Calculate Transformation Matrix (Robot to Camera)
    T_rc = T_rb @ np.linalg.inv(T_cb)
    print("Transformation Matrix (Robot to Camera):\n", T_rc)

    return T_rc




def main():
    T_rc = calculate_T_rc(images, CHECKERBOARD, square_size, mtx, dist, T_rb)
    x_cam = (pixel_x - mtx[0, 2]) * z_cam / mtx[0, 0]
    y_cam = (pixel_y - mtx[1, 2]) * z_cam / mtx[1, 1]
    point_cam = np.array([x_cam, y_cam, z_cam, 1])
    point_robot = T_rc @ point_cam

    print("target position in robot coordinate", point_robot[:3])



# Results of test.jpg file (almost correct), unit: meter
# Transformation Matrix (Camera to Board):
#  [[-0.99041472  0.08780456  0.1066257   0.04587595]
#  [-0.06151014 -0.97155021  0.22870655  0.02102377]
#  [ 0.1236737   0.21995577  0.96763851  0.31470227]
#  [ 0.          0.          0.          1.        ]]
# Transformation Matrix (Robot to Camera):
#  [[ 0.99041472  0.06151014 -0.1236737  -0.00780899]
#  [ 0.08780456 -0.97155021  0.21995577 -0.05282305]
#  [-0.1066257  -0.22870655 -0.96763851  0.31421786]
#  [ 0.          0.          0.          1.        ]]
# target position in robot coordinate [-0.03872742  0.00216589  0.07230824]