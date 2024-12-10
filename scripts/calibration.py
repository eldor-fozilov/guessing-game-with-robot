import cv2
import numpy as np


file_name = 'image.png'
# mtx = np.array([[337.5550842285156,                   0, 322.2391662597656],
#                 [                0,  337.5550842285156, 179.506591796875],
#                 [                0,                   0,                 1]])  # Zed left camera & 640x360 image
mtx = np.array([[ 908.9029541015625,                0.0,  639.3457641601562],
                [               0.0,   908.635009765625, 360.27484130859375],
                [               0.0,                0.0,                1.0]])  # realsense camera & 1280x720 image
dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

T_vb = np.array([[ 0, 0, 1, 184.0],
                 [ 0,-1, 0,  20.0],
                 [ 1, 0, 0,  25.5],
                 [ 0, 0, 0,   1.0]]) # transformation matrix from vehicle to checkerboard


CHECKERBOARD = (7,9) # 체커보드 행과 열당 내부 코너 수
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# 3D 점의 세계 좌표 정의
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * 5  # 체커보드의 각 사각형 길이
objpoints = []
imgpoints = [] 
axis = np.float32([[10,0,0], [0,10,0], [0,0,5]]).reshape(-1,3)


def draw(img, corners, imgpts):  # bgr -> rgb 순서로 바꿈
    c = corners[0].ravel()
    corner = (int(c[0]), int(c[1]))
    cv2.line(img, corner, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (0,0,255), 2)
    cv2.line(img, corner, (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0,255,0), 2)
    cv2.line(img, corner, (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (255,0,0), 2)
    return img


img = cv2.imread(file_name)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 이미지에서 원하는 개수의 코너가 발견되면 ret = true
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
# 원하는 개수의 코너가 감지되면 픽셀 좌표 미세조정 -> 체커보드 이미지 표시
if ret == True:
    objpoints.append(objp)
    # 주어진 2D 점에 대한 픽셀 좌표 미세조정
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)

    img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    _, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
    
    # world axis project to image plane
    imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    img = draw(img, corners2, imgpts)

    cv2.imshow('img',img)
    cv2.imwrite("modified_image.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# print("rvecs : \n", rvecs) # rotation
# print("tvecs : \n", tvecs) # translation
rot_matrix, _ = cv2.Rodrigues(rvecs)
T_cb = np.vstack([np.hstack([rot_matrix, tvecs]), [0,0,0,1]])
T_vc = T_vb @ np.linalg.inv(T_cb)
print("T_vehicle-cam: \n", T_vc)