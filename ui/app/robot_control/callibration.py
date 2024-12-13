import cv2

cap = cv2.VideoCapture(2)
ret, frame = cap.read()

if ret:
    cv2.imshow("Captured Frame", frame)
    cv2.imwrite("grid_image.jpg", frame)


cap.release()

pattern_size = (7, 9)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(gray.dtype)
ret, corners = cv2.findChessboardCorners(pattern_size, None)

if ret:
    cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
    cv2.imshow("Chessboard", frame)
    cv2.waitKey(0)

