import cv2
import cv2
import numpy as np


capture = cv2.VideoCapture(
    'red_car_moving.mp4')


bg_sub = cv2.createBackgroundSubtractorMOG2()

while True:

    ret, frame = capture.read()

    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    if frame is None:
        break

    fg_mask = bg_sub.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, ksize=5)

    x, y, w, h = cv2.boundingRect(fg_mask)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imshow("frame", frame)
    cv2.imshow("sub", fg_mask)

    if cv2.waitKey(1) == ord('s'):
        break

capture.release()
cv2.destroyAllWindows()
