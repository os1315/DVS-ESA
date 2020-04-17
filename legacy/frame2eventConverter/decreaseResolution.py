import cv2
import numpy as np

# LOAD VIDEO

cap = cv2.VideoCapture('WZ.mp4')

fx = 2
fy = 2

frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / fx)
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / fy)

video = np.empty((frameCount, int(frameHeight/2), int((frameWidth-1)/2)), np.dtype('uint8'))

fc = 0
ret = True

while fc < frameCount and ret:
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = gray_frame.reshape(frameHeight, fy, frameWidth, fx)
    gray_frame = gray_frame.mean(axis=(1, 3))

    gray_frame = gray_frame[:, 1:frameWidth]
    gray_frame = gray_frame.reshape(int(frameHeight/2), 2, int((frameWidth-1)/2), 2)
    video[fc] = gray_frame.mean(axis=(1, 3))

    fc += 1

cap.release()


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 169, (213, 120), False)

cv2.namedWindow('video')
for n in range(frameCount):
    cv2.imshow('video', video[n])
    out.write(video[n])
    cv2.waitKey(20)
