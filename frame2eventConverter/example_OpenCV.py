import cv2
import numpy as np

import convInterpolate as CI

# LOAD VIDEO
cap = cv2.VideoCapture('example_video.avi')

frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))

fc = 0
ret = True

while fc < frameCount and ret:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    video[fc] = gray_frame
    fc += 1

cap.release()

# CONVERT
converter = CI.convInterpolate(15, 1, 0.5, video[0])
EVENT_LIST = []
event_video = np.empty((frameCount-1, int(frameHeight), int(frameWidth)), np.dtype('int8'))



for n in range(1, frameCount):
    new_events, event_frame = converter.update(video[n])
    event_video[n-1] = event_frame * 127
    EVENT_LIST = EVENT_LIST + new_events
    print(f'Converted: {n}/{frameCount}')

# SAVE EVENTS DATA TO CSV
EVENT_FILE = open("eventlist.csv", 'w')
EVENT_FILE.write("x; y; t; p;\n")
for event in EVENT_LIST:
    EVENT_FILE.write(
        str(event[0]) + "; " + str(event[1]) + "; " + "{:.0f}".format(event[2]) + "; " + str(
            event[3]) + ";\n")
EVENT_FILE.close()

# PLAY VIDEO using OpenCV
# These 4 lines only create a larger copy of the video for viewing
resized = np.repeat(video, 2, axis=1)
resized = np.repeat(resized, 2, axis=2)
resized_eve = np.repeat(event_video, 2, axis=1)
resized_eve = np.repeat(resized_eve, 2, axis=2)

cv2.namedWindow('video')
cv2.namedWindow('event video')
while True:
    for n in range(frameCount-1):
        cv2.imshow('video', resized[n+1])
        cv2.imshow('event video', resized_eve[n])
        cv2.waitKey(100)



