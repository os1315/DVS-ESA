import cv2
import numpy as np
from matplotlib.animation import FuncAnimation

import convInterpolate as CI

import matplotlib.pyplot as plt

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
event_list = []
event_video = np.empty((frameCount-1, int(frameHeight), int(frameWidth)), np.dtype('int8'))



for n in range(1, frameCount):
    new_events, event_frame = converter.update(video[n])
    event_video[n-1] = event_frame * 127
    event_list = event_list + new_events
    print(f'Converted: {n}/{frameCount}')

# event_video = event_video * 127 + 127

# PLAY VIDEO: OpenCV
# This 4 lines only create a larger copy of the video for viewing
resized = np.repeat(video, 2, axis=1)
resized = np.repeat(resized, 2, axis=2)
resized_eve = np.repeat(event_video, 2, axis=1)
resized_eve = np.repeat(resized_eve, 2, axis=2)

# cv2.namedWindow('video')
# cv2.namedWindow('event video')
# while True:
#     for n in range(frameCount-1):
#         cv2.imshow('video', resized[n+1])
#         cv2.imshow('event video', resized_eve[n])
#         cv2.waitKey(100)


# PLAY VIDEO: Matplotlib
both_images = np.empty((frameCount-1, frameHeight*2, frameWidth*4))
both_images[:, :, :frameWidth*2] = resized_eve
both_images[:, :, frameWidth*2:] = resized[1:]


fig, ax = plt.subplots(figsize=(5, 8))

def update(i):
    ax.imshow(both_images[i], cmap='gray')


anim = FuncAnimation(fig, update, frames=range(frameCount), interval=100)
anim.save('example_gif.gif', dpi=80, writer='imagemagick')
plt.close()


