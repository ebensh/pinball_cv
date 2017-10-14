#!/usr/bin/python

import argparse
from collections import deque
import cv2
import itertools
import numpy as np

import common

def main():
  cap = cv2.VideoCapture(args.infile)
  
  FRAME_BUFFER_SIZE = 10
  BLEND_ALPHA = 0.33
  frames_past = deque([], maxlen=FRAME_BUFFER_SIZE)
  frames_future = deque([], maxlen=FRAME_BUFFER_SIZE)

  # Loop until the frame buffers are full at the start.
  while cap.isOpened() and len(frames_past) < FRAME_BUFFER_SIZE:
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    frames_past.append((raw_frame, cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)))
  while cap.isOpened() and len(frames_future) < FRAME_BUFFER_SIZE:
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    frames_future.append((raw_frame, cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)))

  cv2.namedWindow('combined', cv2.WINDOW_NORMAL)
  cv2.createTrackbar('saturation_min', 'combined', 0, 255, lambda x: 1)
  cv2.createTrackbar('saturation_max', 'combined', 0, 255, lambda x: 1)

  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    current_frame, current_frame_gray = frames_future.popleft()
    frames_future.append((raw_frame, cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)))

    frame_printer = common.FramePrinter()


    saturation_min = cv2.getTrackbarPos('saturation_min','combined')
    saturation_max = cv2.getTrackbarPos('saturation_max','combined')
    frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, (0, saturation_min, 0), (255, saturation_max, 255))
    frame_printer.add_image(current_frame, 'current')
    frame_printer.add_image(mask, 'mask')
    frame_printer.add_image(cv2.bitwise_and(current_frame, current_frame, mask=mask), 'mask applied')

    frames_past.append((current_frame, current_frame_gray))

    combined_image = frame_printer.get_combined_image()
    common.display_image(combined_image, 'combined')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Track a pinball.')
  parser.add_argument('--infile', required=True, type=str, help='Input video file path.')
  parser.add_argument('--display_all_images', default=False, type=bool,
                      help='Display all (debug) images.')
  args = parser.parse_args()
  main()
