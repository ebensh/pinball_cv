#!/usr/bin/python

import argparse
from collections import deque
import cv2
import itertools
import numpy as np

import common

def main():
  cv2.namedWindow('combined', cv2.WINDOW_NORMAL)
  cv2.createTrackbar('hue_min', 'combined', 0, 255, lambda x: 1)
  cv2.createTrackbar('hue_max', 'combined', 0, 255, lambda x: 1)
  cv2.createTrackbar('saturation_min', 'combined', 0, 255, lambda x: 1)
  cv2.createTrackbar('saturation_max', 'combined', 0, 255, lambda x: 1)
  cv2.createTrackbar('value_min', 'combined', 0, 255, lambda x: 1)
  cv2.createTrackbar('value_max', 'combined', 0, 255, lambda x: 1)

  while True:
    cap = cv2.VideoCapture(args.infile)
    while cap.isOpened():
      grabbed, raw_frame = cap.read()
      if not grabbed: break
      frame_hsv = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)

      hue_min = cv2.getTrackbarPos('hue_min','combined')
      hue_max = cv2.getTrackbarPos('hue_max','combined')
      saturation_min = cv2.getTrackbarPos('saturation_min','combined')
      saturation_max = cv2.getTrackbarPos('saturation_max','combined')
      value_min = cv2.getTrackbarPos('value_min','combined')
      value_max = cv2.getTrackbarPos('value_max','combined')
    
      h,s,v = cv2.split(frame_hsv)
      h_mask = cv2.inRange(h, hue_min, hue_max)
      s_mask = cv2.inRange(s, saturation_min, saturation_max)
      v_mask = cv2.inRange(v, value_min, value_max)
      
      combined_image = cv2.vconcat([
        cv2.hconcat([h, s, v]),
        cv2.hconcat([cv2.bitwise_and(h, h, mask=h_mask),
                     cv2.bitwise_and(s, s, mask=s_mask),
                     cv2.bitwise_and(v, v, mask=v_mask)])])
    
      common.display_image(combined_image, 'combined')
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Track a pinball.')
  parser.add_argument('--infile', required=True, type=str, help='Input video file path.')
  args = parser.parse_args()
  main()
