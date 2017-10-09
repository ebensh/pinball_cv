#!/usr/bin/python

import argparse
import cv2
import numpy as np

import common

def main():
  cap = cv2.VideoCapture(args.infile)
  heat_map = cv2.imread(args.heatmap, cv2.COLOR_BGR2GRAY)
  base_frame = None

  # Note that we invert the heat_map here, so only those *cold* portions of the
  # heat map will be kept when we use it as a mask later.
  heat_map = 255 - heat_map
  
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    common.display_image(raw_frame, 'raw_frame', True)

    if base_frame is None:
      base_frame = raw_frame
      common.display_image(base_frame, 'base_frame', args.display_all_images)
      common.display_image(heat_map, 'heat_map', args.display_all_images)

    frame = cv2.absdiff(raw_frame, base_frame)
    frame = cv2.bitwise_and(frame, heat_map)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)

    common.display_image(frame, show=args.display_all_images)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Track a pinball.')
  parser.add_argument('--infile', required=True, type=str, help='Input video file path.')
  parser.add_argument('--heatmap', required=True, type=str, help='Heat map image file path.')
  parser.add_argument('--display_all_images', default=False, type=bool,
                      help='Display all (debug) images.')
  args = parser.parse_args()
  main()
