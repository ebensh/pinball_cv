#!/usr/bin/python

import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Track a pinball.')
parser.add_argument('--infile', required=True, type=str, help='Input video file path.')
parser.add_argument('--heatmap', required=True, type=str, help='Heat map image file path.')
parser.add_argument('--debug', default=False, type=bool, help='Display processing steps.')
args = parser.parse_args()

def main():
  cap = cv2.VideoCapture('intro_removed.mp4')
  base_frame = None
  heat_map = cv2.imread('heat_map_intro_removed.png', cv2.COLOR_BGR2GRAY)

  if args.debug:
    cv2.namedWindow('base_frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('raw_frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('heat_map', cv2.WINDOW_NORMAL)
    cv2.namedWindow('p1', cv2.WINDOW_NORMAL)

  # Note that we invert the heat_map here, so only those *cold* portions of the
  # heat map will be kept when we use it as a mask later.
  _, heat_map = cv2.threshold(heat_map, 2, 255, cv2.THRESH_BINARY_INV)
  # Dilate then erode, to "join" the pixels into blobs
  heat_map = cv2.morphologyEx(heat_map, cv2.MORPH_CLOSE, kernel=np.ones((5,5),np.uint8))
  #heat_map = heat_map > 0  # Convert to logical map
  if args.debug: cv2.imshow('heat_map', heat_map)
  
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    if args.debug: cv2.imshow('raw_frame', raw_frame)

    if base_frame is None:
      base_frame = cv2.bitwise_and(raw_frame, cv2.cvtColor(heat_map, cv2.COLOR_GRAY2BGR))
      if args.debug: cv2.imshow('base_frame', base_frame)

    frame = cv2.absdiff(raw_frame, base_frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    frame = cv2.bitwise_and(frame, heat_map)
    
    cv2.imshow('p1', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
