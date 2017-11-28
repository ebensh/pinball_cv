#!/usr/bin/python

import argparse
from collections import deque
import cv2
import itertools
import numpy as np

import common

def main():
  cv2.namedWindow('combined', cv2.WINDOW_NORMAL)

  cap = cv2.VideoCapture(args.infile)
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break

    gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 120)
    
    common.display_image(edges, 'edges')
    if cv2.waitKey(33) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Track a pinball.')
  parser.add_argument('--infile', required=True, type=str, help='Input video file path.')
  args = parser.parse_args()
  main()
