#!/usr/bin/python

import argparse
import cv2
import itertools
import numpy as np

import common


def main():
  cap = cv2.VideoCapture(args.infile)

  frame_count = 0
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    if frame_count == args.frame:
      cv2.imwrite(args.outfile, raw_frame)
      break
    frame_count += 1
  cap.release()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Track a pinball.')
  parser.add_argument('--infile', required=True, type=str, help='Input video file path.')
  parser.add_argument('--outfile', type=str, help='Path to output image file.')
  parser.add_argument('--frame', type=int, help='Frame number to export (0 index).')
  args = parser.parse_args()
  main()
