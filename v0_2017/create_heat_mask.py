#!/usr/bin/python

import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Create a heat mask for a given video.')
parser.add_argument('--infile', required=True, type=str, help='Input video file path.')
parser.add_argument('--outfile', required=True, type=str, help='Output image path.')
parser.add_argument('--debug', default=False, type=bool, help='Display processing steps.')
args = parser.parse_args()

def main():
  cap = cv2.VideoCapture(args.infile)
  base_frame = None
  heat_map = None
  num_frames = 0

  if args.debug:
    cv2.namedWindow('base_frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('raw_frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('p1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('p2', cv2.WINDOW_NORMAL)
    cv2.namedWindow('heat_map', cv2.WINDOW_NORMAL)
  
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    num_frames += 1
    if args.debug: cv2.imshow('raw_frame', raw_frame)
    
    if base_frame is None:
      base_frame = raw_frame
      if args.debug: cv2.imshow('base_frame', base_frame)
      heat_map = np.zeros(base_frame.shape[0:2], dtype=np.float64)
    
    frame = cv2.absdiff(raw_frame, base_frame)
    if args.debug: cv2.imshow('p1', frame)

    _, frame = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)
    if args.debug: cv2.imshow('p2', frame)

    heat_map[frame > 0] += 1
    if args.debug:
      heat_map_display = (heat_map / num_frames)  # [0, 1] percentage of time this pixel differed
      _, heat_map_display = cv2.threshold(heat_map_display, 0.01, 1.0, cv2.THRESH_BINARY)
      cv2.imshow('heat_map', heat_map_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  heat_map /= num_frames  # Normalize as percentage of frames
  heat_map /= np.max(heat_map)  # Normalize to [0.0, 1.0] range
  heat_map *= 255  # Convert to greyscale for writing as png.
  cv2.imwrite(args.outfile, heat_map)
  
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
