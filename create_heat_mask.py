#!/usr/bin/python

import cv2
import numpy as np

def main():
  cap = cv2.VideoCapture('intro_removed.mp4')
  base_frame = None
  heat_map = None
  num_frames = 0

  cv2.namedWindow('base_frame', cv2.WINDOW_NORMAL)
  cv2.namedWindow('raw_frame', cv2.WINDOW_NORMAL)
  cv2.namedWindow('p1', cv2.WINDOW_NORMAL)
  cv2.namedWindow('p2', cv2.WINDOW_NORMAL)
  cv2.namedWindow('heat_map', cv2.WINDOW_NORMAL)
  
  while cap.isOpened():
    ret, raw_frame = cap.read()
    if not ret: break
    num_frames += 1
    cv2.imshow('raw_frame', raw_frame)
    
    if base_frame is None:
      base_frame = raw_frame
      cv2.imshow('base_frame', base_frame)
      heat_map = np.zeros(base_frame.shape[0:2], dtype=np.float64)
    
    frame = cv2.absdiff(raw_frame, base_frame)
    cv2.imshow('p1', frame)

    _, frame = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)
    cv2.imshow('p2', frame)

    heat_map += frame / 255
    #print len(np.flatnonzero(heat_map))
    heat_map_display = (heat_map / num_frames)  # [0, 1] percentage of time this pixel differed
    _, heat_map_display = cv2.threshold(heat_map_display, 0.02, 1.0, cv2.THRESH_BINARY)
    #heat_map_display /= heat_map_display.max()
    
    cv2.imshow('heat_map', heat_map_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
