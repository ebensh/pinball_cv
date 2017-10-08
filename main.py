#!/usr/bin/python

import cv2
import numpy as np

def main():
  cap = cv2.VideoCapture('pinball.mp4')
  base_frame = None

  cv2.namedWindow('base_frame', cv2.WINDOW_NORMAL)
  cv2.namedWindow('raw_frame', cv2.WINDOW_NORMAL)
  cv2.namedWindow('p1', cv2.WINDOW_NORMAL)
  cv2.namedWindow('p2', cv2.WINDOW_NORMAL)
  
  while cap.isOpened():
    ret, raw_frame = cap.read()
    cv2.imshow('raw_frame', raw_frame)
    
    if base_frame is None:
      base_frame = raw_frame
      prev_frame = raw_frame
      cv2.imshow('base_frame', base_frame)
    
    frame = cv2.absdiff(raw_frame, base_frame)
    cv2.imshow('p1', frame)

    _, frame = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
    print frame
    cv2.imshow('p2', frame)

    prev_frame = frame
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
