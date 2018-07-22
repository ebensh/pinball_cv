#!/usr/bin/python

import cv2
import numpy as np

def main():
  cap = cv2.VideoCapture('intro_removed.mp4')

  num_frames = 0
  while cap.isOpened():
    grabbed, frame = cap.read()
    if not grabbed: break
    num_frames += 1
    
  cap.release()
  cv2.destroyAllWindows()
  print "Number of frames processed: {0}".format(num_frames)

if __name__ == '__main__':
  main()
