#!/usr/bin/python

import argparse
import cv2
import numpy as np
from itertools import izip

import common

def main():
  cap = cv2.VideoCapture(args.infile)
  heat_map = None
  num_frames = 0

  _, base_frame = cap.read()

  # For debugging / visual inspection.
  #alphas = [0.2, 0.1, 0.05, 0.01, 0.001]
  #backgrounds = [np.float32(base_frame) for alpha in alphas]

  alpha = 0.2
  background_weighted = np.float32(base_frame)
  background_weighted_min = np.float32(base_frame)
  background_weighted_max = np.float32(base_frame)

  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break

    # http://opencvpython.blogspot.com/2012/07/background-extraction-using-running.html
    common.display_image(raw_frame, 'raw_frame')
    #for alpha, background in izip(alphas, backgrounds):
    #  cv2.accumulateWeighted(raw_frame, background, alpha)
    #  background_scaled = cv2.convertScaleAbs(background)
    #  common.display_image(background_scaled, str(alpha), args.display_all_images)

    cv2.accumulateWeighted(raw_frame, background_weighted, alpha)
    background_weighted_min = cv2.min(background_weighted_min, background_weighted)
    background_weighted_max = cv2.max(background_weighted_max, background_weighted)

    common.display_image(cv2.convertScaleAbs(background_weighted), 'weighted', args.display_all_images)
    common.display_image(cv2.convertScaleAbs(background_weighted_min), 'min', args.display_all_images)
    common.display_image(cv2.convertScaleAbs(background_weighted_max), 'max', args.display_all_images)
    common.display_image(cv2.absdiff(cv2.convertScaleAbs(background_weighted_max), cv2.convertScaleAbs(background_weighted_min)), 'range', args.display_all_images)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
  cv2.imwrite(args.outfile_prefix + "_weighted.png", background_weighted)
  cv2.imwrite(args.outfile_prefix + "_range.png", cv2.absdiff(cv2.convertScaleAbs(background_weighted_max), cv2.convertScaleAbs(background_weighted_min)))
  
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Create a heat mask for a given video.')
  parser.add_argument('--infile', required=True, type=str, help='Input video file path.')
  parser.add_argument('--outfile_prefix', required=True, type=str, help='Output image path.')
  parser.add_argument('--display_all_images', default=False, type=bool,
                      help='Display all (debug) images.')
  args = parser.parse_args()

  main()
