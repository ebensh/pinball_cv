#!/usr/bin/python

import argparse
import cv2
import numpy as np

import common

def main():
  cap = cv2.VideoCapture(args.infile)
  background = cv2.imread(args.background)
  background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
  mask = cv2.imread(args.mask, cv2.COLOR_BGR2GRAY)
  print "Background: {0}, {1}".format(background.shape, background.dtype)
  print "Mask: {0}, {1}".format(mask.shape, mask.dtype)

  dynamic_background = np.float32(background)
  alpha = 0.33
  
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

    common.display_image(raw_frame, 'raw_frame', True)
    common.display_image(background, 'background', True)
    common.display_image(mask, 'mask', True)
    frame = cv2.absdiff(raw_frame, background)
    common.display_image(frame, 'raw_frame - background', True)
    
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    common.display_image(frame, 'masked', True)

    # Now build a dynamic background after the mask is applied.
    cv2.accumulateWeighted(frame, dynamic_background, alpha)
    common.display_image(dynamic_background, 'dynamic_bg', True)
    frame = cv2.absdiff(frame, cv2.convertScaleAbs(dynamic_background))
    common.display_image(frame, 'without dynamic_bg', True)

    _, pinball_mask = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY)
    #pinball_mask = cv2.inRange(pinball_mask, 35, 120)
    common.display_image(pinball_mask, 'pinball_mask', True)
    
    # frame = cv2.absdiff(raw_frame, base_frame)
    # frame = cv2.bitwise_and(frame, heat_map)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)

    # common.display_image(frame, show=args.display_all_images)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Track a pinball.')
  parser.add_argument('--infile', required=True, type=str, help='Input video file path.')
  parser.add_argument('--background', type=str, help='Path to background to subtract.')
  parser.add_argument('--mask', type=str, help='Mask to apply to each frame.')
  parser.add_argument('--display_all_images', default=False, type=bool,
                      help='Display all (debug) images.')
  args = parser.parse_args()
  main()
