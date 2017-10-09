#!/usr/bin/python

import argparse
import cv2
import numpy as np

import common

def main():
  img = cv2.imread(args.infile)
  common.display_image(img, 'original', args.display_all_images)

  # Apply effects here to see outputs.
  _, img = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
  common.display_image(img, show=args.display_all_images)
  
  kernel = np.ones((9, 9), np.uint8)
  img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  common.display_image(img, show=args.display_all_images)

  if args.outfile:
    cv2.imwrite(args.outfile, img)

  if args.display_all_images:
    while True:
      if cv2.waitKey(0) & 0xFF == ord('q'):
        break
  
  cv2.destroyAllWindows()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Create a heat mask for a given video.')
  parser.add_argument('--infile', required=True, type=str, help='Input image path.')
  parser.add_argument('--outfile', type=str, help='Output image path.')
  parser.add_argument('--display_all_images', default=False, type=bool,
                      help='Display all (debug) images.')
  args = parser.parse_args()
  main()
