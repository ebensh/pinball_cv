#!/usr/bin/python

import argparse
import inspect
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Create a heat mask for a given video.')
parser.add_argument('--infile', required=True, type=str, help='Input image path.')
parser.add_argument('--outfile', type=str, help='Output image path.')
parser.add_argument('--debug', default=True, type=bool, help='Display processing steps.')
args = parser.parse_args()

debug_step = 0
def display_image(img, title=None, show=args.debug):
  global debug_step
  if args.debug:
    if title is None:
      # Get the caller's line number so we can identify which point in the
      # process we're at without uniquely naming each one.
      frame, filename, line_num, function_name, lines, index = inspect.stack()[1]
      title = "Step {0}, Line {1}".format(debug_step, line_num)
      debug_step += 1
    cv2.imshow(title, img)


def main():
  img = cv2.imread(args.infile)
  display_image(img, 'original')

  # Apply effects here to see outputs.
  _, img = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
  display_image(img)
  
  kernel = np.ones((5, 5), np.uint8)
  img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  display_image(img)

  if args.outfile:
    cv2.imwrite(args.outfile, img)

  if args.debug:
    while True:
      if cv2.waitKey(0) & 0xFF == ord('q'):
        break
  
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
