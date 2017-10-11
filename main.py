#!/usr/bin/python

import argparse
from collections import deque
import cv2
import itertools
import numpy as np

import common

def main():
  cap = cv2.VideoCapture(args.infile)
  #background = cv2.imread(args.background, cv2.IMREAD_GRAYSCALE)
  #mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
  #print "Background: {0}, {1}".format(background.shape, background.dtype)
  #print "Mask: {0}, {1}".format(mask.shape, mask.dtype)
  
  FRAME_BUFFER_SIZE = 20
  frame_buffer = deque([], maxlen=FRAME_BUFFER_SIZE)
  
  # Loop until the frame buffer is full at the start.
  while cap.isOpened() and len(frame_buffer) < FRAME_BUFFER_SIZE:
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    frame_buffer.append((raw_frame, cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)))

  dynamic_background = np.zeros(frame_buffer[0][0].shape, dtype=np.float64)
  alpha = 0.2

  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    frame_buffer.append((raw_frame, cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)))

    frame_printer = common.FramePrinter()
    
    # We're processing FRAME_BUFFER_SIZE images in the past.
    frame, frame_gray = frame_buffer[0]
    common.display_image(frame, 'input frame', True)
    common.display_image(frame_gray, 'input frame gray', True)
    frame_printer.add_image(frame, 'frame')
    frame_printer.add_image(frame_gray, 'frame gray')
    common.display_image(frame_printer.get_combined_image())

    # # Subtract away the unchanging background.
    # cv2.accumulateWeighted(frame_gray, dynamic_background, alpha)
    # frame_gray = cv2.absdiff(frame_gray, cv2.convertScaleAbs(dynamic_background))
    # common.display_image(frame_gray, 'frame gray - dynamic_background', True)

    # # We want to accumulate large values for pixels that are often different
    # # in the next FRAME_BUFFER_SIZE frames, to detect lights that go on or off.
    # # Once we normalize this accumulation the ball should be relatively small,
    # # as it is in motion.
    # base_frame = cv2.cvtColor(frame_buffer[5], cv2.COLOR_BGR2GRAY)
    # accumulated_difference = np.zeros(base_frame.shape, dtype=np.float64)
    # for future_frame in itertools.islice(frame_buffer, 6, None):
    #   future_base_frame = cv2.cvtColor(future_frame, cv2.COLOR_BGR2GRAY)
    #   diff = cv2.absdiff(base_frame, future_base_frame)
    #   cv2.accumulate(diff, accumulated_difference)
    # accumulated_difference = cv2.convertScaleAbs(accumulated_difference)
    # accumulated_difference = cv2.inRange(accumulated_difference, 0, 50)
    # common.display_image(accumulated_difference, 'accumulated difference')

    # # Mask away the light changes we see in the future.
    # #_, accumulated_difference = cv2.threshold(accumulated_difference, 50, 255, cv2.THRESH_BINARY_INV)
    # frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=accumulated_difference)
    # _, _, _, best_guess_loc = cv2.minMaxLoc(frame_gray)
    # print best_guess_loc
    # cv2.circle(frame, best_guess_loc, 30, (255, 255, 255))
    # common.display_image(frame, 'frame masked')

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Track a pinball.')
  parser.add_argument('--infile', required=True, type=str, help='Input video file path.')
  #parser.add_argument('--background', type=str, help='Path to background to subtract.')
  #parser.add_argument('--mask', type=str, help='Mask to apply to each frame.')
  parser.add_argument('--display_all_images', default=False, type=bool,
                      help='Display all (debug) images.')
  args = parser.parse_args()
  main()
