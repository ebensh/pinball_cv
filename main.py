#!/usr/bin/python

import argparse
from collections import deque
import cv2
import itertools
import numpy as np

import common

def main():
  cap = cv2.VideoCapture(args.infile)
  background = cv2.imread(args.background, cv2.IMREAD_GRAYSCALE)
  mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
  print "Background: {0}, {1}".format(background.shape, background.dtype)
  print "Mask: {0}, {1}".format(mask.shape, mask.dtype)
  
  FRAME_BUFFER_SIZE = 20
  frame_buffer = deque([], maxlen=FRAME_BUFFER_SIZE)
  
  dynamic_background = np.zeros(background.shape, dtype=np.float64)
  alpha = 0.2    
  
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break

    frame_buffer.append(raw_frame)
    if len(frame_buffer) < FRAME_BUFFER_SIZE:
      continue  # Loop until the frame buffer is full at the start.

    # We're processing FRAME_BUFFER_SIZE images in the past.
    frame = frame_buffer[0]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    common.display_image(frame, 'input frame', True)
    common.display_image(frame_gray, 'input frame gray', True)

    # Subtract away the unchanging background.
    cv2.accumulateWeighted(frame_gray, dynamic_background, alpha)
    frame_gray = cv2.absdiff(frame_gray, cv2.convertScaleAbs(dynamic_background))
    common.display_image(frame_gray, 'frame gray - dynamic_background', True)

    # We want to accumulate large values for pixels that are often different
    # in the next FRAME_BUFFER_SIZE frames, to detect lights that go on or off.
    # Once we normalize this accumulation the ball should be relatively small,
    # as it is in motion.
    base_frame = cv2.cvtColor(frame_buffer[5], cv2.COLOR_BGR2GRAY)
    accumulated_difference = np.zeros(base_frame.shape, dtype=np.float64)
    for future_frame in itertools.islice(frame_buffer, 6, None):
      future_base_frame = cv2.cvtColor(future_frame, cv2.COLOR_BGR2GRAY)
      diff = cv2.absdiff(base_frame, future_base_frame)
      cv2.accumulate(diff, accumulated_difference)
    accumulated_difference = cv2.convertScaleAbs(accumulated_difference)
    accumulated_difference = cv2.inRange(accumulated_difference, 0, 50)
    common.display_image(accumulated_difference, 'accumulated difference')

    # Mask away the light changes we see in the future.
    #_, accumulated_difference = cv2.threshold(accumulated_difference, 50, 255, cv2.THRESH_BINARY_INV)
    frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=accumulated_difference)
    _, _, _, best_guess_loc = cv2.minMaxLoc(frame_gray)
    print best_guess_loc
    cv2.circle(frame, best_guess_loc, 30, (255, 255, 255))
    common.display_image(frame, 'frame masked')


    # -----------
    # raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

    # common.display_image(background, 'background', True)
    # common.display_image(mask, 'mask', True)

    # cv2.accumulateWeighted(raw_frame, dynamic_background, alpha)
    # frame = cv2.absdiff(raw_frame, cv2.convertScaleAbs(dynamic_background))
    # common.display_image(frame, 'raw_frame - dynamic_background', True)

    # # Mask the play field using the statically computed mask, which keeps areas
    # # that don't vary a lot.
    # frame = cv2.bitwise_and(frame, frame, mask=mask)
    # common.display_image(frame, 'masked', True)

    # frame = cv2.erode(frame, None, iterations=2)
    # frame = cv2.dilate(frame, None, iterations=4)
    # frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # #_, frame = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY)
    # common.display_image(frame)
    
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break
    # continue

    # # Subtract the statically computed background, highlighting foreground.
    # frame = cv2.absdiff(raw_frame, background)
    # common.display_image(frame, 'raw_frame - background', True)

    # # Mask the play field using the statically computed mask, which keeps areas
    # # that don't vary a lot.
    # frame = cv2.bitwise_and(frame, frame, mask=mask)
    # common.display_image(frame, 'masked', True)

    # #_, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_TOZERO)
    # #frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    # #frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 10)
    # #common.display_image(frame, 'adaptive_threshold', True)

    # # Now build a dynamic background after the mask is applied.
    # cv2.accumulateWeighted(frame, dynamic_background, alpha)
    # common.display_image(dynamic_background, 'dynamic_bg', True)
    # frame = cv2.absdiff(frame, cv2.convertScaleAbs(dynamic_background))
    # common.display_image(frame, 'without dynamic_bg', True)

    # #_, pinball_mask = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY)
    # pinball_mask = cv2.inRange(frame, 30, 100)
    # common.display_image(pinball_mask, 'pinball_mask', True)
    
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
