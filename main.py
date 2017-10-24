#!/usr/bin/python

import argparse
import cv2
import itertools
import numpy as np

import common

def main():
  cap = cv2.VideoCapture(args.infile)
  video = None
  #background = cv2.imread(args.background, cv2.IMREAD_GRAYSCALE)
  #mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
  #print "Background: {0}, {1}".format(background.shape, background.dtype)
  #print "Mask: {0}, {1}".format(mask.shape, mask.dtype)
  
  FRAME_BUFFER_SIZE = 10
  
  # Loop until the frame buffer is full at the start.
  while cap.isOpened() and len(frame_buffer) < FRAME_BUFFER_SIZE:
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    frame_buffer.append((raw_frame, cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)))

  dynamic_background = np.zeros(frame_buffer[0][1].shape, dtype=np.float64)
  alpha = 0.33

  cv2.namedWindow('combined', cv2.WINDOW_NORMAL)
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    frame_buffer.append((raw_frame, cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)))

    frame_printer = common.FramePrinter()
    
    # We're processing FRAME_BUFFER_SIZE images in the past.
    frame, frame_gray = frame_buffer[0]
    frame_printer.add_image(frame, 'frame')
    frame_printer.add_image(frame_gray, 'frame gray')

    if False:
      # We want to accumulate large values for pixels that are often different
      # in the next several frames, to detect lights that go on or off.
      # Once we normalize this accumulation the ball should be relatively small,
      # as it is in motion.
      base_frame_gray = frame_buffer[1][1]  # Gray image three frames forward.
      #minimum_difference = np.full(base_frame_gray.shape, 255, dtype=np.uint8)
      maximum_difference = np.zeros(base_frame_gray.shape, dtype=np.uint8)
      #accumulated_difference = np.zeros(base_frame_gray.shape, dtype=np.float64)
      for _, future_frame_gray in itertools.islice(frame_buffer, 2, None):
        diff = cv2.absdiff(base_frame_gray, future_frame_gray)
        #minimum_difference = cv2.min(minimum_difference, diff)
        maximum_difference = cv2.max(maximum_difference, diff)
        #cv2.accumulate(diff, accumulated_difference)
        #delta_difference = maximum_difference - minimum_difference
        #delta_difference *= 255.0 / delta_difference.max()
        #accumulated_difference = cv2.convertScaleAbs(accumulated_difference)
        #frame_printer.add_image(minimum_difference, 'minimum difference')
      frame_printer.add_image(maximum_difference, 'maximum difference')
      #frame_printer.add_image(delta_difference, 'delta difference')
      #frame_printer.add_image(accumulated_difference, 'accumulated difference')

      # Mask away the light changes we see in the future.
      _, future_difference_mask = cv2.threshold(maximum_difference, 25, 255, cv2.THRESH_BINARY_INV)
      frame_printer.add_image(future_difference_mask, 'future difference mask')

      foreground_mask = background_difference
      foreground_mask *= 255.0 / foreground_mask.max()
      foreground_mask = cv2.bitwise_and(foreground_mask,
                                        foreground_mask,
                                        mask=future_difference_mask)
      _, foreground_mask = cv2.threshold(foreground_mask, 75, 255, cv2.THRESH_BINARY)
      frame_printer.add_image(foreground_mask, 'foreground_mask')

      frame_printer.add_image(cv2.bitwise_and(frame, frame, mask=foreground_mask), 'foreground')
    
      # _, _, _, best_guess_loc = cv2.minMaxLoc(frame_gray)
      # print best_guess_loc
      # cv2.circle(frame, best_guess_loc, 30, (255, 255, 255))
      # common.display_image(frame, 'frame masked')
    
    if False:
      # Create a mask for the unchanging background and subtract it away.
      cv2.accumulateWeighted(frame_gray, dynamic_background, alpha)
      background_difference = cv2.absdiff(frame_gray, cv2.convertScaleAbs(dynamic_background))
      frame_printer.add_image(background_difference, 'frame_gray - dynamic_background')

      frame_gray_float = (frame_gray / 255.0).astype(np.float32)
      future_frame_gray_float = (frame_buffer[1][1] / 255.0).astype(np.float32)
      (point_x, point_y), value = cv2.phaseCorrelate(frame_gray_float, future_frame_gray_float)
      print point_x, point_y, value
      cv2.circle(frame, point, 30, (255, 255, 255))
      frame_printer.add_image(frame, 'translation point')

    combined_image = frame_printer.get_combined_image()
    common.display_image(combined_image, 'combined')
    if video is None:
      video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'),
                              30.0, combined_image.shape[1::-1])
    video.write(combined_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  video.release()
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
