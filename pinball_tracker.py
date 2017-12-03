#!/usr/bin/python

# Tracks all pinball-ish blobs across a pinball playfield video, outputting a
# json dictionary of frame number -> keypoint triples (x, y, size).

import argparse
import common
import ConfigParser
import cv2
import itertools
import json
import numpy as np


class OutputSegment(object):
  @staticmethod
  def _get_blob_detector():
    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 256;
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 600
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.6
    
    # Filter by Convexity
    params.filterByConvexity = False
    #params.minConvexity = 0.5
      
    # Filter by Inertia
    #params.filterByInertia = True
    #params.minInertiaRatio = 0.5
    return cv2.SimpleBlobDetector_create(params)
  
  def __init__(self, path):
    self._path = path
    self._detector = OutputSegment._get_blob_detector()
    self._frame_to_keypoints = {}

  def process_frame(self, frame, frame_index):
    # Do blob detection and dictionary updating here.
    # IMPORTANT! The frame should have BLACK objects on WHITE background.
    keypoints = self._detector.detect(frame)
    print keypoints
    self._frame_to_keypoints[frame_index] = [
      (kp.pt[0], kp.pt[1], kp.size) for kp in keypoints]

  def release(self):
    # Write the json keypoints.
    with open(self._path, 'wb') as keypoints_file:
      json.dump(self._frame_to_keypoints, keypoints_file)


def main():
  FRAME_BUFFER_SIZE = 9
  CURRENT_FRAME_INDEX = FRAME_BUFFER_SIZE/2
  game_config = ConfigParser.ConfigParser()
  game_config.read(args.game_config)
  input_rows = game_config.getint('PinballFieldVideo', 'rows')
  input_cols = game_config.getint('PinballFieldVideo', 'cols')
  cap = cv2.VideoCapture(game_config.get('PinballFieldVideo', 'path'))
  
  keypoints_path = game_config.get('PinballFieldVideo', 'keypoints_path')
  keypoint_detector = OutputSegment(keypoints_path)

  frame_buffer = common.FrameBuffer(FRAME_BUFFER_SIZE, (input_rows, input_cols, 3))

  if args.display_all_images:
    cv2.namedWindow('past_stats', cv2.WINDOW_NORMAL)
    cv2.namedWindow('combined', cv2.WINDOW_NORMAL)
    cv2.namedWindow('masks', cv2.WINDOW_NORMAL)
  frame_index = 0
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break

    common.display_image(raw_frame, 'raw_frame', args.display_all_images)
    frame_buffer.append(raw_frame)

    past = frame_buffer.get_view(None, CURRENT_FRAME_INDEX - 2)
    past_gray = frame_buffer.get_view(None, CURRENT_FRAME_INDEX - 2, color=False)
    current_frame = frame_buffer.get_view(CURRENT_FRAME_INDEX, CURRENT_FRAME_INDEX + 1)[0]
    current_frame_gray = frame_buffer.get_view(CURRENT_FRAME_INDEX, CURRENT_FRAME_INDEX + 1, color=False)[0]
    #current_frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    future = frame_buffer.get_view(CURRENT_FRAME_INDEX + 1 + 2, None)
    future_gray = frame_buffer.get_view(CURRENT_FRAME_INDEX + 1 + 2, None, color=False)

    past_stats = common.get_named_statistics(past_gray)
    future_stats = common.get_named_statistics(future_gray)

    if frame_index % 3 == 0:
      common.display_image(
          cv2.vconcat([cv2.hconcat(past_gray), cv2.hconcat(future_gray)]),
          'combined', args.display_all_images)
      past_stats_printer = common.FramePrinter()
      common.print_statistics(past_stats, past_stats_printer)
      common.display_image(past_stats_printer.get_combined_image(), 'past_stats', args.display_all_images)
    
    # Subtract out the unchanging background (mean past, mean future) from current frame
    foreground_mask_past = np.absolute(current_frame_gray.astype(np.int16) - past_stats.mean.astype(np.int16))
    foreground_mask_future = np.absolute(current_frame_gray.astype(np.int16) - future_stats.mean.astype(np.int16))

    # Threshold the differences and combine them.
    foreground_mask = np.logical_and(foreground_mask_past >= 25, foreground_mask_future >= 25)

    # Mask away the areas we know are changing based on thresholded ptp (ptp past, ptp future).
    # Take the absolute difference (per pixel) from the mean in each frame.
    changing_mask_past = np.absolute(past_gray.astype(np.int16) - past_stats.mean.astype(np.int16)) >= 5
    # Count how many frames were significantly different and threshold.
    changing_mask_past = np.sum(changing_mask_past, axis=0) >= 3
    
    # Take the absolute difference (per pixel) from the mean in each frame.
    changing_mask_future = np.absolute(future_gray.astype(np.int16) - future_stats.mean.astype(np.int16)) >= 5
    # Count how many frames were significantly different.
    changing_mask_future = np.sum(changing_mask_future, axis=0) >= 3

    changing_mask = np.logical_or(changing_mask_past, changing_mask_future)

    # Create a mask from the HSV image to identify bright areas (high value).
    #lights_mask = np.uint8(255) * (current_frame_hsv[:,:,2] > 235)
    # Erode then dilate.
    #lights_mask = cv2.morphologyEx(lights_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

    # Create a mask from the HSV image to identify saturated areas (non-gray).
    #colorful_mask = np.uint8(255) * (current_frame_hsv[:,:,1] > 20)
    #colorful_mask = cv2.morphologyEx(colorful_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

    #changing_mask = np.logical_or(changing_mask, lights_mask)
    #changing_mask = np.logical_or(changing_mask, colorful_mask)

    # The final mask is the foreground (keep) minus the changing mask (remove).
    final_mask = np.uint8(255) * np.logical_and(foreground_mask, np.logical_not(changing_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Erode (remove small outlying pixels), then dilate.
    final_mask_polished = final_mask.copy()
    final_mask_polished = cv2.erode(final_mask_polished, np.ones((3, 3),np.uint8), iterations=1)
    final_mask_polished = cv2.dilate(final_mask_polished, kernel, iterations=2)
    #final_mask_polished = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    common.display_image(np.hstack([
      np.uint8(255) * foreground_mask,
      np.uint8(255) * changing_mask,
      final_mask,
      final_mask_polished]), 'masks', args.display_all_images)

    keypoint_detector.process_frame(cv2.bitwise_not(final_mask_polished), frame_index)  # Invert for blob detection.

    frame_index += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  print 'Frames processed: %d' % frame_index

  cv2.destroyAllWindows()
  cap.release()
  keypoint_detector.release()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Track a pinball.')
  parser.add_argument('--game_config', required=True, type=str, help='Game configuration file.')
  parser.add_argument('--display_all_images', default=False, type=bool,
                      help='Display all (debug) images.')
  args = parser.parse_args()
  main()
