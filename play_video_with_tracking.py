#!/usr/bin/python

import argparse
from collections import deque
import ConfigParser
import cv2
from itertools import islice, izip
import numpy as np
import pickle

import common

def main():
  FADE_STEP_COUNT = 30
  game_config = ConfigParser.ConfigParser()
  game_config.read(args.game_config)
  input_rows = game_config.getint('PinballFieldVideo', 'rows')
  input_cols = game_config.getint('PinballFieldVideo', 'cols')
  cap = cv2.VideoCapture(game_config.get('PinballFieldVideo', 'path'))
  video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'),
                          30.0, (input_cols, input_rows))

  frame_to_keypoints = None
  with open(game_config.get('PinballFieldVideo', 'keypoints_path'), 'rb') as keypoints_file:
    frame_to_keypoints = pickle.load(keypoints_file)

  background = cv2.imread(args.background, cv2.IMREAD_COLOR)
  # We use point_history as a mask that fades over time. Each time we see a
  # point (circle) we set it in the mask to 255. We create a new mask of the
  # current frame's points - the background mask, keeping only the remaining
  # points.
  keypoints_mask_history = np.zeros(background.shape[:2], np.uint8)

  previous_pinball_points = deque(maxlen=10)

  frame_index = 0
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    frame = raw_frame.copy()

    keypoints = frame_to_keypoints.get(frame_index + 4, [])
    keypoints = [(int(round(x)), int(round(y)), int(round(size))) for x, y, size in keypoints]
    print frame_index, keypoints

    # Create a mask of the current keypoints, using small size for locality.
    keypoints_mask = np.zeros(frame.shape[:2], np.uint8)
    for x, y, size in keypoints:
      # Filled circle in the mask - use a small size to avoid masking out the ball.
      cv2.circle(keypoints_mask, (x, y), 6, 255, -1)

    # If a point is in the current mask but *NOT* the historical mask, then we
    # keep it.
    keypoints_kept = set()
    keypoints_filtered = set()
    for x, y, size in keypoints:
      if np.logical_and(keypoints_mask[y, x] > 0, keypoints_mask_history[y, x] == 0):
        keypoints_kept.add((x, y, size))
      else:
        keypoints_filtered.add((x, y, size))

    # def euclidean_distance(x1, y1, x2, y2): return (x2 - x1)**2 + (y2 - y1)**2
    # pinball_x = pinball_y = None
    # if previous_pinball_points:
    #   pinball_x, pinball_y, _ = previous_pinball_points[-1]
    #   for x, y, size in list(keypoints_kept):  # Copy the list so we can remove while iterating
    #     if euclidean_distance(x, y, pinball_x, pinball_y) >= 100:
    #       keypoints_kept.remove((x, y, size))
    #       keypoints_filtered.add((x, y, size))

    kept_color = (255, 0, 255)  # Purple if we're not sure.
    if len(keypoints_kept) == 1:
      kept_color = (0, 255, 0)  # Green if we're sure.
      previous_pinball_points.append(list(keypoints_kept)[0])
    filtered_color = (0, 0, 255)  # Red if we're filtering.

    for x, y, size in keypoints_kept:
      cv2.circle(frame, (x, y), int(size/2), kept_color, -1)
    for x, y, size in keypoints_filtered:
      cv2.circle(frame, (x, y), int(size/2), filtered_color, -1)

      #cv2.drawMarker(frame, (int(x), int(y)), color, markerSize=int(size/2),
        #               line_type=cv2.MARKER_TILTED_CROSS)

    for p1, p2 in izip(previous_pinball_points, islice(previous_pinball_points, 1, None)):
      cv2.arrowedLine(frame, p1[:2], p2[:2], (0, 255, 0))

    # Add our current mask to the history.
    keypoints_mask_history = cv2.subtract(keypoints_mask_history, 255 / FADE_STEP_COUNT)
    keypoints_mask_history = np.maximum(keypoints_mask_history, keypoints_mask)

    common.display_image(frame, show=True)
    common.display_image(keypoints_mask, show=True)
    common.display_image(keypoints_mask_history, show=True)
    video.write(frame)
    
    
    frame_index += 1
    if cv2.waitKey(33) & 0xFF == ord('q'):
      break
    
  cap.release()
  video.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Play video with tracking overlay.')
  parser.add_argument('--game_config', required=True, type=str, help='Game configuration file.')
  parser.add_argument('--display_all_images', default=False, type=bool,
                      help='Display all (debug) images.')
  parser.add_argument('--background', required=True, type=str, help='Background to help visualize.')
  args = parser.parse_args()
  main()
