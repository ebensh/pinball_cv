#!/usr/bin/python

import argparse
from collections import deque
import ConfigParser
import cv2
from itertools import islice, izip
import numpy as np
import pickle

import common

COLOR_NEGATIVE = (0, 0, 255)    # Red
COLOR_UNKNOWN  = (255, 0, 255)  # Purple
COLOR_POSITIVE = (0, 255, 0)    # Green

LABEL_NEGATIVE = -1
LABEL_UNKNOWN  = 0
LABEL_POSITIVE = 1
class LabeledPoint(object):
  def __init__(self, x, y, size, label=LABEL_UNKNOWN):
    self.x = x
    self.y = y
    self.size = size
    self.label = label
  def color(self):
    if self.label == LABEL_NEGATIVE: return COLOR_NEGATIVE
    elif self.label == LABEL_UNKNOWN: return COLOR_UNKNOWN
    return COLOR_POSITIVE


class LabeledFrame(object):
  def __init__(self, frame_index, labeled_points):
    self.frame_index = frame_index
    self.labeled_points = labeled_points

  def draw_on_mask(self, mask):
    for lp in self.labeled_points:
      # Filled circle in the mask - use a small size to avoid masking out the ball.
      cv2.circle(mask, (lp.x, lp.y), 3, 255, -1)
    return mask

  def draw_on_frame(self, frame):
    # TODO: Handle more than one pinball :)
    for lp in self.labeled_points:
      cv2.circle(frame, (lp.x, lp.y), 3, lp.color(), -1)
      cv2.putText(frame, str(self.frame_index), (lp.x, max(lp.y-10, 0)),
                  cv2.FONT_HERSHEY_PLAIN, 1, lp.color(), 2)
    return frame

def main():
  FADE_STEP_COUNT = 60  # ~2 seconds
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
  num_frames = len(frame_to_keypoints)
  assert num_frames == max(frame_to_keypoints.keys()) + 1

  labeled_frames = [None] * num_frames
  for index in xrange(num_frames):
    keypoints = [LabeledPoint(int(round(x)), int(round(y)), int(round(size)))
                 for x, y, size in frame_to_keypoints[index]]
    labeled_frames[index] = LabeledFrame(index, keypoints)

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
    frame = raw_frame.copy()  # Unnecessary copy to preserve original. Remove?

    labeled_frame = labeled_frames[frame_index]
    labeled_points = labeled_frame.labeled_points

    # Create a mask of the current keypoints.
    keypoints_mask = np.zeros(frame.shape[:2], np.uint8)
    labeled_frame.draw_on_mask(keypoints_mask)

    # Create a historical mask of the previous N frames.
    historical_mask = np.zeros(frame.shape[:2], np.uint8)
    for past_frame in labeled_frames[frame_index-FADE_STEP_COUNT:frame_index-1]:
      past_frame.draw_on_mask(historical_mask)

    # If a current point is in the historical mask then we filter it.
    for lp in labeled_points:
      if historical_mask[lp.y, lp.x] > 0:
        lp.label = LABEL_NEGATIVE

    if sum(map(lambda lp: lp.label == LABEL_UNKNOWN, labeled_points)) == 1:
      for lp in labeled_points:
        if lp.label == LABEL_UNKNOWN:
          lp.label = LABEL_POSITIVE
          break
    # TODO: If only one point remains, flip it to positive?
    
    # TODO: Create an arrowed line connecting previous positive points?
    #for p1, p2 in izip(previous_pinball_points, islice(previous_pinball_points, 1, None)):
    #  cv2.arrowedLine(frame, p1[:2], p2[:2], (0, 255, 0))

    common.display_image(raw_frame, show=True)
    for lf in labeled_frames[frame_index-10:frame_index+1]:
      lf.draw_on_frame(frame)
    common.display_image(frame, show=True)
    video.write(frame)    

    frame_index += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
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
