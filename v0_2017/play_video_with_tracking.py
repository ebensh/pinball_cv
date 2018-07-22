#!/usr/bin/python

import argparse
from collections import deque
import ConfigParser
import cv2
from itertools import islice, izip
import numpy as np
import pickle
import sys

import common
from common import eprint

COLOR_NEGATIVE     = (0, 0, 255)    # Red
COLOR_UNKNOWN      = (255, 0, 255)  # Purple
COLOR_POSITIVE     = (0, 255, 0)    # Green
COLOR_EXTRAPOLATED = (0, 255, 255)  # Yellow

LABEL_NEGATIVE = 0
LABEL_UNKNOWN  = 1
LABEL_POSITIVE = 2
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
  def xy(self):
    return (self.x, self.y)


class LabeledFrame(object):
  def __init__(self, frame_index, labeled_points):
    self.frame_index = frame_index
    self.labeled_points = labeled_points

  def add_point(self, point):
    self.labeled_points.append(point)

  def num_points(self, label=None):
    if label is None:
      return len(self.labeled_points)
    return sum(map(lambda lp: lp.label == label, self.labeled_points))

  def positive_point(self):
    for point in self.labeled_points:
      if point.label == LABEL_POSITIVE:
        return point
    return None

  def draw_on_mask(self, mask):
    for lp in self.labeled_points:
      # Filled circle in the mask - use a small size to avoid masking out the ball.
      cv2.circle(mask, (lp.x, lp.y), 4, 255, -1)
    return mask

  def draw_on_frame(self, frame):
    # TODO: Handle more than one pinball :)
    for lp in self.labeled_points:
      cv2.circle(frame, (lp.x, lp.y), 4, lp.color(), -1)
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

  frame_index = 0
  pass_iteration = 0
  point_label_counts = {pass_iteration: [0, 0, 0]}  # Negative, Unknown, Positive
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    frame = raw_frame.copy()  # Unnecessary copy to preserve original. Remove?

    labeled_frame = labeled_frames[frame_index]
    labeled_points = labeled_frame.labeled_points
    def prev_frame(n): return labeled_frames[max(frame_index - n, 0)]
    def next_frame(n): return labeled_frames[min(frame_index + n, len(labeled_frames) - 1)]

    # Create a mask of the current keypoints.
    keypoints_mask = np.zeros(frame.shape[:2], np.uint8)
    labeled_frame.draw_on_mask(keypoints_mask)

    # Create a historical mask of the previous N frames.
    historical_mask = np.zeros(frame.shape[:2], np.uint8)
    for past_frame in labeled_frames[max(frame_index-FADE_STEP_COUNT,0):frame_index-1]:
      past_frame.draw_on_mask(historical_mask)

    # If a current point is in the historical mask then we filter it.
    for lp in labeled_points:
      if lp.label == LABEL_UNKNOWN and historical_mask[lp.y, lp.x] > 0:
        lp.label = LABEL_NEGATIVE

    if labeled_frame.num_points(LABEL_UNKNOWN) == 1:
      # If only one point remains UNKNOWN, then we flip it to positive (though we
      # may later undo this!)
      for lp in labeled_points:
        if lp.label == LABEL_UNKNOWN:
          lp.label = LABEL_POSITIVE
          break

    def extrapolate(lp1, lp2):
      vx = lp2.x - lp1.x
      vy = lp2.y - lp1.y
      return (lp2.x + vx, lp2.y + vy, int((lp1.size + lp2.size) / 2))
    def lerp((x1, y1), (x2, y2)):
      return ((x1 + x2) / 2,
              (y1 + y2) / 2)
    def dist((x1, y1), (x2, y2)):
      return (x2 - x1)**2 + (y2 - y1)**2
    def in_bounds(input_cols, input_rows, (x, y)):
      return (x >= 0 and x < input_cols and y >= 0 and y < input_rows)
    
    # Extrapolate from the n-2 and n-1 frame to get a past prediction of current
    # location as a vector of the same magnitude and direction (best guess).
    two_before = prev_frame(2).positive_point()
    one_before = prev_frame(1).positive_point()
    past_extrapolation = None
    if two_before and one_before:
      past_extrapolation = extrapolate(two_before, one_before)
      if not in_bounds(input_cols, input_rows, past_extrapolation[:2]):
        past_extrapolation = None
      else:
        cv2.arrowedLine(frame, two_before.xy(), past_extrapolation[:2],
                        COLOR_EXTRAPOLATED)

    one_after = next_frame(1).positive_point()
    two_after = next_frame(2).positive_point()
    future_extrapolation = None
    if two_after and one_after:
      future_extrapolation = extrapolate(two_after, one_after)
      if not in_bounds(input_cols, input_rows, future_extrapolation[:2]):
        future_extrapolation = None
      else:
        cv2.arrowedLine(frame, two_after.xy(), future_extrapolation[:2],
                        COLOR_EXTRAPOLATED)

    before_and_after_lerp = None
    if one_before and one_after:
      before_and_after_lerp = lerp(one_before.xy(), one_after.xy())
      if not in_bounds(input_cols, input_rows, before_and_after_lerp):
        before_and_after_lerp = None
      else:
        cv2.arrowedLine(frame, one_before.xy(), before_and_after_lerp,
                        COLOR_EXTRAPOLATED)
        cv2.arrowedLine(frame, one_after.xy(), before_and_after_lerp,
                        COLOR_EXTRAPOLATED)

    # If we have no keypoints in the current frame, try adding the
    # past_extrapolation, future_extrapolation, and past_and_future_lerp as unknown
    # keypoints for next pass.
    if not labeled_points or labeled_frame.num_points() == labeled_frame.num_points(LABEL_NEGATIVE):
      print "Adding points to frame:", frame_index
      if past_extrapolation:
        labeled_points.append(LabeledPoint(
          past_extrapolation[0], past_extrapolation[1], one_before.size))
      if future_extrapolation:
        labeled_points.append(LabeledPoint(
          future_extrapolation[0], future_extrapolation[1], one_after.size))
      if before_and_after_lerp:
        labeled_points.append(LabeledPoint(
          before_and_after_lerp[0], before_and_after_lerp[1], one_before.size))

    if past_extrapolation and future_extrapolation:
      # This is where we expect the ball to be, based on past and future
      # POSITIVE points.
      past_and_future_lerp = lerp(past_extrapolation[:2], future_extrapolation[:2])
      least_distance, closest_point = sys.maxint, None
      for lp in labeled_points:
        distance = dist(lp.xy(), past_and_future_lerp)
        if distance < least_distance:
          least_distance = distance
          closest_point = lp
      if closest_point and closest_point.label == LABEL_POSITIVE and least_distance > 30:
        closest_point.label = LABEL_UNKNOWN
      if closest_point and least_distance < 30:
        closest_point.label = LABEL_POSITIVE
        for lp in labeled_points:
          if lp.label != LABEL_POSITIVE:
            lp.label = LABEL_NEGATIVE
            
    for lp in labeled_points:
      point_label_counts[pass_iteration][lp.label] += 1

    #common.display_image(raw_frame, show=True)
    for lf in labeled_frames[frame_index-3:frame_index+4]:
      lf.draw_on_frame(frame)
    for iteration, label_counts in point_label_counts.iteritems():
      cv2.putText(frame, str(label_counts), (0, 12 * (iteration + 1)),
                  cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    cv2.putText(frame, ':'.join([str(pass_iteration), str(frame_index)]),
                (0, input_rows - 12), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    common.display_image(frame, show=True)
    video.write(frame)
    
    frame_index += 1
    if frame_index == cap.get(cv2.CAP_PROP_FRAME_COUNT):
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      frame_index = 0
      pass_iteration += 1
      point_label_counts[pass_iteration] = [0, 0, 0]
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
