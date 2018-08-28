#!/usr/bin/env python3

# Given a game config will return a score for the tracker's output keypoints.
# The score returned is the percentage of frames that had the golden keypoint
# as one of the possible keypoints.
# Note that this can be used to score a "selected" keypoint file by removing
# all non-selected keypoints.

import argparse
import configparser
import cv2
import itertools
import json
import numpy as np

import common
import pinball_types


def main():
  game_config = configparser.ConfigParser()
  game_config.read(args.game_config)
  input_rows = game_config.getint('PinballFieldVideo', 'rows')
  input_cols = game_config.getint('PinballFieldVideo', 'cols')
  
  keypoints_path = game_config.get('PinballFieldVideo', 'keypoints_path')
  golden_keypoints_path = game_config.get('PinballFieldVideo', 'keypoints_golden_path')

  keypoints = common.load_json_keypoints_as_list(keypoints_path)
  with open(golden_keypoints_path, 'r') as golden_keypoints_file:
    golden_keypoints = json.load(golden_keypoints_file)

  video = pinball_types.PinballVideo(common.get_all_frames_from_video(
      game_config.get('PinballFieldVideo', 'path')),
      all_keypoints=keypoints, all_golden_keypoints=golden_keypoints)

  # Correctly identified *before any other keypoints suggested*.
  score_correct_at_head = 0.0
  # Correctly identified regardless of order.
  score_correct = 0.0
  
  if args.display_all_images:
    cv2.namedWindow('frame_with_keypoints', cv2.WINDOW_NORMAL)

  for frame in video.frames:
    is_golden_keypoint = [kp in frame.golden_keypoints
                          for kp in frame.keypoints]
    num_correct_at_head = len(list(itertools.takewhile(lambda x: x == True,
                                                       is_golden_keypoint)))
    num_correct = sum(is_golden_keypoint)

    score_correct_at_head += float(num_correct_at_head) / len(frame.golden_keypoints)
    score_correct += float(num_correct) / len(frame.golden_keypoints)    

    if args.display_all_images:
      print("Frame {0}: {1}/{3} at head, {2}/{3} anywhere".format(
        frame.ix, num_correct_at_head, num_correct, len(frame.golden_keypoints)))
    
      img = frame.img.copy()

      keypoints_mask = common.keypoints_to_mask(input_rows, input_cols, frame.keypoints)
      golden_keypoints_mask = common.keypoints_to_mask(input_rows, input_cols, frame.golden_keypoints)
      img = common.draw_colorized_mask(img, keypoints_mask, (255, 0, 255))
      img = common.draw_colorized_mask(img, golden_keypoints_mask, (0, 255, 255))

      common.display_image(img, 'frame_with_keypoints', args.display_all_images)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cv2.destroyAllWindows()

  if video.num_frames > 0:
    score_correct_at_head = 100.0 * score_correct_at_head / video.num_frames
    score_correct = 100.0 * score_correct / video.num_frames
  print("Final score: {0:.2f}% at head, {1:.2f}% anywhere".format(
      score_correct_at_head, score_correct))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Score a keypoints file.')
  parser.add_argument('--game_config', required=True, type=str, help='Game configuration file.')
  parser.add_argument('--display_all_images', default=False, type=bool,
                      help='Display all (debug) images.')
  args = parser.parse_args()
  main()
