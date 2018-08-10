#!/usr/bin/env python3

import argparse
import configparser
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

import common

WINDOW_COLOR_FRAME = 'ColorFrame'
WINDOW_KEYPOINTS = 'Keypoints'
WINDOW_COMBINED = 'Combined'

mouse_info = None
# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
def handle_click(event, x, y, flags, params):
  global mouse_info

  if event == cv2.EVENT_LBUTTONDOWN:
    mouse_info = [params, [x, y]]
    print("Mouse clicked:", mouse_info)


def main():
  global mouse_info
  game_config = configparser.ConfigParser()
  game_config.read(args.game_config)
  input_rows = game_config.getint('PinballFieldVideo', 'rows')
  input_cols = game_config.getint('PinballFieldVideo', 'cols')

  # A standard pinball is 17/16", a standard playfield is 20.25", so the ratio
  # between them is ~5%. Assuming we've cropped the playfield correctly, the
  # number of columns should represent 20.25" in pixels.
  PINBALL_RADIUS_PX = int(17.0/16 / 20.25 * input_cols / 2)

  video_frames = common.get_all_frames_from_video(
      game_config.get('PinballFieldVideo', 'path'))
  frame_to_keypoints_list = common.load_json_keypoints_as_list(
      game_config.get('PinballFieldVideo', 'keypoints_path'))
  if not frame_to_keypoints_list:
    print("No keypoints found, starting from scratch.")
    frame_to_keypoints_list = [[] for _ in video_frames]
  keypoint_circles = common.get_all_keypoint_masks(
      input_rows, input_cols, frame_to_keypoints_list, PINBALL_RADIUS_PX, 1)

  # Display our windows and set up their callbacks.
  common.display_image(video_frames[0], title=WINDOW_COLOR_FRAME)
  cv2.setMouseCallback(WINDOW_COLOR_FRAME, handle_click, WINDOW_COLOR_FRAME)
  #if keypoint_circles:
  common.display_image(keypoint_circles[0], title=WINDOW_KEYPOINTS)
  common.display_image(common.add_bgr_and_gray(video_frames[0], keypoint_circles[0]), title=WINDOW_COMBINED)
  #else:
  #  common.display_image(np.zeros_like(video_frames[0]), title=WINDOW_KEYPOINTS)
  #  common.display_image(np.zeros_like(video_frames[0]), title=WINDOW_COMBINED)
  # No callback on the keypoints only window, they're just to help see.
  cv2.setMouseCallback(WINDOW_COMBINED, handle_click, WINDOW_COMBINED)

  # TODO: Combine the keypoints and golden keypoints into one structured file,
  # ideally a sort of per-frame object that contains arbitrary information
  # about it. Then we can wrap each frame in a python object, providing utility
  # methods on a per-frame basis.
  # If the golden keypoints file already exists, load it.
  try:
    with open(game_config.get('PinballFieldVideo', 'keypoints_golden_path'), 'r') as golden_keypoints_file:
      golden_keypoints = json.load(golden_keypoints_file)
    last_frame_processed = None
    for i, keypoint in enumerate(reversed(golden_keypoints)):
      if keypoint:
        last_frame_processed = len(golden_keypoints) - i - 1
        print("The last frame processed is", last_frame_processed)
        break
  except IOError: 
    print("No golden keypoints found, starting from scratch.")
    golden_keypoints = [[]] * len(video_frames)
    last_frame_processed = -1
    
  i = last_frame_processed + 1
  while i < len(video_frames):
    video_frame = video_frames[i]
    keypoint_circle = keypoint_circles[i]
    keypoints_in_frame = frame_to_keypoints_list[i]
    golden_keypoint = golden_keypoints[i]
    if golden_keypoint:
      print("Frame", i, "has golden keypoint", golden_keypoint)
    else:
      print("Frame", i, "needs a keypoint")

    common.display_image(video_frame, title=WINDOW_COLOR_FRAME)
    common.display_image(keypoint_circle, title=WINDOW_KEYPOINTS)
    common.display_image(common.add_bgr_and_gray(video_frame, keypoint_circle),
                         title=WINDOW_COMBINED)
    
    print("Waiting for key or click...")
    while True:
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
        with open(game_config.get('PinballFieldVideo', 'keypoints_golden_path'), 'w') as golden_keypoints_file:
          json.dump(golden_keypoints, golden_keypoints_file)
        cv2.destroyAllWindows()
        return 0
      elif key == ord('n'):
        break  # The while loop, continue with next frame
      elif key == ord('p'):
        i -= 2
        break  # The while loop, continue with previous frame
      elif mouse_info:
        window, mouse_xy = mouse_info
        mouse_info = None

        if window == WINDOW_COMBINED:
          selected_keypoint = None
          for (x, y, size) in keypoints_in_frame:
            if common.dist(mouse_xy, (x, y)) < 5 * PINBALL_RADIUS_PX:
              selected_keypoint = [x, y, size]
          if selected_keypoint:
            print("Setting frame", i, "golden keypoint:", selected_keypoint)
            golden_keypoints[i] = selected_keypoint
            break  # Go to the next frame
          else:
            # Must've been a missed click / not close enough.
            print("No keypoint detected close by, please try again.")
        elif window == WINDOW_COLOR_FRAME:
          golden_keypoints[i] = mouse_xy + [PINBALL_RADIUS_PX]
          print("Setting frame", i, "golden keypoint:", golden_keypoints[i])
          break
    i += 1
  print("All done! :)")
  with open(game_config.get('PinballFieldVideo', 'keypoints_golden_path'), 'w') as golden_keypoints_file:
    json.dump(golden_keypoints, golden_keypoints_file)
  cv2.destroyAllWindows()
  return 0
        

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Play video with tracking overlay.')
  parser.add_argument('--game_config', required=True, type=str, help='Game configuration file.')
  parser.add_argument('--display_all_images', default=False, type=bool,
                      help='Display all (debug) images.')
  args = parser.parse_args()
  main()
