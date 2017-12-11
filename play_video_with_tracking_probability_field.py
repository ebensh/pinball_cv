import argparse
from collections import deque
import configparser
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

from lib import common

game_config_path = '/home/ebensh/src/pinball_cv/configs/hot_hand_full_resolution.cfg'  # For running in Atom
path_base = 'hot_hand_ebensh_2017_11_27'
game_config = configparser.ConfigParser(defaults={path_base: path_base})
game_config.read(game_config_path)
input_rows = game_config.getint('PinballFieldVideo', 'rows')
input_cols = game_config.getint('PinballFieldVideo', 'cols')

# A standard pinball is 17/16", a standard playfield is 20.25", so the ratio
# between them is ~5%. Assuming we've cropped the playfield correctly, the
# number of columns should represent 20.25" in pixels.
PINBALL_RADIUS_PX = int(17.0/16 / 20.25 * input_cols)

video_frames = common.get_all_frames_from_video(
    game_config.get('PinballFieldVideo', 'path'))
frame_to_keypoints = common.load_json_keypoints_as_dict(
    game_config.get('PinballFieldVideo', 'keypoints_path'))
video_masks_small = common.get_all_keypoint_masks(input_rows, input_cols,
                                                  frame_to_keypoints,
                                                  fixed_radius=1)
video_masks_large = common.get_all_keypoint_masks(input_rows, input_cols,
                                                  frame_to_keypoints,
                                                  fixed_radius=5)
#common.p_bgr(video_frames[100])
heat_mask = np.sum(video_masks_large, axis=0, dtype=np.uint32)
# We want low-frequency points to count for a lot, and high-frequency points
# to count for a little, so we do a log of the reversed frequency.
heat_mask = np.log(heat_mask.max() - heat_mask + 1)
heat_mask = heat_mask / heat_mask.max()  # Normalize to 0, 1
common.p_heat(heat_mask) #, '/var/tmp/heat_map_1.png')
heat_mask_weighted = np.sum(video_masks_large * heat_mask, axis=0)
#common.p_heat(heat_mask_weighted)
heat_mask_weighted = np.log(heat_mask_weighted * 10 + 1)
common.p_heat(heat_mask_weighted) # , '/var/tmp/heat_map_weighted_1.png')

ix = 465
past = np.zeros_like(video_masks_small[ix], dtype=np.float32)
future = np.zeros_like(video_masks_small[ix], dtype=np.float32)
past += 1.0 * video_masks_small[ix-3] + 3.0 * video_masks_small[ix-2] + 6.0 * video_masks_small[ix-1]
future += 6.0 * video_masks_small[ix+1] + 3.0 * video_masks_small[ix+2] + 1.0 * video_masks_small[ix+3]
probability = np.full_like(video_masks_small[ix], 1.0 / video_masks_small[ix].size,
                           dtype=np.float32) + past + future
probability = cv2.GaussianBlur(probability, (25, 25), 0)
common.p_heat(np.hstack([past, probability, future]))
common.p_bgr(video_frames[ix])
common.p_heat(1.0 * video_masks_small[ix] * probability * heat_mask)
for x, y, size in frame_to_keypoints[ix]:
  #probability[x, y] =
  print(x, y, size, probability[y, x])

common.p_gray(common.hconcat_frames(video_masks[ix-1:ix+2]))


#common.p_bgr(video_frames[ix])
NUM_FRAMES = len(video_frames)
# TODO: vectorize this later.
