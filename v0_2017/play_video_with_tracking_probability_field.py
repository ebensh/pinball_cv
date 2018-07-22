import argparse
from collections import deque
import configparser
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

from lib import common

game_config_path = '/home/monk/src/pinball_cv/configs/hot_hand_full_resolution.cfg'  # For running in Atom
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

# TODO(ebensh): Don't use the percentiles of the counts, instead try things
# like the number of standard deviations from the mean of the distribution
# of frequencies. This may lead to a more accurate handling of valid hotspots.
keypoint_large_count = np.sum(video_masks_large, axis=0, dtype=np.uint32) / 255
noisy_pixels = np.zeros_like(keypoint_large_count)
noisy_pixels[keypoint_large_count > np.percentile(keypoint_large_count, 99)] = 1
common.p_heat(noisy_pixels)

prior = np.copy(keypoint_large_count).astype(np.float32)
prior += 1.0  # Assume an event can happen anywhere with small probability.
prior[noisy_pixels > 0] = 0.1  # Except the noisy pixels, they get even less.
prior = prior / prior.max()  # Normalize to [0, 1]
common.p_heat(prior)

common.p_histogram(prior.flatten())

for ix in range(400, 500):
  past = np.zeros_like(video_masks_large[ix], dtype=np.float32)
  future = np.zeros_like(video_masks_large[ix], dtype=np.float32)
  past += 1.0 * video_masks_large[ix-3] + 3.0 * video_masks_large[ix-2] + 6.0 * video_masks_large[ix-1]
  future += 6.0 * video_masks_large[ix+1] + 3.0 * video_masks_large[ix+2] + 1.0 * video_masks_large[ix+3]
  probability = prior * (past + future + 0.05)
  #probability = cv2.GaussianBlur(probability, (25, 25), 0)
  common.p_heat(np.hstack([past, probability, future]))

  #ix = 100
  to_print = np.copy(video_frames[ix])
  to_print[:,:,0] = cv2.add(to_print[:,:,0], video_masks_large[ix])
  to_print[:,:,1] = cv2.add(to_print[:,:,1], video_masks_large[ix])
  to_print[:,:,2] = cv2.add(to_print[:,:,2], video_masks_large[ix])
  common.p_bgr(to_print)

past = np.zeros_like(video_masks_small[ix], dtype=np.float32)
future = np.zeros_like(video_masks_small[ix], dtype=np.float32)
past += 1.0 * video_masks_small[ix-3] + 3.0 * video_masks_small[ix-2] + 6.0 * video_masks_small[ix-1]
future += 6.0 * video_masks_small[ix+1] + 3.0 * video_masks_small[ix+2] + 1.0 * video_masks_small[ix+3]
probability = prior * (past + future)
#probability = cv2.GaussianBlur(probability, (25, 25), 0)
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
