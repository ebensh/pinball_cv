#!/usr/bin/env python3

# Tracks all pinball-ish blobs across a pinball playfield video, outputting a
# json dictionary of frame number -> keypoint triples (x, y, radius).

import argparse
import configparser
import cv2
import itertools
import json
import numpy as np

import common
from display import Display
import pinball_types

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
    self._frame_to_keypoints[frame_index] = [
      (kp.pt[0], kp.pt[1], kp.size) for kp in keypoints]
    print(self._frame_to_keypoints[frame_index])

  def release(self):
    # Write the json keypoints.
    with open(self._path, 'w') as keypoints_file:
      json.dump(self._frame_to_keypoints, keypoints_file)


def main():
  game_config = configparser.ConfigParser()
  game_config.read(args.game_config)
  input_rows = game_config.getint('PinballFieldVideo', 'rows')
  input_cols = game_config.getint('PinballFieldVideo', 'cols')
  cap = cv2.VideoCapture(game_config.get('PinballFieldVideo', 'path'))
  
  keypoints_path = game_config.get('PinballFieldVideo', 'keypoints_path')
  keypoint_detector = OutputSegment(keypoints_path)

  video = pinball_types.PinballVideo(common.get_all_frames_from_video(
      game_config.get('PinballFieldVideo', 'path')), all_keypoints=None)

  display = Display({
    'original': True and args.display_all_images,
    'combined': True and args.display_all_images,
    'past_stats': True and args.display_all_images,
    'masks': True and args.display_all_images
  })

  for frame in video.frames:
    display.Clear()
    display.Add('original', frame.img)
    lookback, lookahead = 2, 2
    # Here we want an ndarray of the past and future in color and gray.
    # Using simple slicing will return views.
    # ndarray of 2 x rows x cols x 3

    # TODO: More elegantly handle the ix = 1 case, where past and future
    # are not the same size (1 frame versus 2), making concatenation hard.
    past_gray = None
    if frame.ix > 0:
      start = max(frame.ix - lookback, 0)
      end = frame.ix
      past_gray = video.imgs_gray[start:end]
    else:
      past_gray = np.zeros_like(video.imgs_gray[frame.ix:frame.ix+1])
    past_stats = common.get_named_statistics(past_gray)
    #print(past_stats)

    future_gray = None
    if frame.ix < video.num_frames - 1:
      start = frame.ix + 1
      end = min(frame.ix + 1 + lookahead, video.num_frames)
      future_gray = video.imgs_gray[start:end]
    else:
      future_gray = np.zeros_like(video.imgs_gray[frame.ix:frame.ix+1])
    future_stats = common.get_named_statistics(future_gray)
    #print(future_stats)
      
    # TODO: Once the size discrepancy is handled remove this check :)
    if past_gray.shape == future_gray.shape:
      display.Add('combined', 
          np.concatenate([np.concatenate(past_gray, axis=1),
                          np.concatenate(future_gray, axis=1)],
                          axis=0))

    past_stats_printer = common.FramePrinter()
    common.print_statistics(past_stats, past_stats_printer)
    display.Add('past_stats', past_stats_printer.get_combined_image())
    
    # Subtract out the unchanging background (mean past, mean future) from current frame.
    foreground_mask_past = cv2.absdiff(frame.img_gray, past_stats.mean)
    foreground_mask_future = cv2.absdiff(frame.img_gray, future_stats.mean)

    # Threshold the differences and combine them.
    foreground_mask = np.logical_and(foreground_mask_past >= 25, foreground_mask_future >= 25)

    # Mask away the areas we know are changing based on thresholded ptp (ptp past, ptp future).
    # Take the absolute difference (per pixel) from the mean in each frame.
    #changing_mask_past = np.absolute(past_gray.astype(np.int16) - past_stats.mean.astype(np.int16)) >= 5
    # Count how many frames were significantly different and threshold.
    #changing_mask_past = np.sum(changing_mask_past, axis=0) >= 3
    
    # Take the absolute difference (per pixel) from the mean in each frame.
    #changing_mask_future = np.absolute(future_gray.astype(np.int16) - future_stats.mean.astype(np.int16)) >= 5
    # Count how many frames were significantly different.
    #changing_mask_future = np.sum(changing_mask_future, axis=0) >= 3

    #changing_mask = np.logical_or(changing_mask_past, changing_mask_future)
    changing_mask = np.zeros_like(frame.img_gray)

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
    
    display.Add('masks', np.hstack([
        np.uint8(255) * foreground_mask,
        np.uint8(255) * changing_mask,
        final_mask,
        final_mask_polished]))

    keypoint_detector.process_frame(cv2.bitwise_not(final_mask_polished), frame.ix)  # Invert for blob detection.
    print("Processed frame:", frame.ix)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

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
