#!/usr/bin/env python3

# To use this script:
# pinball_cv/> python3 -m detection.tools.segmenter
# This is because python package paths are *weird*.

# Splits an input video into several videos of sub-regions, such as the pinball
# playfield and score regions.

import argparse
import configparser
import cv2
import logging
import numpy as np

import common

class OutputSegment(object):
  def __init__(self, path, region, input_rows, input_cols, output_rows, output_cols):
    self._path = path
    self._region = region
    self._rows = output_rows
    self._cols = output_cols
    self._mask = common.get_region_as_mask(input_rows, input_cols, region)
    self._perspective_transform = common.get_perspective_transform(output_rows, output_cols, region)
    # Note the swap of rows and cols here, as VideoWriter takes (width, height).
    self._video = cv2.VideoWriter(self._path, cv2.VideoWriter_fourcc(*'X264'),
                                  30.0, (output_cols, output_rows))

  def process_frame(self, frame):
    area_of_interest = cv2.bitwise_and(frame, frame, mask=self._mask)
    # Note the swap of rows and cols here, as warpPerspective takes (width, height).
    area_of_interest = cv2.warpPerspective(
        area_of_interest, self._perspective_transform,
        dsize=(self._cols, self._rows))
    self._video.write(area_of_interest)

  def release(self):
    self._video.release()


def main():
  game_config = configparser.ConfigParser()
  game_config.read(args.game_config)
  input_rows = game_config.getint("InputVideo", "rows")
  input_cols = game_config.getint("InputVideo", "cols")
  logging.info("Processing: %s", game_config.get("InputVideo", "path"))
  cap = cv2.VideoCapture(game_config.get("InputVideo", "path"))
  if not cap.isOpened():
    logging.error("Could not open input file! Exiting.")
    return

  output_segments = []
  for video in ["PinballFieldVideo", "ScoreBoardVideo"]:
    if not game_config.has_section(video): continue
    segment = OutputSegment(
        path        = game_config.get(video, "path"),
        region      = np.array(eval(game_config.get(video, "region"))),
        input_rows  = input_rows,
        input_cols  = input_cols,
        output_rows = game_config.getint(video, "rows"),
        output_cols = game_config.getint(video, "cols"))
    output_segments.append(segment)

  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break

    for segment in output_segments:
      segment.process_frame(raw_frame)

  cap.release()
  for segment in output_segments:
    segment.release()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Split input video by region.')
  parser.add_argument('--log', required=False, default='INFO', type=str, help='Logging level to print, e.g. INFO.')
  parser.add_argument('--game_config', required=True, type=str, help='Game configuration file.')
  #parser.add_argument('--display_all_images', default=False, type=bool,
  #                    help='Display all (debug) images.')
  args = parser.parse_args()

  if args.log:
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)
  main()

