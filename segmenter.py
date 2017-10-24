#!/usr/bin/python

# Splits an input video into several videos of sub-regions, such as the pinball
# playfield and score regions.

import argparse
import ConfigParser
import cv2
import numpy as np


class OutputSegment(object):
  def __init__(self, path, region, rows, cols):
    self._path = path
    self._region = region
    self._rows = rows
    self._cols = cols
    self._mask = OutputSegment._get_region_as_mask(rows, cols, region)
    self._perspective_transform = OutputSegment._get_perspective_transform(rows, cols, region)
    # Note the swap of rows and cols here, as VideoWriter takes (width, height).
    self._video = cv2.VideoWriter(self._path, cv2.VideoWriter_fourcc(*'XVID'),
                                  30.0, (cols, rows))

  def process_frame(self, frame):
    area_of_interest = cv2.bitwise_and(frame, frame, mask=self._mask)
    # Note the swap of rows and cols here, as warpPerspective takes (width, height).
    area_of_interest = cv2.warpPerspective(
        area_of_interest, self._perspective_transform,
        dsize=(self._cols, self._rows))
    self._video.write(area_of_interest)

  def release(self):
    self._video.release()

  @staticmethod
  def _get_region_as_mask(rows, cols, region):
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.fillConvexPoly(mask, region, 255)
    return mask

  @staticmethod
  def _get_perspective_transform(rows, cols, region):
    corners = np.array([
        (0, 0),            # top left
        (cols-1, 0),       # top right
        (cols-1, rows-1),  # bottom right
        (0, rows-1)], dtype=np.float32)  # bottom left
    return cv2.getPerspectiveTransform(region[:-1].astype(np.float32), corners)


def main():
  game_config = ConfigParser.ConfigParser()
  game_config.read(args.game_config)
  input_rows = game_config.getint("InputVideo", "rows")
  input_cols = game_config.getint("InputVideo", "cols")
  cap = cv2.VideoCapture(game_config.get("InputVideo", "path"))

  output_segments = []
  for video in ["PinballFieldVideo", "ScoreBoardVideo"]:
    output_segments.append(OutputSegment(
        path   = game_config.get(video, "path"),
        region = np.array(eval(game_config.get(video, "region"))),
        rows   = game_config.getint(video, "rows"),
        cols   = game_config.getint(video, "cols")))

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
  parser.add_argument('--game_config', required=True, type=str, help='Game configuration file.')
  #parser.add_argument('--display_all_images', default=False, type=bool,
  #                    help='Display all (debug) images.')
  args = parser.parse_args()
  main()

