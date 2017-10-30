#!/usr/bin/python

# Uses ssocr (for now) to read each frame of a scoreboard video, outputting a
# pickled dictionary of frame number to score read by ssocr.

import argparse
import common
import ConfigParser
import cv2
import numpy as np
import os


class OutputSegment(object):
  def __init__(self, path, region, input_rows, input_cols):
    self._path = path
    print path
    self._region = region
    self._x, self._y, self._cols, self._rows = cv2.boundingRect(region)
    # Note the swap of rows and cols here, as VideoWriter takes (width, height).
    self._video = cv2.VideoWriter(self._path, cv2.VideoWriter_fourcc(*'XVID'),
                                  30.0, (self._cols, self._rows))

  def process_frame(self, frame):
    area_of_interest = frame[self._y:self._y + self._rows,
                             self._x:self._x + self._cols, :]
    #print area_of_interest.shape, area_of_interest.dtype
    self._video.write(area_of_interest)

  def release(self):
    self._video.release()


def main():
  game_config = ConfigParser.ConfigParser()
  game_config.read(args.game_config)
  
  input_rows = game_config.getint("ScoreBoardVideo", "rows")
  input_cols = game_config.getint("ScoreBoardVideo", "cols")
  relative_regions = eval(game_config.get("ScoreBoardVideo", "relative_regions"))
  path = game_config.get("ScoreBoardVideo", "path")
  path_base, path_ext = os.path.splitext(path)

  cap = cv2.VideoCapture(path)
  
  output_segments = []
  for player, region in relative_regions:
    path = path_base + "_" + player + path_ext
    output_segment = OutputSegment(
      path = path,
      region = np.array(region),
      input_rows = input_rows,
      input_cols = input_cols)
    output_segments.append(output_segment)

  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break

    for segment in output_segments:
      segment.process_frame(raw_frame)

  cap.release()
  for segment in output_segments:
    segment.release()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Read score per frame from video.')
  parser.add_argument('--game_config', required=True, type=str, help='Game configuration file.')
  args = parser.parse_args()
  main()

