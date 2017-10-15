#!/usr/bin/python

import argparse
import cv2
import numpy as np

def main():
  # Other useful test: np.reshape(np.arange(5*4*3), (5, 4, 3))

  # The test image is vertical bars of blue, green, red, gray.
  img = np.zeros((4, 4, 3), dtype=np.uint8)

  # Blue
  img[:,:,0] = [[240, 0, 0, 250],
                [220, 0, 0, 230],
                [200, 0, 0, 210],
                [180, 0, 0, 190]]

  # Green
  img[:,:,1] = [[0, 240, 0, 250],
                [0, 220, 0, 230],
                [0, 200, 0, 210],
                [0, 180, 0, 190]]

  # Red
  img[:,:,2] = [[0, 0, 240, 250],
                [0, 0, 220, 230],
                [0, 0, 200, 210],
                [0, 0, 180, 190]]

  def print_by_channel(img):
    rows, cols, channels = img.shape
    for channel in xrange(channels):
      print img[:,:,channel]
  
  # Progressively darker series of imgs.
  num_frames = 5
  frames = np.zeros((num_frames,) + img.shape)  # Prepend a 4th dimension.
  for i in xrange(num_frames):
    frames[i] = img / (i + 1)

  # Manual method to verify against (golden standard but slow).
  def concatenate_frames(frames):
    rows, cols, channels = frames[0].shape
    num_frames = len(frames)
    result = np.zeros((rows, cols * num_frames, channels))
    for i, frame in enumerate(frames):
      result[:, i*cols:(i+1)*cols, :] = frame
    return result


  # Rolling axis 0 rotates the rows, putting the last row (darkest) at the top.
  #img = np.roll(img, 1, 0)
  # Rolling axis 1 rotates the columns, putting the last (white) on the left.
  #img = np.roll(img, 1, 1)
  # Rolling axis 2 rotates the channels.
  #   Before: Blue channel is high in the 1st column, Green in 2nd, Red in 3rd.
  #   After:  Green channel is high in the 1st column, Red in 2nd, Blue in 3rd.
  #print_by_channel(img)
  #print_by_channel(np.roll(img, 1, 2))
  #img = np.roll(img, 1, 2)

  # Test rolling one frame backwards.
  frames = np.roll(frames, -1, 0)
  #concatenated_frames = concatenate_frames(frames)
  #print_by_channel(concatenated_frames)
  cv2.imwrite(args.outfile, concatenate_frames(frames))
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Create a test BMP for inspection.')
  parser.add_argument('--outfile', required=True, type=str, help='Output file path.')
  args = parser.parse_args()
  main()
