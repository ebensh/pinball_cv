#!/usr/bin/env python3

import cv2
import numpy as np

import common
from display import Display
import pinball_types

def main():
  display = Display({
    'original': True,
    'gray': False,
    'background': True,
    'processed': True,
    'mask': True,
  })

  path = 'videos/2017_11_29_hot_hand_mike/pinball_field_video.avi'
  video = pinball_types.PinballVideo(common.get_all_frames_from_video(path), all_keypoints=None)

  for frame in video.frames:
    print(frame.ix)
    display.Clear()
    display.Add('original', frame.img)
    display.Add('gray', frame.img_gray)

    background = np.zeros_like(frame.img, dtype=np.float32)
    alpha = 0.9
    for ix in range(max(frame.ix - 5, 0), max(frame.ix - 1, 0)):
      cv2.accumulateWeighted(video.frames[ix].img, background, alpha)
      print(np.min(background), np.max(background))
    background = cv2.convertScaleAbs(background)
    print(np.min(background), np.max(background))
    display.Add('background', background)

    processed = cv2.absdiff(frame.img, background)
    print(np.min(processed), np.max(processed), np.sum(frame.img-background))
    display.Add('processed', processed)

    mask = np.uint8(255) * (processed > 25)
    print(mask.dtype, np.sum(mask))
    display.Add('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
