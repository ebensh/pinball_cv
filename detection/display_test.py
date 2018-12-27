#!/usr/bin/env python3

import cv2
from matplotlib import pyplot as plt
import numpy as np

import common
from display import Display
import pinball_types

# Useful references:
# https://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/

mouse_info = None
# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
def handle_click(event, x, y, flags, params):
  global mouse_info
  if event == cv2.EVENT_LBUTTONDOWN and not handle_click.start_position:
    handle_click.start_position = [x, y]
    print("Mouse clicked:", handle_click.start_position)
  elif event == cv2.EVENT_LBUTTONUP and handle_click.start_position:
    mouse_info = [params, handle_click.start_position, [x, y]]
    handle_click.start_position = None
    print("Mouse clicked:", mouse_info)
handle_click.start_position = None
  

def main():
  global mouse_info
  display = Display({
    'original': {},
    'gray': {'show': False},
    'background': {},
    'processed': {},
    'mask': {'callback': handle_click},
    'histogram': {},
  })

  path = 'videos/2017_11_29_hot_hand_mike/pinball_field_video.avi'
  video = pinball_types.PinballVideo(common.get_all_frames_from_video(path), all_keypoints=None)

  while True:
    for frame in video.frames: #[33:48+1]:
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

      mask = np.uint8(255) * np.any(processed > 25, axis=2)
      print(mask.dtype, np.sum(mask))
      display.Add('mask', mask)

      while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
          cv2.destroyAllWindows()
          return
        elif key == ord('n'):
          break
        elif mouse_info:
          window, [start_x, start_y], [end_x, end_y] = mouse_info
          print("Draw a histogram for the box! ", mouse_info)
          mouse_info = None

          # Apply the mask to the original image.
          masked = cv2.bitwise_and(frame.img, frame.img, mask=mask)
          # Clip down to just the ROI.
          masked = masked[start_x:end_x, start_y:end_y, :]

          plt.figure()
          plt.title("'Flattened' Color Histogram")
          plt.xlabel("Bins")
          plt.ylabel("# of Pixels")
          features = []
          for channel, color in zip(range(3), "bgr"):
            # Take a histogram of the ROI of the masked image.
            hist = cv2.calcHist([masked[:,:,channel]], [0], None, [256], [0, 255])
            # Plot it.
            print("Plotting histogram for channel ", channel)
            plt.plot(hist, color = color)
            plt.xlim([0, 255])
          plt.plot()
          plt.show()



  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
