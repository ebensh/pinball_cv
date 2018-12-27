from collections import defaultdict
import cv2
import numpy as np


class Display:
  '''Display is a class that handles displaying several images to a single
  OpenCV window. Each key is a separate window, and each image will be tiled
  within the window.'''
  _windows = {}  # Dict from window name to is_enabled
  _images = defaultdict(list)  # Dict from window name to images

  def __init__(self, windows):
    '''Takes a dictionary of windows from name to properties.'''
    self._windows = windows
    self.InitializeWindows()

  def InitializeWindows(self):
    default_image = np.zeros([50, 100])
    for window, properties in self._windows.items():
      if not properties.get("show", True):
        continue

      cv2.imshow(window, default_image)
      callback = properties.get("callback")
      if callback:
        cv2.setMouseCallback(window, callback, window)

  def Clear(self):
    self._images = defaultdict(list)

  def Add(self, window, image):
    assert window in self._windows

    if self._windows[window].get("show", True):
      self._images[window].append(image)
      #image = np.concatenate(self._images[window], axis=1)  # Horizontally
      cv2.imshow(window, self._images[window][0])