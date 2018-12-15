import cv2
import numpy as np

class Display:
  '''Display is a class that handles displaying several images to a single
  OpenCV window. Each key is a separate window, and each image will be tiled
  within the window.'''
  _windows = {}  # Dict from window name to is_enabled
  _images = {}  # Dict from window name to images

  def __init__(self, windows):
    '''Takes a dictionary of windows from name to bool of enabled.'''
    self._windows = windows
    self.Clear()

  def Clear(self):
    for window, is_enabled in self._windows.items():
      if is_enabled:
        self._images[window] = []

  def Add(self, window, image):
    assert window in self._windows

    if self._windows[window]:
      self._images[window].append(image)
      #image = np.concatenate(self._images[window], axis=1)  # Horizontally
      cv2.imshow(window, self._images[window][0])