import cv2
import inspect
import numpy as np

def display_image(img, title=None, show=True):
  if show:
    if title is None:
      # Get the caller's line number so we can identify which point in the
      # process we're at without uniquely naming each one.
      frame, filename, line_num, function_name, lines, index = inspect.stack()[1]
      title = "{0}:{1}".format(filename, line_num)
    cv2.imshow(title, img)

class FramePrinter(object):
  def __init__(self):
    self._images = []

  def add_image(self, img, caption):
    if len(img.shape) < 3 or img.shape[2] != 3:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    self._images.append((img, caption))

  def get_combined_image(self):
    font = cv2.FONT_HERSHEY_SIMPLEX
    space = 10  # pixels between images

    max_rows = 0
    total_cols = 0
    for img, _ in self._images:
      shape = img.shape
      rows, cols = shape[0], shape[1]
      max_rows = max(max_rows, rows)
      total_cols += cols
    total_cols += (len(self._images) - 1) * space

    combined_image = np.zeros((rows + 30, total_cols, 3), dtype=np.uint8)
    current_col = 0
    for img, caption in self._images:
      shape = img.shape
      rows, cols = shape[0], shape[1]
      combined_image[0:rows, current_col:current_col+cols] = img
      cv2.putText(combined_image, caption, (current_col, rows), font,
                  1, (255,255,255), 2, cv2.LINE_AA)
      current_col += cols + space

    return combined_image
      
    

