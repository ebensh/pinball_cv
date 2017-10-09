import cv2
import inspect

def display_image(img, title=None, show=True):
  if show:
    if title is None:
      # Get the caller's line number so we can identify which point in the
      # process we're at without uniquely naming each one.
      frame, filename, line_num, function_name, lines, index = inspect.stack()[1]
      title = "{0}:{1}".format(filename, line_num)
    cv2.imshow(title, img)
