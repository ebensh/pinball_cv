from collections import namedtuple
import cv2
import inspect
import numpy as np

# http://ipython-books.github.io/featured-01/
def get_data_base(arr):
  """For a given Numpy array, finds the
    base array that "owns" the actual data."""
  base = arr
  while isinstance(base.base, np.ndarray):
    base = base.base
  return base

def arrays_share_data(x, y): return get_data_base(x) is get_data_base(y)


def print_by_channel(img):
  rows, cols, channels = img.shape
  for channel in xrange(channels):
    print img[:,:,channel]

def display_image(img, title=None, show=True):
  if show:
    if title is None:
      # Get the caller's line number so we can identify which point in the
      # process we're at without uniquely naming each one.
      frame, filename, line_num, function_name, lines, index = inspect.stack()[1]
      title = "{0}:{1}".format(filename, line_num)
    cv2.imshow(title, img)


# This is *INEFFICIENT* and is only intended for quick experimentation.
# http://blog.hackerearth.com/descriptive-statistics-with-Python-NumPy
# TODO(ebensh): Add a wrapper class around the named tuple.
NamedStatistics = namedtuple('NamedStatistics', ['minimum', 'maximum', 'ptp', 'mean'])
def get_named_statistics(frames):
  minimum = np.amin(frames, axis=0)
  maximum = np.amax(frames, axis=0)
  return NamedStatistics(
    minimum=minimum,
    maximum=maximum,
    ptp=maximum - minimum,
    mean=cv2.convertScaleAbs(np.mean(frames, axis=0, dtype=np.float64)))
    #median=cv2.convertScaleAbs(np.median(frames, axis=0)),
    #
    #variance=cv2.convertScaleAbs(np.var(frames, axis=0, dtype=np.float64)))

def print_statistics(statistics, printer):
  for field in statistics._fields:
    printer.add_image(getattr(statistics, field), field)
    
    
class FrameBuffer(object):
  def __init__(self, num_frames=1, shape=(640, 480, 3), dtype=np.uint8):
    # Create our frame buffers. We don't store them together because while it
    # would make the rolling easier it would also require the gray version to
    # be stored with three channels.
    self._BUFFER_LENGTH = 2 * num_frames  # Left here in case we want to increase.
    self._num_frames = num_frames
    self._idx = 0
    self._shape = shape
    self._frames = np.zeros((self._BUFFER_LENGTH,) + shape, dtype=dtype)
    self._frames_gray = np.zeros((self._BUFFER_LENGTH,) + shape[0:2], dtype=dtype)

  def append(self, frame):
    idx_to_insert = (self._idx + self._num_frames) % self._BUFFER_LENGTH
    self._frames[idx_to_insert] = frame
    self._frames_gray[idx_to_insert] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    self._idx = (self._idx + 1) % self._BUFFER_LENGTH

  def get_view(self, start, stop, color=True):
    view = None
    if start is None: start = 0
    if stop is None: stop = self._num_frames
    start += self._idx
    stop += self._idx
    if color:
      view = self._frames.take(range(start, stop), axis=0, mode='wrap').view()
    else:
      view = self._frames_gray.take(range(start, stop), axis=0, mode='wrap').view()
    view.setflags(write=False)
    return view

  def get_shape(self, color=True):
    if color: return self._shape
    return self._shape[0:2]

  # Useful for debugging.
  def get_buffers(self):
    return cv2.hconcat(self._frames), cv2.hconcat(self._frames_gray)
  

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

    combined_image = np.zeros((rows, total_cols, 3), dtype=np.uint8)
    current_col = 0
    for img, caption in self._images:
      shape = img.shape
      rows, cols = shape[0], shape[1]
      combined_image[0:rows, current_col:current_col+cols] = img
      cv2.putText(combined_image, caption, (current_col, rows), font,
                  1, (255,255,255), 2, cv2.LINE_AA)
      current_col += cols + space

    return combined_image
      
    

