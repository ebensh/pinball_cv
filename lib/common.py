#!/usr/bin/env python3

from collections import namedtuple
import cv2
import inspect
import json
import matplotlib.pyplot as plt
import numpy as np

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)
def numpy_print_cols(cols=160):
  np.core.arrayprint._line_width = cols

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
    print(img[:,:,channel])

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
#NamedStatistics = namedtuple('NamedStatistics', ['minimum', 'maximum', 'ptp', 'mean'])
NamedStatistics = namedtuple('NamedStatistics', ['mean'])
def get_named_statistics(frames):
  #minimum = np.amin(frames, axis=0)
  #maximum = np.amax(frames, axis=0)
  return NamedStatistics(
    #minimum=minimum,
    #maximum=maximum,
    #ptp=maximum - minimum,
    mean=np.mean(frames, axis=0, dtype=np.float64).astype(np.uint8))
    #median=cv2.convertScaleAbs(np.median(frames, axis=0)),
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



def get_region_as_mask(rows, cols, region):
  mask = np.zeros((rows, cols), dtype=np.uint8)
  cv2.fillConvexPoly(mask, region, 255)
  return mask

def get_perspective_transform(rows, cols, region):
  corners = np.array([
    (0, 0),            # top left
    (cols-1, 0),       # top right
    (cols-1, rows-1),  # bottom right
    (0, rows-1)], dtype=np.float32)  # bottom left
  return cv2.getPerspectiveTransform(region[:-1].astype(np.float32), corners)

# IMPORTANT!!! Subtraction will WRAP with uint8 if it goes negative!
def trim_to_uint8(arr): return np.clip(arr, 0, 255).astype(np.uint8)


def extrapolate(xy1, xy2):
  x1, y1 = xy1
  x2, y2 = xy2
  vx = x2 - x1
  vy = y2 - y1
  return (x2 + vx, y2 + vy)
def lerp(xy1, xy2):
  x1, y1 = xy1
  x2, y2 = xy2
  return ((x1 + x2) / 2, (y1 + y2) / 2)
def dist(xy1, xy2):
  x1, y1 = xy1
  x2, y2 = xy2
  return (x2 - x1)**2 + (y2 - y1)**2
def in_bounds(rows, cols, xy):
  x, y = xy
  return (x >= 0 and x < cols and y >= 0 and y < rows)

# https://matplotlib.org/users/image_tutorial.html
# http://jakevdp.github.io/mpl_tutorial/tutorial_pages/tut2.html
def p_gray(*args, path=None):
  imgs = list(args)
  #plt.figure(figsize=(20,10))
  fig, axs = plt.subplots(1, len(imgs), squeeze=False)
  fig.set_size_inches(20, 10)
  for img, ax in zip(imgs, axs[0]):
    ax.imshow(img, cmap = 'gray')
  if path: plt.savefig(path, bbox_inches='tight')
  plt.show()
def p_bgr(img, path=None):
  plt.figure(figsize=(20,10))
  plt.imshow(img[:,:,::-1])
  if path: plt.savefig(path, bbox_inches='tight')
  plt.show()
def p_heat(img, path=None):
  plt.figure(figsize=(20,10))
  plt.imshow(1.0 * img / img.max(), cmap='inferno', interpolation='nearest')
  if path: plt.savefig(path, bbox_inches='tight')
  plt.show()
def p_histogram(img, path=None):
  plt.figure(figsize=(6, 3))
  plt.hist(img, bins=32)
  if path: plt.savefig(path, bbox_inches='tight')
  plt.show()

def load_json_keypoints_as_dict(path):
  with open(path, 'r') as keypoints_file:
    frame_to_keypoints_str = json.load(keypoints_file)
  frame_to_keypoints = {}
  for frame_index_str, keypoints_str in frame_to_keypoints_str.items():
    frame_to_keypoints[int(frame_index_str)] = [
        [int(round(x)), int(round(y)), int(round(size))]
        for x, y, size in keypoints_str]
  assert set(frame_to_keypoints.keys()) == set(range(len(frame_to_keypoints)))
  return frame_to_keypoints

def get_all_frames_from_video(path):
  cap = cv2.VideoCapture(path)
  video_frames = []
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    video_frames.append(raw_frame)
  cap.release()
  return np.array(video_frames)

def keypoints_to_mask(rows, cols, keypoints, fixed_radius=None):
  mask = np.zeros([rows, cols], np.uint8)
  for x, y, size in keypoints:
    if fixed_radius: size = fixed_radius
    if size == 1: mask[y, x] = 255
    else: cv2.circle(mask, (x, y), size, color=255, thickness=-1)
  return mask

def get_all_keypoint_masks(rows, cols, frame_to_keypoints, fixed_radius=None):
  video_masks = []
  for frame_index in sorted(frame_to_keypoints.keys()):
    video_masks.append(keypoints_to_mask(rows, cols,
                                         frame_to_keypoints[frame_index],
                                         fixed_radius))
  return np.array(video_masks)

def hconcat_frames(frames):
  num_frames, rows, cols = frames.shape[:3]
  return frames.swapaxes(0, 1).reshape([rows, num_frames * cols])
