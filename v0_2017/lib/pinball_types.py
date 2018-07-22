import cv2
import numpy as np

class PinballFrame:
  """Represents a single frame of a pinball video."""
  def __init__(self, ix, img, keypoints):
    self._ix = ix  # The frame's index.
    self._img = img  # A view into the video's color images.
    self._keypoints = keypoints # A list of keypoints in the frame.

  @property
  def ix(self): return self._ix
  @property
  def img(self): return self._img
  @property
  def img_gray(self): return cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
  @property
  def keypoints(self): return self._keypoints
  
class PinballVideo:
  """Represents a pinball video."""
  def __init__(self, imgs, all_keypoints=None):
    self._imgs = imgs  # The raw video images (ndarray, disk mapped?).
    if not all_keypoints:
      all_keypoints = [[]] * len(imgs)
    self._all_keypoints = all_keypoints  # Each image's keypoints (ndarray).
    
    self._frames = []
    for i, (img, keypoints) in enumerate(zip(imgs, all_keypoints)):
      self._frames.append(PinballFrame(i, img, keypoints))
      # Make sure it's a view, not a copy.
      assert self.frames[i].img.base is self._imgs
      assert np.may_share_memory(self.frames[i].img.base, self._imgs)
      assert not self.frames[i].img.flags['OWNDATA']

  @property
  def imgs(self): return self._imgs
  @property
  def all_keypoints(self): return self._all_keypoints
  @property
  def frames(self): return self._frames
  @property
  def num_frames(self): return len(self._frames)

      
class PinballVideoData:
  """A pickle-able object containing all pinball video data."""
  imgs = None
  keypoints = None
