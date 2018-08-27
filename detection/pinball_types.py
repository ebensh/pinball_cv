import cv2
import numpy as np
import common

# TODO: Implement a GameConfig abstraction

class PinballFrame:
  """Represents a single frame of a pinball video."""
  def __init__(self, ix, img, img_gray, keypoints, golden_keypoints):
    self._ix = ix  # The frame's index.
    self._img = img  # A view into the video's color images.
    self._img_gray = img_gray  # A view into the video's grayscale images.
    self._keypoints = keypoints  # A list of keypoints in the frame.
    self._golden_keypoints = golden_keypoints  # A list of the known keypoints.

  @property
  def ix(self): return self._ix
  @property
  def img(self): return self._img
  @property
  def img_gray(self): return self._img_gray
  @property
  def keypoints(self): return self._keypoints
  @property
  def golden_keypoints(self): return self._golden_keypoints

  
class PinballVideo:
  """Represents a pinball video."""
  def __init__(self, imgs, all_keypoints=None, all_golden_keypoints=None):
    self._imgs = imgs  # The raw video images (ndarray, disk mapped?).
    # We copy the converted ndarray here because without doing so they have
    # a mysterious base object that isn't imgs but exists somewhere. I'd
    # rather make a copy and know that this is where the memory is alloc'd. 
    self._imgs_gray = common.convert_bgr_planes_to_gray(imgs).copy()
    if not all_keypoints:
      all_keypoints = [[]] * len(imgs)
    self._all_keypoints = all_keypoints
    if not all_golden_keypoints:
      all_golden_keypoints = [[]] * len(imgs)
    self._all_golden_keypoints = all_golden_keypoints
    
    self._frames = []
    for i, (img, img_gray, keypoints, golden_keypoints) in enumerate(zip(self._imgs, self._imgs_gray, all_keypoints, all_golden_keypoints)):
      self._frames.append(PinballFrame(i, img, img_gray, keypoints, golden_keypoints))

      # Make sure it's a view, not a copy, when we use the properties.
      assert self.frames[i].img.base is self._imgs
      assert self.frames[i].img_gray.base is self._imgs_gray
      assert np.may_share_memory(self.frames[i].img.base, self._imgs)
      assert np.may_share_memory(self.frames[i].img_gray.base, self._imgs_gray)
      assert not self.frames[i].img.flags['OWNDATA']
      assert not self.frames[i].img_gray.flags['OWNDATA']

  @property
  def imgs(self): return self._imgs
  @property
  def imgs_gray(self): return self._imgs_gray
  @property
  def all_keypoints(self): return self._all_keypoints
  @property
  def all_golden_keypoints(self): return self._all_golden_keypoints
  @property
  def frames(self): return self._frames
  @property
  def num_frames(self): return len(self._frames)
