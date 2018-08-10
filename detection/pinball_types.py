import cv2
import numpy as np
import common

class PinballFrame:
  """Represents a single frame of a pinball video."""
  def __init__(self, ix, img, img_gray, keypoints):
    self._ix = ix  # The frame's index.
    self._img = img  # A view into the video's color images.
    self._img_gray = img_gray  # A view into the video's grayscale images.
    self._keypoints = keypoints  # A list of keypoints in the frame.

  @property
  def ix(self): return self._ix
  @property
  def img(self): return self._img
  @property
  def img_gray(self): return self._img_gray
  @property
  def keypoints(self): return self._keypoints
  
class PinballVideo:
  """Represents a pinball video."""
  def __init__(self, imgs, all_keypoints=None):
    self._imgs = imgs  # The raw video images (ndarray, disk mapped?).
    self._imgs_gray = common.convert_bgr_planes_to_gray(imgs)
    if not all_keypoints:
      all_keypoints = [[]] * len(imgs)
    self._all_keypoints = all_keypoints  # Each image's keypoints (ndarray).
    
    self._frames = []
    for i, (img, img_gray, keypoints) in enumerate(zip(self._imgs, self._imgs_gray, all_keypoints)):
      self._frames.append(PinballFrame(i, img, img_gray, keypoints))

      # Make sure it's a view, not a copy, when we use the properties.
      assert self.frames[i].img.base is self._imgs
      # imgs_gray is not itself the base object! (whaaat XD)
      # But fortunately they share a common base.
      assert self.frames[i].img_gray.base is self._imgs_gray.base
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
  def frames(self): return self._frames
  @property
  def num_frames(self): return len(self._frames)

      
class PinballVideoData:
  """A pickle-able object containing all pinball video data."""
  imgs = None
  keypoints = None
