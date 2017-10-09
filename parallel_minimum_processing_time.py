#!/usr/bin/python

import threading
import Queue
import cv2
import numpy as np

# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class FileVideoStream:
  def __init__(self, path, queueSize=128):
    self._input_video = cv2.VideoCapture(path)
    self._queue = Queue.Queue(maxsize=queueSize)
    self._stopped = False

  def start(self):
    """Start a thread to call the .update method."""
    t = threading.Thread(target=self.run, args=())
    t.daemon = True
    t.start()

  def run(self):
    """Run until self._stopped, reading frames and putting them in the queue."""
    while True:
      if self._stopped: return
      (grabbed, frame) = self._input_video.read()
      if not grabbed:  # We've processed everything!
        self.stop()
        return
      self._queue.put(frame)  # Blocks until slot is available.

  def read(self):
    while True:
      if self._stopped: return None
      try:
        frame = self._queue.get_nowait()
        return frame
      except Queue.Empty:
        pass

  def stop(self):
    self._stopped = True
    #self._stream.release()


def main():
  stream = FileVideoStream('intro_removed.mp4')
  stream.start()

  def process_frame(stream, results):
    num_frames = 0
    while True:
      frame = stream.read()
      if frame is None:
        results.put(num_frames)
        return
      num_frames += 1
      #print threading.current_thread(), num_frames
    

  results = Queue.Queue()
  processing_threads = [threading.Thread(target=process_frame, args=(stream, results))
                        for _ in xrange(10)]
  for thread in processing_threads:
    thread.start()
  for thread in processing_threads:
    thread.join()

  num_frames = 0
  for _ in xrange(len(processing_threads)):
    num_frames += results.get()
    
  print num_frames
  cv2.destroyAllWindows()

  print "Number of frames processed: {0}".format(num_frames)

if __name__ == '__main__':
  main()
