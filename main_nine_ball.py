#!/usr/bin/python

import argparse
from collections import deque
import cv2
import itertools
import numpy as np

import common

def get_blob_detector():
  # Set up the SimpleBlobdetector with default parameters.
  params = cv2.SimpleBlobDetector_Params()
     
  # Change thresholds
  params.minThreshold = 0;
  params.maxThreshold = 256;
     
  # Filter by Area.
  params.filterByArea = True
  params.minArea = 30
  params.maxArea = 500
    
  # Filter by Circularity
  params.filterByCircularity = True
  params.minCircularity = 0.8
    
  # Filter by Convexity
  params.filterByConvexity = False
  #params.minConvexity = 0.5
    
  # Filter by Inertia
  params.filterByInertia = False
  #params.minInertiaRatio = 0.5
  return cv2.SimpleBlobDetector_create(params)


def main():
  cap = cv2.VideoCapture(args.infile)
  
  FRAME_BUFFER_SIZE = 10
  BLEND_ALPHA = 0.33
  frames_past = deque([], maxlen=FRAME_BUFFER_SIZE)
  frames_future = deque([], maxlen=FRAME_BUFFER_SIZE)

  detector = get_blob_detector()
  
  # Loop until the frame buffers are full at the start.
  while cap.isOpened() and len(frames_past) < FRAME_BUFFER_SIZE:
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    frames_past.append((raw_frame, cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)))
  while cap.isOpened() and len(frames_future) < FRAME_BUFFER_SIZE:
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    frames_future.append((raw_frame, cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)))

  cv2.namedWindow('combined', cv2.WINDOW_NORMAL)
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    current_frame, current_frame_gray = frames_future.popleft()
    frames_future.append((raw_frame, cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)))

    frame_printer = common.FramePrinter()

    accumulated_past = np.zeros(current_frame.shape, dtype=np.float64)
    accumulated_future = np.zeros(current_frame.shape, dtype=np.float64)
    min_past = np.full(current_frame_gray.shape, 255, dtype=np.uint8)
    max_past = np.zeros(current_frame_gray.shape, dtype=np.uint8)
    mean_past = np.zeros(current_frame.shape, dtype=np.float64)    
    min_future = np.full(current_frame_gray.shape, 255, dtype=np.uint8)
    max_future = np.zeros(current_frame_gray.shape, dtype=np.uint8)
    mean_future = np.zeros(current_frame.shape, dtype=np.float64)

    
    # Accumulate the past by iterating -> and the future by iterating <- :)
    # This is inefficient, but I want to see how the "pure" version works, only
    # using the images in the past and future buffers. If we iteratively add the
    # current frame there will be some (admittedly very small) noise from
    # distant passed and future frames.
    for past_frame, past_frame_gray in frames_past:
      min_past = np.minimum(min_past, past_frame_gray)
      max_past = np.maximum(max_past, past_frame_gray)
      cv2.accumulate(past_frame, mean_past)
      cv2.accumulateWeighted(past_frame, accumulated_past, BLEND_ALPHA)
    for future_frame, future_frame_gray in reversed(frames_future):
      min_future = np.minimum(min_future, future_frame_gray)
      max_future = np.maximum(max_future, future_frame_gray)
      cv2.accumulate(future_frame, mean_future)
      cv2.accumulateWeighted(future_frame, accumulated_future, BLEND_ALPHA)
    mean_past /= len(frames_past)
    mean_future /= len(frames_future)
    mean_past_gray = cv2.cvtColor(cv2.convertScaleAbs(mean_past), cv2.COLOR_BGR2GRAY)
    mean_future_gray = cv2.cvtColor(cv2.convertScaleAbs(mean_future), cv2.COLOR_BGR2GRAY)
    delta_past = max_past - min_past
    delta_future = max_future - min_future

    count_past = np.zeros(current_frame_gray.shape, dtype=np.uint8)
    count_future = np.zeros(current_frame_gray.shape, dtype=np.uint8)
    for past_frame, past_frame_gray in frames_past:
      count_past += (cv2.absdiff(past_frame_gray, mean_past_gray) > 10)
    for future_frame, future_frame_gray in frames_future:
      count_future += (cv2.absdiff(future_frame_gray, mean_future_gray) > 10)

    print count_past
    _, mask_past = cv2.threshold(count_past, 3, 255, cv2.THRESH_BINARY_INV)
    _, mask_future = cv2.threshold(count_future, 3, 255, cv2.THRESH_BINARY_INV)
    common.display_image(mask_past, 'mask_past')

    # accumulated_past_gray = (accumulated_past * 255.0 / accumulated_past.max()).astype(np.uint8)
    # accumulated_past_gray = cv2.cvtColor(accumulated_past_gray, cv2.COLOR_BGR2GRAY)
    # past_difference = cv2.absdiff(current_frame_gray, accumulated_past_gray)
    # past_difference = cv2.GaussianBlur(past_difference, (5, 5), 0)
    # _, past_difference = cv2.threshold(past_difference, 50, 255, 0)
    # accumulated_future_gray = (accumulated_future * 255.0 / accumulated_future.max()).astype(np.uint8)
    # accumulated_future_gray = cv2.cvtColor(accumulated_future_gray, cv2.COLOR_BGR2GRAY)
    # future_difference = cv2.absdiff(current_frame_gray, accumulated_future_gray)
    # future_difference = cv2.GaussianBlur(future_difference, (5, 5), 0)
    # #_, future_difference = cv2.threshold(future_difference, 50, 255, 0)
    
    #_, mask_past = cv2.threshold(delta_past, 25, 255, cv2.THRESH_BINARY_INV)
    #_, mask_future = cv2.threshold(delta_future, 25, 255, cv2.THRESH_BINARY_INV)
    
    final = cv2.absdiff(current_frame, cv2.convertScaleAbs(accumulated_past))
    final = cv2.bitwise_and(final, final, mask=mask_past)
    final = cv2.bitwise_and(final, final, mask=mask_future)
    final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    final = cv2.GaussianBlur(final, (3, 3), 0)
    _, final = cv2.threshold(final, 40, 255, cv2.THRESH_BINARY)

    keypoints = detector.detect(cv2.bitwise_not(final))
    print keypoints
    current_frame_with_keypoints = cv2.drawKeypoints(current_frame, keypoints, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    common.display_image(final, 'final')
    common.display_image(current_frame_with_keypoints, 'current_frame_with_keypoints')    
    
    #past_difference = (accumulated_past * 255.0 / accumulated_past.max()).astype(np.uint8)
    #future_difference = (accumulated_future * 255.0 / accumulated_future.max()).astype(np.uint8)
    #past_difference = cv2.absdiff(current_frame_gray, cv2.cvtColor(past_difference, cv2.COLOR_BGR2GRAY))
    #future_difference = cv2.absdiff(current_frame_gray, cv2.cvtColor(future_difference, cv2.COLOR_BGR2GRAY))
    frame_printer.add_image(accumulated_past, 'accumulated_past')
    #frame_printer.add_image(accumulated_past_gray, 'accumulated_past_gray')
    #frame_printer.add_image(past_difference, 'past_difference')
    frame_printer.add_image(delta_past, 'delta_past')
    frame_printer.add_image(current_frame, 'current_frame')
    frame_printer.add_image(delta_future, 'delta_future')
    #frame_printer.add_image(future_difference, 'future_difference')
    #frame_printer.add_image(accumulated_future_gray, 'accumulated_future_gray')
    frame_printer.add_image(accumulated_future, 'accumulated_future')
    
    

    frames_past.append((current_frame, current_frame_gray))

    combined_image = frame_printer.get_combined_image()
    common.display_image(combined_image, 'combined')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Track a pinball.')
  parser.add_argument('--infile', required=True, type=str, help='Input video file path.')
  #parser.add_argument('--background', type=str, help='Path to background to subtract.')
  #parser.add_argument('--mask', type=str, help='Mask to apply to each frame.')
  parser.add_argument('--display_all_images', default=False, type=bool,
                      help='Display all (debug) images.')
  args = parser.parse_args()
  main()
