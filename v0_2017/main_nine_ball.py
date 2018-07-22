#!/usr/bin/python

import argparse
import cv2
import itertools
import numpy as np
import pickle

import common

def get_blob_detector():
  # Set up the SimpleBlobdetector with default parameters.
  params = cv2.SimpleBlobDetector_Params()
     
  # Change thresholds
  params.minThreshold = 0;
  params.maxThreshold = 256;
     
  # Filter by Area.
  params.filterByArea = True
  params.minArea = 75  # The pinball is ~17 pixels across, or 201 area.
  params.maxArea = 600
    
  # Filter by Circularity
  params.filterByCircularity = True
  params.minCircularity = 0.6
    
  # Filter by Convexity
  params.filterByConvexity = False
  #params.minConvexity = 0.5
    
  # Filter by Inertia
  #params.filterByInertia = True
  #params.minInertiaRatio = 0.5
  return cv2.SimpleBlobDetector_create(params)


def get_pinball_field_poly(frame):
  # TODO: Replace this with something dynamic and intelligent, super cool, like
  # they have in those movies with the people who type on the keyboard AT THE
  # SAME TIME omfg that's so 1337. ... Til then just hardcode the values.
  # Points here are X, Y, where top left is 0, 0.
  return np.array([
    (72, 191),   # top left
    (231, 191),  # top right
    (266, 539),  # bottom right
    (36, 539),   # bottom left
    (72, 191)])   # close back to first point  

def get_pinball_field_mask(frame, pinball_field_poly):
  mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # Bool?
  return cv2.fillConvexPoly(mask, pinball_field_poly, 255)

def get_perspective_transform(frame, pinball_field_poly):
  rows, cols = frame.shape[:2]
  corners_of_frame = np.array([
    (0, 0),            # top left
    (cols-1, 0),       # top right
    (cols-1, rows-1),  # bottom right
    (0, rows-1)], dtype=np.float32)      # bottom left
  return cv2.getPerspectiveTransform(pinball_field_poly[:-1].astype(np.float32), corners_of_frame)

# IMPORTANT!!! Subtraction will WRAP with uint8 if it goes negative!
def trim_to_uint8(arr): return np.clip(arr, 0, 255).astype(np.uint8)


def main():
  FRAME_BUFFER_SIZE = 9
  CURRENT_FRAME_INDEX = FRAME_BUFFER_SIZE/2

  cap = cv2.VideoCapture(args.infile)

  detector = get_blob_detector()  # REMEMBER: DETECTS BLACK OBJECTS, WHITE BG.
  frame_to_keypoints = {}

  _, raw_frame = cap.read()
  frame_buffer = common.FrameBuffer(FRAME_BUFFER_SIZE, raw_frame.shape)

  video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'),
                          30.0, raw_frame.shape[1::-1])

  pinball_field_poly = get_pinball_field_poly(raw_frame)
  pinball_field_mask = get_pinball_field_mask(raw_frame, pinball_field_poly)
  pinball_perspective_transform = get_perspective_transform(raw_frame, pinball_field_poly)

  pinball_points = np.zeros(raw_frame.shape, dtype=np.uint8)

  if args.display_all_images:
    cv2.namedWindow('past_stats', cv2.WINDOW_NORMAL)
    cv2.namedWindow('combined', cv2.WINDOW_NORMAL)
  frame_count = 0
  while cap.isOpened():
    grabbed, raw_frame = cap.read()
    if not grabbed: break
    #if frame_count % 2 != 0: continue

    frame_printer = common.FramePrinter()
    
    pinball_area = cv2.bitwise_and(raw_frame, raw_frame, mask=pinball_field_mask)
    #pinball_area = cv2.warpPerspective(pinball_area, pinball_perspective_transform,
    #                                   dsize=raw_frame.shape[1::-1])
    common.display_image(pinball_area, 'pinball_area', args.display_all_images)
    frame_buffer.append(pinball_area)

    past = frame_buffer.get_view(None, CURRENT_FRAME_INDEX - 2)
    past_gray = frame_buffer.get_view(None, CURRENT_FRAME_INDEX - 2, color=False)
    current_frame = frame_buffer.get_view(CURRENT_FRAME_INDEX, CURRENT_FRAME_INDEX + 1)[0]
    current_frame_gray = frame_buffer.get_view(CURRENT_FRAME_INDEX, CURRENT_FRAME_INDEX + 1, color=False)[0]
    future = frame_buffer.get_view(CURRENT_FRAME_INDEX + 1 + 2, None)
    future_gray = frame_buffer.get_view(CURRENT_FRAME_INDEX + 1 + 2, None, color=False)

    past_stats = common.get_named_statistics(past_gray)
    future_stats = common.get_named_statistics(future_gray)

    if True:  # frame_count % 30 == 0:
      common.display_image(
          cv2.vconcat([cv2.hconcat(past_gray), cv2.hconcat(future_gray)]),
          'combined', args.display_all_images)
      past_stats_printer = common.FramePrinter()
      common.print_statistics(past_stats, past_stats_printer)
      common.display_image(past_stats_printer.get_combined_image(), 'past_stats', args.display_all_images)


    # current_frame = frame_buffer.get_view(CURRENT_FRAME_INDEX, CURRENT_FRAME_INDEX + 1)[0]
    # current_frame_gray = frame_buffer.get_view(CURRENT_FRAME_INDEX, CURRENT_FRAME_INDEX + 1, color=False)[0]
    # window = frame_buffer.get_view(CURRENT_FRAME_INDEX - 2, CURRENT_FRAME_INDEX + 3)  # -2 to +2
    # window_gray = frame_buffer.get_view(CURRENT_FRAME_INDEX - 2, CURRENT_FRAME_INDEX + 3, color=False)  # -2 to +2
    # window_stats = common.get_named_statistics(window_gray)
    # common.display_image(cv2.hconcat(window_gray), 'window_gray', args.display_all_images)
    # window_stats_printer = common.FramePrinter()
    # common.print_statistics(window_stats, window_stats_printer)
    # common.display_image(window_stats_printer.get_combined_image(), 'window_stats', args.display_all_images)

    # foreground_mask = trim_to_uint8(current_frame_gray.astype(np.int16) - window_stats.mean.astype(np.int16))
    # common.display_image(foreground_mask, 'foreground_mask')
    # _, foreground_mask = cv2.threshold(foreground_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # common.display_image(foreground_mask, 'foreground_mask_thresholded')
    # common.display_image(cv2.bitwise_and(current_frame, current_frame, mask=foreground_mask))
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break
    # continue
    
    # Subtract out the unchanging background (mean past, mean future) from current frame
    foreground_mask_past = np.absolute(current_frame_gray.astype(np.int16) - past_stats.mean.astype(np.int16))
    foreground_mask_future = np.absolute(current_frame_gray.astype(np.int16) - future_stats.mean.astype(np.int16))

    # Threshold the differences and combine them.
    foreground_mask = np.bitwise_and(foreground_mask_past >= 25, foreground_mask_future >= 25)

    # Mask away the areas we know are changing based on thresholded ptp (ptp past, ptp future).
    # Take the absolute difference (per pixel) from the mean in each frame.
    changing_mask_past = np.absolute(past_gray.astype(np.int16) - past_stats.mean.astype(np.int16)) >= 5
    # Count how many frames were significantly different and threshold.
    changing_mask_past = np.sum(changing_mask_past, axis=0) >= 3
    
    # Take the absolute difference (per pixel) from the mean in each frame.
    changing_mask_future = np.absolute(future_gray.astype(np.int16) - future_stats.mean.astype(np.int16)) >= 5
    # Count how many frames were significantly different.
    changing_mask_future = np.sum(changing_mask_future, axis=0) >= 3

    changing_mask = np.bitwise_or(changing_mask_past, changing_mask_future)

    # The final mask is the foreground (keep) minus the changing mask (remove).
    final_mask = 255 * np.bitwise_and(foreground_mask, np.bitwise_not(changing_mask)).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Erode (remove small outlying pixels), then dilate.
    final_mask_polished = final_mask.copy()
    final_mask_polished = cv2.erode(final_mask_polished, np.ones((3, 3),np.uint8), iterations=1)
    final_mask_polished = cv2.dilate(final_mask_polished, kernel, iterations=2)
    #final_mask_polished = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    common.display_image(np.hstack([
      255 * foreground_mask.astype(np.uint8),
      255 * changing_mask.astype(np.uint8),
      final_mask,
      final_mask_polished]), 'masks', args.display_all_images)

    keypoints = detector.detect(cv2.bitwise_not(final_mask_polished))
    frame_to_keypoints[frame_count] = [(kp.pt[0], kp.pt[1], kp.size) for kp in keypoints]
    # Fade away old points.
    pinball_points = trim_to_uint8(pinball_points.astype(np.int16) - 1)
    # Then draw new points.
    pinball_points = cv2.drawKeypoints(pinball_points, keypoints, np.array([]), (0,255,255),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    pinball_points_display = current_frame.copy()
    pinball_points_display[pinball_points > 0] = pinball_points[pinball_points > 0]
    common.display_image(pinball_points_display, 'pinball_points')
    if video:
      video.write(pinball_points_display)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  print 'Frames processed: %d' % frame_count

  # Write the pickled keypoints.
  with open(args.keypoints, 'wb') as keypoints_file:
    pickle.dump(frame_to_keypoints, keypoints_file)

  cap.release()
  video.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Track a pinball.')
  parser.add_argument('--infile', required=True, type=str, help='Input video file path.')
  #parser.add_argument('--background', type=str, help='Path to background to subtract.')
  #parser.add_argument('--mask', type=str, help='Mask to apply to each frame.')
  parser.add_argument('--keypoints', type=str, help='Path to keypoints file to write.')
  parser.add_argument('--display_all_images', default=False, type=bool,
                      help='Display all (debug) images.')
  args = parser.parse_args()
  main()
