# Converts from original_video.webm to processed_video.mp4
# -ss and -t parameters could be specified for start time and length.

ffmpeg \
    -ss 00:00:03  \
    -i original_video.webm \
    -t 00:00:27   \
    -r 30         \
    -an           \
    -vf "transpose=1,scale=iw*.5:ih*.5" \
    -c:v libx264 \
    -b:v 3M \
    -movflags faststart \
    processed_video.avi
