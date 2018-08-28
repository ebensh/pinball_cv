ffmpeg \
    -i ball_1.mp4 \
    -r 30         \
    -an           \
    -vf "transpose=1,scale=iw*.5:ih*.5" \
    -c:v libx264 \
    -b:v 3M \
    -movflags faststart \
    pinball_field_video_ball_1.avi
