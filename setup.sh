# KeepVid 1080p video only https://www.youtube.com/watch?v=Lzkk2EThJBA
#   => write to atlantis_tutorial_video.mp4

avconv -ss 00:00:10 -i atlantis_tutorial_video.mp4 -t 00:00:10 -codec copy short_clip.mp4
avconv -ss 00:00:10 -i atlantis_tutorial_video.mp4 -t 00:05:00 -codec copy intro_removed.mp4

./create_average_background.py --infile=intro_removed.mp4 --outfile_prefix=intro_removed --display_all_images=True
./edit_image.py --infile=intro_removed_range.png --display_all_images=True  --outfile=intro_removed_range_mask.png
./main.py --infile=intro_removed.mp4 --background=intro_removed_weighted.png --mask=intro_removed_range_mask.png --display_all_images=True
