# KeepVid 1080p video only https://www.youtube.com/watch?v=Lzkk2EThJBA
#   => write to atlantis_tutorial_video.mp4

avconv -ss 00:00:10 -i atlantis_tutorial_video.mp4 -t 00:00:10 -codec copy short_clip.mp4
avconv -ss 00:00:10 -i atlantis_tutorial_video.mp4 -s 640x480 -codec copy intro_removed.mp4
