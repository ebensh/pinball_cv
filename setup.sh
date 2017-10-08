# KeepVid 1080p video only https://www.youtube.com/watch?v=Lzkk2EThJBA
#   => write to atlantis_tutorial_video.mp4

avconv -ss 00:00:10 -i atlantis_tutorial_video.mp4 -t 00:00:10 -codec copy pinball.mp4
