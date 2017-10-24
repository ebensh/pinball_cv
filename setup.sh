# KeepVid 1080p video only https://www.youtube.com/watch?v=Lzkk2EThJBA
#   => write to atlantis_tutorial_video.mp4
# Nine Ball: https://www.youtube.com/watch?v=t8KeBP19qes, nine_ball.mp4

# ffmpeg: https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu

#avconv -ss 00:00:10 -i atlantis_tutorial_video.mp4 -t 00:00:10 -codec copy short_clip.mp4
#avconv -ss 00:00:10 -i atlantis_tutorial_video.mp4 -t 00:05:00 -codec copy intro_removed.mp4
#ffmpeg -i input_videos/atlantis_tutorial_video.mp4 -ss 00:00:10 -vf scale="iw*.5:ih*.5" -t 00:01:00 input_videos/atlantis_intro_removed.mp4
ffmpeg -i input_videos/atlantis_original.mp4 -ss 00:00:10 -t 00:01:00 input_videos/atlantis.mp4

#avconv -ss 00:01:05 -i input_videos/nine_ball_original.mp4 scale="trunc(oh*a*2)/2:480" -t 00:00:58 -codec copy input_videos/nine_ball.mp4
#avconv -i input_videos/nine_ball_original.mp4 scale="1080x608" -codec copy input_videos/nine_ball.mp4
#ffmpeg -i input_videos/nine_ball_original.mp4 -ss 00:01:01 -vf scale="iw*.5:ih*.5" -t 00:01:02 input_videos/nine_ball.mp4
ffmpeg -i input_videos/nine_ball_original.mp4 -ss 00:01:01 -t 00:01:02 input_videos/nine_ball.mp4

#./create_average_background.py --infile=intro_removed.mp4 --outfile_prefix=intro_removed --display_all_images=True
#./edit_image.py --infile=intro_removed_range.png --display_all_images=True  --outfile=intro_removed_range_mask.png
#./main.py --infile=intro_removed.mp4 --background=intro_removed_weighted.png --mask=intro_removed_range_mask.png --display_all_images=True


# ./segmenter.py --game_config=configs/nine_ball_full_resolution.cfg
#ffmpeg -i processed_videos/nine_ball_score_board.avi -filter:v "crop=555:63:30:10" processed_videos/nine_ball_score_board_player1.avi
# ffmpeg -i processed_videos/nine_ball_score_board_player1.avi "processed_videos/player1_%04d.jpg"

./segmenter.py --game_config=configs/atlantis_full_resolution.cfg
ffmpeg -i processed_videos/atlantis_score_board.avi -filter:v "crop=132:20:0:0" processed_videos/atlantis_score_board_player1.avi
ffmpeg -i processed_videos/atlantis_score_board_player1.avi "processed_videos/player1_%04d.jpg"
