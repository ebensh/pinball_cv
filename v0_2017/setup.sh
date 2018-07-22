# KeepVid 1080p video only
# Atlantis: https://www.youtube.com/watch?v=Lzkk2EThJBA
# Nine Ball: https://www.youtube.com/watch?v=t8KeBP19qes
# Rollergames: https://www.youtube.com/watch?v=hkWUsgzkthw

# ffmpeg: https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu

#avconv -ss 00:00:10 -i atlantis_tutorial_video.mp4 -t 00:00:10 -codec copy short_clip.mp4
#avconv -ss 00:00:10 -i atlantis_tutorial_video.mp4 -t 00:05:00 -codec copy intro_removed.mp4
#ffmpeg -i input_videos/atlantis_tutorial_video.mp4 -ss 00:00:10 -vf scale="iw*.5:ih*.5" -t 00:01:00 input_videos/atlantis_intro_removed.mp4
#ffmpeg -i input_videos/atlantis_original.mp4 -ss 00:00:10 -t 00:01:00 input_videos/atlantis.mp4

#avconv -ss 00:01:05 -i input_videos/nine_ball_original.mp4 scale="trunc(oh*a*2)/2:480" -t 00:00:58 -codec copy input_videos/nine_ball.mp4
#avconv -i input_videos/nine_ball_original.mp4 scale="1080x608" -codec copy input_videos/nine_ball.mp4
#ffmpeg -i input_videos/nine_ball_original.mp4 -ss 00:01:01 -vf scale="iw*.5:ih*.5" -t 00:01:02 input_videos/nine_ball.mp4
#ffmpeg -i input_videos/nine_ball_original.mp4 -ss 00:01:01 -t 00:01:02 input_videos/nine_ball.mp4

#ffmpeg -i input_videos/rollergames_original.mp4 -ss 00:02:16 -t 00:00:29 input_videos/rollergames.mp4

# https://stackoverflow.com/questions/11004137/re-sampling-h264-video-to-reduce-frame-rate-while-maintaining-high-image-quality
ffmpeg -i input_videos/hot_hand_ebensh_2017_11_27_original.mp4 -ss 00:00:03 -t 00:00:27 -r 30 -an -vf "transpose=1,scale=iw*.5:ih*.5" -c:v libx264 -b:v 3M -movflags faststart input_videos/hot_hand_ebensh_2017_11_27.mp4

# ./segmenter.py --game_config=configs/atlantis_full_resolution.cfg
# ffmpeg -i processed_videos/atlantis_score_board.avi -filter:v "crop=132:20:0:0" processed_videos/atlantis_score_board_player1.avi
# ffmpeg -i processed_videos/atlantis_score_board_player1.avi "processed_videos/player1_%04d.jpg"

# ./segmenter.py --game_config=configs/nine_ball_full_resolution.cfg
#ffmpeg -i processed_videos/nine_ball_score_board.avi -filter:v "crop=555:63:30:10" processed_videos/nine_ball_score_board_player1.avi
# ffmpeg -i processed_videos/nine_ball_score_board_player1.avi "processed_videos/player1_%04d.jpg"

# ./segmenter.py --game_config=configs/rollergames_full_resolution.cfg
./segmenter.py --game_config=configs/hot_hand_full_resolution.cfg
