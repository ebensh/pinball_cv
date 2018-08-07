ffmpeg -ss 00:01:44.800 -i 003.MP4 -vframes 1 003_frame.png

for k1 in $(seq -0.3 0.02 -0.2); do
	for k2 in $(seq 0.0 0.02 0.2); do
	#for k2 in $(seq -0.2 0.02 0.0); do
		outfile=$(echo "out_${k1}_${k2}.png" | sed 's/-/n/g')
		ffmpeg -i 003_frame.png \
		    -vf lenscorrection=k1=${k1}:k2=${k2} \
		    ${outfile}
	done
done

# k1 > 0 == fisheye
# k1 < 0 == counter-fisheye 
# k2 > 0 == roll corners "under" (more fisheye)
# k2 < 0 == roll corners "up"    (less fisheye)

# I believe that we need negative k1 and positive k2

# -0.28 0.07 is pretty good
