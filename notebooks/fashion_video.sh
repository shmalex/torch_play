ffmpeg -r 60 -f image2 -s 720,720 -i ./fashion_images/%6d.jpg -c:v h264_nvenc -crf 15 video.mp4
# ffmpeg -r 5 -f image2 -s 648,648 -i ./clustering_video/figure_%5d.png -c:v h264_nvenc -crf 15 ../video_1.mp4
#ffmpeg -r 5 -f image2 -s 648,648 -i ./clustering_video/figure_%5d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p  ../video_1.mp4
# -pix_fmt yuv420p
