8k video less than 8gb vram. 

```
how to use:
run.bat
Tested on Windows.

added standalone option for converting equirectangular sbs vr videos with masks to alpha packed fisheye
-add label _mask to masks before extension first part same as the video. example: video.mp4 video_mask.mp4 amd place in folder

python pipeline.py "C:\Videos" --alpha-packer True --fisheye180 True
