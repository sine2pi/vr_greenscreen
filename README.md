<img width="1703" height="184" alt="hikaru" src="https://github.com/user-attachments/assets/0588ad14-d4a4-4245-a874-c145650c92eb" />




  <img width="200" height="100" alt="Screenshot 2026-09-19 171929" src="https://github.com/user-attachments/assets/76c6ea78-18f7-4408-ab69-a4d44659c2ad" />

8k video less than 8gb vram. 

```
how to use:
run.bat
Tested on Windows.

added standalone option for converting equirectangular sbs vr videos with masks to alpha packed fisheye
-add label _mask to masks before extension first part same as the video. example: video.mp4 video_mask.mp4 amd place in folder

python pipeline.py "C:\Videos" --alpha-packer True --fisheye180 True
