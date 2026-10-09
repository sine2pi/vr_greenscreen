<img width="1123" height="900" alt="image2" src="https://github.com/user-attachments/assets/10bcc3b4-0b19-44eb-b110-751d1320642f" />


```

 Daily Breaking Changes

To turn the videos that are in the videos folder into greenscreen videos:
 use:
     python pipeline.py "videos"

To turn the videos that are in the videos folder into 180 sbs fisheye videos with an alpha packed mask ready for passthrough in deovr use: 
     python pipeline.py "Videos" --alpha True --fisheye180 True



^There are many flags and arguments some of which work really well.. the others flags are bad ideas that need to be removed eventually.
*^You probably should already know and regularly use all the required dependencies.
