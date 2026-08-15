how to use:

python vrmasking.py folder

folder = (path to folder of videos)

```python
python vrmasking.py (path to folder of videos)

masks and composite videos will automatically be places in the input folder with the appropriate names.

optional:
    ('--mask-height', type=int, default=1008)
    ('--segment-length', type=float, default=10)
    ('--erode', type=int, default=0)
    ('--dilate', type=int, default=0)
    ('--prompt', type=str, default='one woman')
    ('--seed-model', type=str, default='sam3', choices=['sam3', 'sam3video', 'sapiens', 'hybrid'], help='Seed mask mode (sam3, sam3video, sapiens, or hybrid)')
    ('--sapiens-threshold', type=float, default=0.5, help='Threshold for converting Sapiens alpha matte to a binary mask')
    ('--gate-dilate', type=int, default=5, help='Dilate SAM3 gating in hybrid mode')
    ('--no-normalize-input', dest='normalize_input', action='store_false', help='Skip upfront input normalization/transcoding')
    parser.set_defaults(normalize_input=True)
    ('--overlay-output', type=str, default='input_path', help='Where you want the composited video to be saved. Default: input folder')
    ('--overlay-color', type=str, default='0x00ff00', help='Background color, uses 0x00ff00 for pure green by default')


```
Runs fine on windows.
The sapiens thing was just to test an idea and can be ignored but it works fine if you want to experiment. Sam3 is safest bet. You can also try sam3.1 video... also changing these guys :
``` python
SAM3_BOX_CXCYWH_NORM = (0.5, 0.5, 0.5, 0.5) <- positive
SAM3_BOX2_CXCYWH_NORM = (0.5, 0.9, 0.9, 0.18) <-negative

```
will help focus the text prompt. Make sure the text prompt and the box are the same person. 

The script itself doesn't need anything special to run.. the encoder is nvenc but you can just change that. The AI models need the usual suspects of pytorch etc. No special make model or year of python or ffmpeg or anything whatever you have installed on your system is fine. 

