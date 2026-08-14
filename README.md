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
    ('--seed-model', type=str, default='sam3', choices=['sam3', 'sapiens', 'hybrid'], help='Seed mask guy (sam3, sapiens, or hybrid)')
    ('--sapiens-threshold', type=float, default=0.5, help='Threshold for converting Sapiens alpha matte to a binary mask')
    ('--gate-dilate', type=int, default=5, help='Dilate SAM3 gating in hybrid mode')
    ('--no-normalize-input', dest='normalize_input', action='store_false', help='Skip upfront input normalization/transcoding')
    parser.set_defaults(normalize_input=True)
    ('--overlay-output', type=str, default='input_path', help='Where you want the composited video to be saved. Default: input folder')
    ('--overlay-color', type=str, default='0x00ff00', help='Background color, uses 0x00ff00 for pure green by default')


```
Runs fine on windows.
