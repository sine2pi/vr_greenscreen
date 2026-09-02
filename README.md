<img width="1703" height="184" alt="hikaru" src="https://github.com/user-attachments/assets/0588ad14-d4a4-4245-a874-c145650c92eb" />

how to use:

Install whatever you need to run the ai models.

python pipeline.py path_to_videos

Alpha packing options added

    '--mask-height', type=int, default=1008
    '--segment-length', type=float, default=4
    '--erode', type=int, default=4
    '--dilate', type=int, default=-4
    '--prompt', type=str, default='one woman'
    '--warmup', type=int, default=6
    '--seed-model', type=str, default='sam3video', choices=['sam3', 'sam3video', 'sam31video', 'sapiens', 'hybrid'], help='Seed mask mode sam3, sam3video, sam31video, sapiens, or hybrid'
    '--sapiens-threshold', type=float, default=0.5, help='Threshold for converting Sapiens alpha matte to a binary mask'
    '--gate-dilate', type=int, default=5, help='Dilate SAM3 gating in hybrid mode'
    '--matanyone-version', type=str, default='v2', choices=['v1', 'v2'], help='Select MatAnyone runtime version'
    '--ma2-mem-every', type=int, default=6, help='Override MatAnyone mem_every works for v1 and v2; e.g. 2 or 3 for faster refresh'
    '--ma2-max-mem-frames', type=int, default=2, help='Override MatAnyone memory window in frames works for v1 and v2'
    '--ma2-use-long-term', type=str, default='off', choices=['auto', 'on', 'off'], help='Override MatAnyone long-term memory mode works for v1 and v2'
    '--temporal-median-window', type=int, default=0, help='Temporal median window for alpha cleanup. 0 disables; use odd values >= 3 e.g. 5'
    '--no-normalize-input', dest='normalize_input', action='store_false', help='Skip upfront input normalization/transcoding'
    '--overlay-output', type=str, default='input_path', help='Write a composited video with the mask over the original source'
    '--overlay-color', type=str, default='0x00ff00', help='Background color for overlay use 0x00ff00 for pure green'
    '--overlay-mask', type=str, default=None, help='Write a composited video with a provided mask over the original source'
    '--alpha-packer', type=str, default=None, help='Run alpha packer on its own. '
    '--alpha', type=bool, default=False, help='Run alpha packer instead of overlay within pipeline. --alpha <true|false> default is False'


refine with sapiens wip 
