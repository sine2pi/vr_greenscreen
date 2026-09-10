<img width="1703" height="184" alt="hikaru" src="https://github.com/user-attachments/assets/0588ad14-d4a4-4245-a874-c145650c92eb" />

how to use:

Install whatever you need to run the ai models. Pytorch cuda the usual suspects. This runs on windows and uses ffmpeg.

python pipeline.py ./videos

Alpha packing options added:
python pipeline.py ./videos --alpha True

Setting --alpha True will turn off the background color pathway and create an alpha packed video with the mask suitable for DEOVR.
added decompose alpha for creating mask and video mp4 from already alpha packed videos.. for training purposes.

python pipeline.py

    --mask-height', type=int, default=1600)
    --segment-length', type=float, default=2)
    --erode', type=int, default=0)
    --dilate', type=int, default=0)
    --prompt', type=str, default='woman
    --warmup', type=int, default=6)
    --add-box', type=bool, default=False)
    --sub-box', type=bool, default=False)
    --seed-model', type=str, default='sam3video', choices=['sam3', 'sam3video', 'sam31video', 'sapiens', 'hybrid'])
    --sapiens-threshold', type=float, default=0.5, help='Threshold for converting Sapiens alpha matte to a binary mask
    --gate-dilate', type=int, default=5)
    --propagation-backend', type=str, default='matanyone', choices=['matanyone', 'sam3', 'sam3_sapiens', 'sapiens'])
    --sam3-unsplit-sbs', action='store_true', help='SAM3 backends only: run propagation on unsplit SBS segments (1 SAM3 job per segment) instead of per-eye jobs
    --refine-fg-threshold', type=float, default=0.95, help='SAM confidence threshold for sure foreground in sam3_sapiens refinement
    --refine-bg-threshold', type=float, default=0.05, help='SAM confidence threshold for sure background in sam3_sapiens refinement
    --refine-unknown-dilate', type=int, default=5, help='Dilate unknown/boundary region before Sapiens edge refinement (sam3_sapiens)
    --matanyone-version', type=str, default='v2', choices=['v1', 'v2'], help='Select MatAnyone runtime version
    --ma2-mem-every', type=int, default=3, help='Override MatAnyone mem_every (works for v1 and v2; e.g. 2 or 3 for faster refresh
    --ma2-max-mem-frames', type=int, default=2, help='Override MatAnyone memory window in frames (works for v1 and v2
    --ma2-use-long-term', type=str, default='off', choices=['auto', 'on', 'off'], help='Override MatAnyone long-term memory mode (works for v1 and v2
    --temporal-median-window', type=int, default=0, help='Temporal median window for alpha cleanup. 0 disables; use odd values >= 3 (e.g. 5)
    --tta-enable', action='store_true', help='Enable self-supervised test-time adaptation (TTA) for MatAnyone before propagating each segment
    --tta-steps', type=int, default=8, help='Number of TTA gradient steps per segment/eye (only used with --tta-enable)
    --tta-lr', type=float, default=1e-4, help='Learning rate for TTA adaptation (only used with --tta-enable
    --tta-warmup-steps', type=int, default=3, help='Sensory-memory settle passes per TTA step before scoring the prediction (only used with --tta-enable
    --tta-supervised-weight', type=float, default=1.0, help='Weight for the supervised L1 loss against each SAM3 mask (only used with --tta-enable
    --tta-max-size', type=int, default=0, help='Working resolution cap for TTA frames/masks (0 = use --mask-height). Lower this (e.g. 512) to cut VRAM usage on large inputs
    --no-normalize-input', dest='normalize_input', action='store_false', help='Skip upfront input normalization/transcoding
    --overlay-output', type=str, default='input_path', help='Write a composited video with the mask over the original source
    --overlay-color', type=str, default='0x00ff00', help='Background color for overlay (use 0x00ff00 for pure green)
    --overlay-mask', type=str, default=None, help='Write a composited video with a provided mask over the original source
    --alpha-packer', type=str, default=None, help='Run alpha packer on its own. Provide folder with video and mask (_mask.<ext>)
    --decompose-alpha', '--decompose_alpha', dest='decompose_alpha', action='store_true', help='Takes alpha packed videos and separates them into individual video and mask files
    --decompose-clean-mask', type=str, default='assets/black_mask.png', help='PNG overlay used to clean alpha payload regions in decomposed video output. Use "none" to disable and keep stream copy.
    --alpha', type=bool, default=False, help='Run alpha packer instead of overlay within pipeline. --alpha <true|false> default is False
    --show-plots', type=bool, default=False, help='Sam3 mask plots will be displayed if True. Default is False
    --fisheye180', nargs='?', const=FISHEYE180_PIPELINE_MODE, default=None, help='Convert an SBS equirectangular input video or folder to SBS fisheye180
    --debug', type=int, default=None, help='Debug mode: process only the first N segments


