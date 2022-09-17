# instant-nerf-note
Test Project link: https://github.com/NVlabs/instant-ngp

## Useful Command

### Use Colmap
```bash
python3 scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 16 --images <data/your-data>
```

### Rendering a video
```bash
python3 scripts/run.py --scene <yout-transforms.json> --load_snapshot <yout-transforms_base.msgpack> --video_n_seconds <video length> --video_fps 30 --width 1920 --height 1080 --mode nerf --video_camera_path transforms_base_cam.json
```

