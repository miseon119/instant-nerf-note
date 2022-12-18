# instant-nerf-note
Test Project link: https://github.com/NVlabs/instant-ngp

## Build
```bash
cmake . -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=$(which nvcc)
cmake --build build --config RelWithDebInfo -j
```

## Capture image for Instant NeRF
1. Generation pipeline uses COLMAP to determine camera positions.
2. train

## Useful Commands

### Use Colmap
```bash
python3 scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 16 --images <data/your-data>
```


### Training
```bash
./build/testbed --scene transforms.json
```

### Rendering a video
```bash
python3 scripts/run.py --scene <yout-transforms.json> --load_snapshot <yout-transforms_base.msgpack> --video_n_seconds <video length> --video_fps 30 --width 1920 --height 1080 --mode nerf --video_camera_path transforms_base_cam.json
```

