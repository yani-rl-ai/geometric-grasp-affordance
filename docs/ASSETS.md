# Third-party assets

The simulation scene uses the following third-party material. **Before making the repository public, confirm that the terms of each item allow
redistribution, add the required attributions below, and tick the boxes.**

| Path | Content | Origin | Licence / terms | Confirmed |
|---|---|---|---|---|
| `assets/ycb/003_cracker_box/`, `006_mustard_bottle/`, `011_banana/`, `025_mug/`, `056_tennis_ball/` | textured meshes (`textured.obj`, `texture_map.png`, `textured.mtl`) of five objects | YCB Object and Model Set, <https://www.ycbbenchmarks.com/> (B. Calli et al.) | check the terms of use on the YCB website | [ ] |
| `assets/fetch/robot.xml`, `assets/fetch/shared.xml`, `assets/fetch/meshes/*.stl`, `assets/textures/block.png` | Fetch robot MJCF description and collision meshes, in the layout of the Fetch environments of OpenAI Gym / Gymnasium-Robotics (`robot0:` prefixes) | Fetch Robotics robot description, converted for MuJoCo | check the licence of the repository these files were taken from and of the Fetch description | [ ] |
| `scene.xml` | table, objects, lights, cameras, actuators | this work | same licence as the code | [x] |

Software dependencies (not redistributed here, installed with `pip install -r requirements.txt`): MuJoCo (Apache-2.0), NumPy, OpenCV, PyTorch,
Ultralytics (AGPL-3.0 as published by Ultralytics; the `yolo26s-seg.pt` weights are downloaded by Ultralytics on first use and are not part of this repository).

Algorithm: DBL-GNG, C. Z. Siow, A. A. Saputra, T. Obo, N. Kubota, *Mathematics* 12(12), 1909, 2024; the implementation in `dbl_gng_paper.py` was written for
this project from the published algorithm.
