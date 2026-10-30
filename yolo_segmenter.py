#!/usr/bin/env python3
"""
YOLO26 instance-segmentation wrapper used by the whole pipeline.

Only the instance MASKS are used downstream (paper Sec. V-A): the COCO class
labels are not used to estimate grasp regions, so the confidence threshold is
set low (0.05) -- the COCO detector has no "cracker box" class and labels it
"book", "bowl", "bottle" or nothing, usually with low confidence.

Weights: `yolo26s-seg.pt` is looked up (1) in $YOLO_WEIGHTS, (2) next to this
file; if it does not exist there, Ultralytics downloads it to that location on
first use (needs internet once).
"""
import os

YOLO_MODEL_NAME = "yolo26s-seg.pt"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class YoloSegmenter:
    def __init__(self, conf=0.05, weights=None):
        self.conf = conf
        path = weights or os.environ.get("YOLO_WEIGHTS") or os.path.join(SCRIPT_DIR, YOLO_MODEL_NAME)
        try:
            from ultralytics import YOLO
            print(f"Loading {path} for instance segmentation (conf >= {conf})...", flush=True)
            self.model = YOLO(path)
        except Exception as e:  # make the failure explicit instead of silently disabling perception
            raise RuntimeError(
                f"Could not load the YOLO26 segmentation weights '{path}': {e}\n"
                f"Install requirements.txt, and either allow Ultralytics to download '{YOLO_MODEL_NAME}' "
                f"(internet needed once) or place the file next to yolo_segmenter.py / set YOLO_WEIGHTS.") from e
        self.enabled = True
        print("YOLO segmentation ready.", flush=True)

    def predict(self, rgb):
        """rgb: HxWx3 uint8 (RGB). Returns the raw Ultralytics Results object."""
        return self.model(rgb[:, :, ::-1], conf=self.conf, verbose=False)[0]  # Ultralytics expects BGR
