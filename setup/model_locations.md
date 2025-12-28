# Model Locations

This file documents the models downloaded by `step2_install_models.py` during the setup run. Each block shows the model name (as printed), the filename, file type, filesystem location, file size (bytes and approximate MB), and the fetch method used.

---

Model Name: YOLOv8s
Filename: `yolov8s.pt`
Type: `.pt`
Location: `/content/models/yolo/yolov8s.pt`
Size: `22,588,772` bytes (~22.6 MB)
Fetch method: `GitHub (curl)`

---

Model Name: VITPose
Filename: `vitpose-b.pth`
Type: `.pth`
Location: `/content/models/vitpose/vitpose-b.pth`
Size: `360,038,314` bytes (~360.0 MB)
Fetch method: `GitHub (curl)`

---

Model Name: RTMPose (COCO)
Filename: `rtmpose-l-coco-384x288.onnx`
Type: `.onnx`
Location: `/content/models/rtmlib/rtmpose-l-coco-384x288.onnx`
Size: `111,117,190` bytes (~111.1 MB)
Fetch method: `GitHub (curl)`

---

Model Name: RTMPose (Halpe26)
Filename: `rtmpose-l-halpe26-384x288.onnx`
Type: `.onnx`
Location: `/content/models/rtmlib/rtmpose-l-halpe26-384x288.onnx`
Size: `112,923,563` bytes (~112.9 MB)
Fetch method: `GitHub (curl)`

---

Model Name: MotionAGFormer
Filename: `motionagformer-base-h36m.pth.tr`
Type: `.pth.tr` (checkpoint/converted)
Location: `/content/models/motionagformer/motionagformer-base-h36m.pth.tr`
Size: `141,930,389` bytes (~141.9 MB)
Fetch method: `Google Drive (gdown)`

---

Model Name: RTM Wholebody
Filename: `rtmw3d-l.onnx`
Type: `.onnx`
Location: `/content/models/wb3d/rtmw3d-l.onnx`
Size: `230,000,270` bytes (~230.0 MB)
Fetch method: `Drive copy (rtmw3d export copy)`

---

Model Name: OSNet x1.0 (PyTorch)
Filename: `osnet_x1_0_msmt17.pt`
Type: `.pt`
Location: `/content/models/reid/osnet_x1_0_msmt17.pt`
Size: `10,910,553` bytes (~10.9 MB)
Fetch method: `Google Drive (gdown)`

---

Model Name: OSNet x0.25 (PyTorch)
Filename: `osnet_x0_25_msmt17.pt`
Type: `.pt`
Location: `/content/models/reid/osnet_x0_25_msmt17.pt`
Size: `3,057,863` bytes (~3.1 MB)
Fetch method: `HuggingFace (wget)`

---

Model Name: OSNet x0.25 (ONNX)
Filename: `osnet_x0_25_msmt17.onnx`
Type: `.onnx`
Location: `/content/models/reid/osnet_x0_25_msmt17.onnx`
Size: `907,169` bytes (~0.9 MB)
Fetch method: `HuggingFace (wget)`

---

Notes:
- Sizes recorded from `ls -lR /content/models` (values in bytes and approximate MB shown).
- If you want a checksum column (MD5/SHA256) added, I can compute and append checksums for each file.
- After you confirm these entries, I can prune the download branches in `step2_install_models.py` to keep only the proven working fetch method for each model.
