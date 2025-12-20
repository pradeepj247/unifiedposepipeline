# Unified Pose Pipeline - Visual Workflow

```
╔════════════════════════════════════════════════════════════════════╗
║                    UNIFIED POSE PIPELINE WORKFLOW                  ║
╚════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────┐
│                         FRESH COLAB SESSION                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃  STEP 1: SETUP (setup_unified.py)            ┃
        ┃  ─────────────────────────────────────────    ┃
        ┃  🚀 ONE-TIME SETUP PER SESSION (~5-10 min)   ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ Step 0   │  │ Step 1-5 │  │ Step 6-9 │
              │ Mount    │  │ Install  │  │ Models & │
              │ Drive    │  │ Packages │  │ Data     │
              └──────────┘  └──────────┘  └──────────┘
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                        ┌─────────────────┐
                        │   ENVIRONMENT   │
                        │     READY       │
                        └─────────────────┘
                                  │
                                  ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃  STEP 2: VERIFY (verify.py)                  ┃
        ┃  ─────────────────────────────────────────    ┃
        ┃  🔍 COMPREHENSIVE CHECK (~30 sec)             ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
      ┌──────────┐          ┌──────────┐          ┌──────────┐
      │ Imports  │          │ Models   │          │   GPU    │
      │ ✅ Torch │          │ ✅ YOLO  │          │ ✅ CUDA  │
      │ ✅ CV2   │          │ ✅ ViT   │          │ ✅ Device│
      │ ✅ YOLO  │          │ ✅ RTM   │          │ ✅ ONNX  │
      │ ✅ RTM   │          │          │          │          │
      └──────────┘          └──────────┘          └──────────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  ▼
                        ┌─────────────────┐
                        │  ALL CHECKS     │
                        │     PASSED      │
                        └─────────────────┘
                                  │
                                  ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃  STEP 3: RUN (udp.py --config <file.yaml>)   ┃
        ┃  ─────────────────────────────────────────    ┃
        ┃  🎯 CONFIG-DRIVEN EXECUTION                   ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                  │
                                  ▼
                        ┌─────────────────┐
                        │  LOAD CONFIG    │
                        │  configs/*.yaml │
                        └─────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          ┌──────────────────┐        ┌──────────────────┐
          │   DETECTION      │        │  POSE ESTIMATION │
          │   ─────────      │        │  ───────────────  │
          │   • YOLO Model   │        │  • ViTPose OR    │
          │   • Confidence   │        │  • RTMPose       │
          │   • Person Det   │        │  • Keypoints     │
          └──────────────────┘        └──────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                        ┌─────────────────┐
                        │  PROCESS INPUT  │
                        │  ─────────────  │
                        │  Image or Video │
                        └─────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          ┌──────────────────┐        ┌──────────────────┐
          │   IMAGE MODE     │        │   VIDEO MODE     │
          │   ──────────     │        │   ──────────     │
          │ 1. Detect boxes  │        │ 1. Frame loop    │
          │ 2. Estimate pose │        │ 2. Detect+Pose   │
          │ 3. Draw results  │        │ 3. Draw+Write    │
          │ 4. Save output   │        │ 4. Progress bar  │
          └──────────────────┘        └──────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                        ┌─────────────────┐
                        │  OUTPUT         │
                        │  ──────         │
                        │  • Annotated    │
                        │  • Statistics   │
                        │  • JSON (opt)   │
                        └─────────────────┘
                                  │
                                  ▼
                            ┏━━━━━━━━┓
                            ┃  DONE  ┃
                            ┗━━━━━━━━┛

════════════════════════════════════════════════════════════════════

                         CONFIGURATION SYSTEM

┌─────────────────────────────────────────────────────────────────────┐
│                        CONFIG FILE (YAML)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  detection:                        ◄─── What YOLO model to use     │
│    model_path: models/yolo/*.pt                                    │
│    confidence_threshold: 0.5                                       │
│                                                                     │
│  pose_estimation:                  ◄─── What pose method to use    │
│    type: vitpose | rtmlib                                          │
│    model_path: ...                                                 │
│                                                                     │
│  input:                            ◄─── What to process            │
│    type: image | video                                             │
│    path: demo_data/...                                             │
│                                                                     │
│  output:                           ◄─── Where to save results      │
│    path: demo_data/outputs/...                                     │
│    save_json: true/false                                           │
│                                                                     │
│  processing:                       ◄─── How to process             │
│    max_frames: 100 | null                                          │
│    device: cuda | cpu                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════

                           DATA FLOW

┌─────────┐         ┌─────────┐         ┌──────────┐
│  Input  │────────▶│ YOLO    │────────▶│ Bounding │
│  Image/ │         │ Detect  │         │ Boxes    │
│  Video  │         │ Persons │         │          │
└─────────┘         └─────────┘         └──────────┘
                                              │
                                              ▼
                                        ┌──────────┐
                                        │ ViTPose  │
                                        │   OR     │
                                        │ RTMPose  │
                                        │ Estimate │
                                        └──────────┘
                                              │
                                              ▼
                                        ┌──────────┐
                                        │ Keypoint │
                                        │ Coords   │
                                        │  (x,y)   │
                                        └──────────┘
                                              │
                                              ▼
                                        ┌──────────┐
                                        │ Visualize│
                                        │ • Boxes  │
                                        │ • Points │
                                        │ • Skel   │
                                        └──────────┘
                                              │
                                              ▼
                                        ┌──────────┐
                                        │  Output  │
                                        │  File    │
                                        └──────────┘

════════════════════════════════════════════════════════════════════

                         MODEL SELECTION

                    ┌──────────────────────┐
                    │   DETECTION (YOLO)   │
                    └──────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  ┌──────────┐          ┌──────────┐          ┌──────────┐
  │ yolov8n  │          │ yolov8s  │          │ yolov8m  │
  │ FASTEST  │          │ BALANCED │          │ ACCURATE │
  │ ⚡⚡⚡    │          │ ⚡⚡     │          │ ⚡       │
  └──────────┘          └──────────┘          └──────────┘

                    ┌──────────────────────┐
                    │ POSE ESTIMATION      │
                    └──────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
  ┌──────────────┐                          ┌──────────────┐
  │  RTMPose     │                          │   ViTPose    │
  │  (FASTER)    │                          │  (ACCURATE)  │
  ├──────────────┤                          ├──────────────┤
  │ • rtmpose-m  │                          │ • vitpose-s  │
  │ • rtmpose-l  │                          │ • vitpose-b  │
  │ • rtmpose-x  │                          │ • vitpose-l  │
  │              │                          │ • vitpose-h  │
  │ 20-40 FPS    │                          │ 10-15 FPS    │
  └──────────────┘                          └──────────────┘

════════════════════════════════════════════════════════════════════

                      TYPICAL SESSION

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    GOOGLE COLAB NOTEBOOK                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  # Cell 1: Setup (run once per session)
  !python setup_unified.py
  # ⏱️  5-10 minutes
  # ✅ Installs everything
  ────────────────────────────────────────────────────────────────

  # Cell 2: Verify
  !python verify.py
  # ⏱️  30 seconds
  # ✅ All checks pass
  ────────────────────────────────────────────────────────────────

  # Cell 3: Quick image test
  !python udp.py --config configs/vitpose_demo.yaml
  # ⏱️  ~5 seconds
  # ✅ Image with pose overlay
  ────────────────────────────────────────────────────────────────

  # Cell 4: Video test (100 frames)
  !python udp.py --config configs/rtmlib_demo.yaml
  # ⏱️  ~10 seconds
  # ✅ Video with pose tracking
  ────────────────────────────────────────────────────────────────

  # Cell 5: View results
  from IPython.display import Image, Video
  Image('demo_data/outputs/vitpose_result.jpg')
  Video('demo_data/outputs/rtmlib_result.mp4')
  ────────────────────────────────────────────────────────────────

  # Cell 6: Full video (custom config)
  # Edit configs/video_demo.yaml: max_frames: null
  !python udp.py --config configs/video_demo.yaml
  # ⏱️  Depends on video length
  # ✅ Complete video processed

════════════════════════════════════════════════════════════════════

                        HELPER COMMANDS

  python run.py setup              # Same as setup_unified.py
  python run.py verify             # Same as verify.py
  python run.py demo vitpose       # Run ViTPose demo
  python run.py demo rtmlib        # Run RTMLib demo
  python run.py demo video         # Run video demo
  python run.py list-configs       # Show available configs
  python run.py help               # Show help

════════════════════════════════════════════════════════════════════

                      SUCCESS INDICATORS

  ✅ Setup:   "🎉 SETUP COMPLETE!"
  ✅ Verify:  "🎉 ALL CHECKS PASSED"
  ✅ Demo:    "✅ Pipeline completed successfully!"

  ❌ Problem: Check error message, run verify.py

════════════════════════════════════════════════════════════════════
```
