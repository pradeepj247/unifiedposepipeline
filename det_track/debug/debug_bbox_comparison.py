#!/usr/bin/env python3
"""
Debug: Compare Raw YOLO Detections vs Canonical Person Bboxes

Shows exactly where bbox shifts/corruption happens in the pipeline.

Usage:
    python debug_bbox_comparison.py --config configs/pipeline_config.yaml --frame 5
    python debug_bbox_comparison.py --config configs/pipeline_config.yaml --frames 5,206,366
"""

import argparse
import numpy as np
import cv2
import yaml
import re
from pathlib import Path


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config with multi-pass resolution"""
    max_passes = 5
    
    for _ in range(max_passes):
        global_vars = config.get('global', {})
        changed = False
        
        def resolve_string(s):
            nonlocal changed
            if not isinstance(s, str):
                return s
            
            def replace_var(match):
                nonlocal changed
                var_name = match.group(1)
                replacement = str(global_vars.get(var_name, match.group(0)))
                if replacement != match.group(0):
                    changed = True
                return replacement
            
            return re.sub(r'\$\{(\w+)\}', replace_var, s)
        
        def resolve_recursive(obj):
            if isinstance(obj, dict):
                return {k: resolve_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_recursive(v) for v in obj]
            elif isinstance(obj, str):
                return resolve_string(obj)
            return obj
        
        config = resolve_recursive(config)
        
        if not changed:
            break
    
    return config


def load_config(config_path):
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-extract current_video from video_file
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        video_name = Path(video_file).stem
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def load_raw_detections(detections_file, frame_num):
    """Load YOLO detections for a specific frame"""
    data = np.load(detections_file, allow_pickle=True)
    
    # Find detections for this frame
    mask = data['frame_numbers'] == frame_num
    
    return {
        'bboxes': data['bboxes'][mask],
        'confidences': data['confidences'][mask],
        'frame_count': np.sum(mask)
    }


def load_canonical_persons(canonical_file, frame_num):
    """Load canonical person bboxes for a specific frame"""
    data = np.load(canonical_file, allow_pickle=True)
    persons = data['persons']
    
    persons_at_frame = []
    
    for person in persons:
        if frame_num in person['frame_numbers']:
            idx = np.where(person['frame_numbers'] == frame_num)[0][0]
            persons_at_frame.append({
                'person_id': person['person_id'],
                'bbox': person['bboxes'][idx],
                'confidence': person['confidences'][idx]
            })
    
    return persons_at_frame


def find_closest_yolo_bbox(person_bbox, yolo_bboxes):
    """Find which YOLO bbox is closest to this person bbox (by IoU or center distance)"""
    if len(yolo_bboxes) == 0:
        return None, None
    
    px1, py1, px2, py2 = person_bbox
    person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
    
    min_dist = float('inf')
    closest_idx = 0
    
    for idx, yolo_bbox in enumerate(yolo_bboxes):
        yx1, yy1, yx2, yy2 = yolo_bbox
        yolo_center = ((yx1 + yx2) / 2, (yy1 + yy2) / 2)
        
        dist = np.sqrt((person_center[0] - yolo_center[0])**2 + 
                      (person_center[1] - yolo_center[1])**2)
        
        if dist < min_dist:
            min_dist = dist
            closest_idx = idx
    
    return closest_idx, min_dist


def analyze_bbox_shift(yolo_bbox, person_bbox):
    """Analyze coordinate shifts between YOLO and canonical person bbox"""
    yx1, yy1, yx2, yy2 = yolo_bbox
    px1, py1, px2, py2 = person_bbox
    
    yolo_w = yx2 - yx1
    yolo_h = yy2 - yy1
    person_w = px2 - px1
    person_h = py2 - py1
    
    return {
        'shift_x1': px1 - yx1,
        'shift_y1': py1 - yy1,
        'shift_x2': px2 - yx2,
        'shift_y2': py2 - yy2,
        'width_change': person_w - yolo_w,
        'height_change': person_h - yolo_h,
        'width_pct': ((person_w - yolo_w) / yolo_w * 100) if yolo_w > 0 else 0,
        'height_pct': ((person_h - yolo_h) / yolo_h * 100) if yolo_h > 0 else 0
    }


def create_comparison_image(frame, yolo_detections, canonical_persons):
    """Create side-by-side comparison image"""
    h, w = frame.shape[:2]
    
    # Create two panels
    panel_yolo = frame.copy()
    panel_canonical = frame.copy()
    
    # Draw YOLO detections (blue)
    for idx, bbox in enumerate(yolo_detections['bboxes']):
        x1, y1, x2, y2 = map(int, bbox)
        conf = yolo_detections['confidences'][idx]
        
        cv2.rectangle(panel_yolo, (x1, y1), (x2, y2), (255, 0, 0), 3)
        label = f"YOLO[{idx}] {conf:.2f}"
        cv2.putText(panel_yolo, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw canonical persons (green)
    for person in canonical_persons:
        x1, y1, x2, y2 = map(int, person['bbox'])
        pid = person['person_id']
        conf = person['confidence']
        
        cv2.rectangle(panel_canonical, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"P{pid} {conf:.2f}"
        cv2.putText(panel_canonical, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Stack side-by-side
    comparison = np.hstack([panel_yolo, panel_canonical])
    
    # Add title bar
    title_h = 50
    title_bar = np.zeros((title_h, comparison.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, "RAW YOLO DETECTIONS", (50, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(title_bar, "CANONICAL PERSONS", (w + 50, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    final = np.vstack([title_bar, comparison])
    
    return final


def generate_analysis_report(frame_num, yolo_detections, canonical_persons):
    """Generate detailed text analysis of bbox coordinates"""
    report = []
    report.append(f"\n{'='*80}")
    report.append(f"FRAME {frame_num} BBOX COORDINATE ANALYSIS")
    report.append(f"{'='*80}\n")
    
    # YOLO detections
    report.append("RAW YOLO DETECTIONS:")
    report.append("-" * 80)
    for idx, bbox in enumerate(yolo_detections['bboxes']):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        conf = yolo_detections['confidences'][idx]
        report.append(f"  [{idx}] bbox: [x1:{x1:.0f}, y1:{y1:.0f}, x2:{x2:.0f}, y2:{y2:.0f}] "
                     f"conf:{conf:.2f} size:{w:.0f}×{h:.0f}")
    report.append("")
    
    # Canonical persons
    report.append("CANONICAL PERSONS AT FRAME:")
    report.append("-" * 80)
    for person in canonical_persons:
        pid = person['person_id']
        px1, py1, px2, py2 = person['bbox']
        pw = px2 - px1
        ph = py2 - py1
        pconf = person['confidence']
        
        report.append(f"  Person {pid}: bbox: [x1:{px1:.0f}, y1:{py1:.0f}, x2:{px2:.0f}, y2:{py2:.0f}] "
                     f"conf:{pconf:.2f} size:{pw:.0f}×{ph:.0f}")
        
        # Find closest YOLO bbox
        closest_idx, dist = find_closest_yolo_bbox(person['bbox'], yolo_detections['bboxes'])
        
        if closest_idx is not None:
            yolo_bbox = yolo_detections['bboxes'][closest_idx]
            shifts = analyze_bbox_shift(yolo_bbox, person['bbox'])
            
            yx1, yy1, yx2, yy2 = yolo_bbox
            yw = yx2 - yx1
            yh = yy2 - yy1
            
            report.append(f"            CLOSEST to YOLO[{closest_idx}] (center distance: {dist:.1f}px)")
            report.append(f"            SHIFT: Δx1:{shifts['shift_x1']:+.0f}, Δy1:{shifts['shift_y1']:+.0f}, "
                         f"Δx2:{shifts['shift_x2']:+.0f}, Δy2:{shifts['shift_y2']:+.0f}")
            report.append(f"            WIDTH changed: {yw:.0f}→{pw:.0f} ({shifts['width_change']:+.0f}px, {shifts['width_pct']:+.1f}%)")
            report.append(f"            HEIGHT changed: {yh:.0f}→{ph:.0f} ({shifts['height_change']:+.0f}px, {shifts['height_pct']:+.1f}%)")
        else:
            report.append(f"            ⚠️  NO MATCHING YOLO DETECTION FOUND!")
        
        report.append("")
    
    report.append("="*80 + "\n")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Debug: Compare YOLO vs Canonical bboxes')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    parser.add_argument('--frame', type=int, default=None,
                       help='Single frame number to analyze')
    parser.add_argument('--frames', type=str, default=None,
                       help='Comma-separated frame numbers (e.g., 5,206,366)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get paths
    video_path = Path(config['global']['video_dir'] + config['global']['video_file']).resolve().absolute()
    detections_file = Path(config['stage1_detect']['output']['detections_file']).resolve().absolute()
    canonical_file = Path(config['stage4b_group_canonical']['output']['canonical_persons_file']).resolve().absolute()
    
    output_dir = Path(config['global']['outputs_dir']).resolve().absolute() / config['global']['current_video']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine frames to analyze
    if args.frames:
        frame_numbers = [int(f.strip()) for f in args.frames.split(',')]
    elif args.frame:
        frame_numbers = [args.frame]
    else:
        print("❌ Please specify --frame or --frames")
        return
    
    print(f"\n{'='*80}")
    print(f"🔍 BBOX COMPARISON DEBUG")
    print(f"{'='*80}\n")
    print(f"📂 Video: {video_path.name}")
    print(f"📊 Detections: {detections_file.name}")
    print(f"📊 Canonical: {canonical_file.name}")
    print(f"🎯 Analyzing frames: {frame_numbers}\n")
    
    # Check files
    if not detections_file.exists():
        print(f"❌ Detections file not found: {detections_file}")
        return
    
    if not canonical_file.exists():
        print(f"❌ Canonical persons file not found: {canonical_file}")
        return
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    # Process each frame
    for frame_num in frame_numbers:
        print(f"\n📍 Processing frame {frame_num}...")
        
        # Load data
        yolo_detections = load_raw_detections(detections_file, frame_num)
        canonical_persons = load_canonical_persons(canonical_file, frame_num)
        
        print(f"   YOLO detections: {yolo_detections['frame_count']}")
        print(f"   Canonical persons: {len(canonical_persons)}")
        
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"   ⚠️  Could not read frame {frame_num}")
            continue
        
        # Create comparison image
        comparison_img = create_comparison_image(frame, yolo_detections, canonical_persons)
        img_path = output_dir / f'comparison_frame_{frame_num:04d}.png'
        cv2.imwrite(str(img_path), comparison_img)
        print(f"   ✅ Saved: {img_path.name}")
        
        # Generate analysis report
        report = generate_analysis_report(frame_num, yolo_detections, canonical_persons)
        txt_path = output_dir / f'bbox_analysis_frame_{frame_num:04d}.txt'
        with open(txt_path, 'w') as f:
            f.write(report)
        print(f"   ✅ Saved: {txt_path.name}")
        
        # Print report to console
        print(report)
    
    cap.release()
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}\n")
    print(f"📦 Output: {output_dir}")
    print(f"   • comparison_frame_XXXX.png (visual comparison)")
    print(f"   • bbox_analysis_frame_XXXX.txt (coordinate analysis)")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
