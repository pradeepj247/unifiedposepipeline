#!/usr/bin/env python3
"""
Debug: Compare YOLO 720p vs Raw Detections vs Canonical

Traces bbox coordinate changes through the pipeline:
  1. YOLO at 720p (ground truth)
  2. Raw detections NPZ (what was stored)
  3. Canonical persons NPZ (final output)

Usage:
    python debug_trace_bbox_corruption.py --frame 5
"""

import argparse
import numpy as np
from pathlib import Path


def parse_yolo_720p_file(txt_file):
    """Parse YOLO 720p detection file"""
    detections = []
    
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    in_detections = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if 'DETECTIONS:' in line:
            in_detections = True
            i += 1
            continue
        
        if '---' in line or '===' in line:
            i += 1
            continue
        
        if not line or not in_detections:
            i += 1
            continue
        
        # Parse detection blocks
        if line.startswith('['):
            # This is a detection header like "[0] person"
            parts = line.split(']')
            idx_str = parts[0].replace('[', '')
            try:
                idx = int(idx_str)
            except:
                i += 1
                continue
            
            detection = {'index': idx}
            
            # Look ahead for bbox, size, confidence
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                
                if 'bbox:' in next_line:
                    # Parse: bbox: [x1:1280.0, y1:22.0, x2:1513.0, y2:669.0]
                    try:
                        bbox_str = next_line.split('[')[1].split(']')[0]
                        coords = {}
                        for coord in bbox_str.split(','):
                            k, v = coord.strip().split(':')
                            coords[k] = float(v)
                        detection['bbox'] = [coords['x1'], coords['y1'], coords['x2'], coords['y2']]
                    except:
                        pass
                
                elif 'size:' in next_line:
                    # Parse: size: 233.0×648.0
                    try:
                        size_str = next_line.split(': ')[1]
                        w, h = size_str.split('×')
                        detection['size'] = (float(w), float(h))
                    except:
                        pass
                
                elif 'confidence:' in next_line:
                    # Parse: confidence: 0.90
                    try:
                        conf_str = next_line.split(': ')[1]
                        detection['confidence'] = float(conf_str)
                    except:
                        pass
            
            if 'bbox' in detection:
                detections.append(detection)
        
        i += 1
    
    return detections


def load_raw_detections(npz_file, frame_num):
    """Load raw detections from NPZ at specific frame"""
    data = np.load(npz_file, allow_pickle=True)
    
    mask = data['frame_numbers'] == frame_num
    
    detections = []
    for idx, (bbox, conf) in enumerate(zip(data['bboxes'][mask], data['confidences'][mask])):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        detections.append({
            'index': idx,
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'size': (w, h),
            'confidence': float(conf)
        })
    
    return detections


def load_canonical_persons(npz_file, frame_num):
    """Load canonical person bboxes at specific frame"""
    data = np.load(npz_file, allow_pickle=True)
    persons = data['persons']
    
    persons_at_frame = []
    
    for person in persons:
        if frame_num in person['frame_numbers']:
            idx = np.where(person['frame_numbers'] == frame_num)[0][0]
            x1, y1, x2, y2 = person['bboxes'][idx]
            w = x2 - x1
            h = y2 - y1
            persons_at_frame.append({
                'person_id': person['person_id'],
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'size': (w, h),
                'confidence': float(person['confidences'][idx])
            })
    
    return persons_at_frame


def scale_720p_to_1080p(bbox_720p, scale_factor=1.5):
    """Convert 720p bbox coordinates back to 1080p equivalent"""
    x1, y1, x2, y2 = bbox_720p
    x1_scaled = x1 * scale_factor
    y1_scaled = y1 * scale_factor
    x2_scaled = x2 * scale_factor
    y2_scaled = y2 * scale_factor
    return [x1_scaled, y1_scaled, x2_scaled, y2_scaled]


def compare_bboxes(bbox1, bbox2, label1, label2):
    """Compare two bboxes and show differences"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    w1 = x2_1 - x1_1
    h1 = y2_1 - y1_1
    w2 = x2_2 - x1_2
    h2 = y2_2 - y1_2
    
    shift_x1 = x1_2 - x1_1
    shift_y1 = y1_2 - y1_1
    shift_x2 = x2_2 - x2_1
    shift_y2 = y2_2 - y2_1
    
    width_change = w2 - w1
    height_change = h2 - h1
    
    width_pct = (width_change / w1 * 100) if w1 > 0 else 0
    height_pct = (height_change / h1 * 100) if h1 > 0 else 0
    
    print(f"\n{label1}:")
    print(f"  bbox: [x1:{x1_1:.1f}, y1:{y1_1:.1f}, x2:{x2_1:.1f}, y2:{y2_1:.1f}]")
    print(f"  size: {w1:.1f}×{h1:.1f}")
    
    print(f"\n{label2}:")
    print(f"  bbox: [x1:{x1_2:.1f}, y1:{y1_2:.1f}, x2:{x2_2:.1f}, y2:{y2_2:.1f}]")
    print(f"  size: {w2:.1f}×{h2:.1f}")
    
    print(f"\n📊 SHIFT:")
    print(f"  Δx1: {shift_x1:+.1f}, Δy1: {shift_y1:+.1f}, Δx2: {shift_x2:+.1f}, Δy2: {shift_y2:+.1f}")
    print(f"  Width: {w1:.1f} → {w2:.1f} ({width_change:+.1f}px, {width_pct:+.1f}%)")
    print(f"  Height: {h1:.1f} → {h2:.1f} ({height_change:+.1f}px, {height_pct:+.1f}%)")
    
    return {
        'shift_x1': shift_x1,
        'shift_y1': shift_y1,
        'shift_x2': shift_x2,
        'shift_y2': shift_y2,
        'width_change': width_change,
        'height_change': height_change,
        'width_pct': width_pct,
        'height_pct': height_pct
    }


def main():
    parser = argparse.ArgumentParser(description='Trace bbox corruption through pipeline')
    parser.add_argument('--frame', type=int, required=True,
                       help='Frame number to analyze')
    parser.add_argument('--yolo-720p', type=str,
                       default='/content/unifiedposepipeline/demo_data/outputs/kohli_nets/p720/frame_{:04d}.txt',
                       help='YOLO 720p detection file pattern')
    parser.add_argument('--raw-npz', type=str,
                       default='/content/unifiedposepipeline/demo_data/outputs/kohli_nets/detections_raw.npz',
                       help='Raw detections NPZ file')
    parser.add_argument('--canonical-npz', type=str,
                       default='/content/unifiedposepipeline/demo_data/outputs/kohli_nets/canonical_persons.npz',
                       help='Canonical persons NPZ file')
    args = parser.parse_args()
    
    frame_num = args.frame
    
    # Format filenames
    yolo_720p_file = args.yolo_720p.format(frame_num)
    
    print(f"\n{'='*80}")
    print(f"🔍 BBOX CORRUPTION TRACE - Frame {frame_num}")
    print(f"{'='*80}\n")
    
    # Load YOLO 720p
    print(f"📂 Loading YOLO 720p: {yolo_720p_file}")
    if not Path(yolo_720p_file).exists():
        print(f"❌ File not found!")
        return
    
    yolo_720p_dets = parse_yolo_720p_file(yolo_720p_file)
    print(f"   ✅ Found {len(yolo_720p_dets)} detections")
    
    # Load raw detections
    print(f"\n📂 Loading raw detections: {args.raw_npz}")
    if not Path(args.raw_npz).exists():
        print(f"❌ File not found!")
        return
    
    raw_dets = load_raw_detections(args.raw_npz, frame_num)
    print(f"   ✅ Found {len(raw_dets)} detections at frame {frame_num}")
    
    # Load canonical persons
    print(f"\n📂 Loading canonical persons: {args.canonical_npz}")
    if not Path(args.canonical_npz).exists():
        print(f"❌ File not found!")
        return
    
    canonical_persons = load_canonical_persons(args.canonical_npz, frame_num)
    print(f"   ✅ Found {len(canonical_persons)} persons at frame {frame_num}")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS: YOLO 720p → Raw Detections → Canonical Persons")
    print(f"{'='*80}\n")
    
    # Compare YOLO 720p with raw detections
    print(f"📊 Step 1: YOLO 720p vs Raw Detections")
    print(f"{'-'*80}\n")
    
    for i, yolo_det in enumerate(yolo_720p_dets):
        print(f"\n🔍 Detection [{i}]")
        print(f"-" * 80)
        
        # Scale 720p to 1080p equivalent
        bbox_720p = yolo_det['bbox']
        bbox_1080p_equiv = scale_720p_to_1080p(bbox_720p)
        
        # Find closest raw detection
        if i < len(raw_dets):
            raw_det = raw_dets[i]
            
            print(f"\n📍 YOLO 720p (scaled to 1080p equiv):")
            print(f"   Original 720p: [x1:{bbox_720p[0]:.1f}, y1:{bbox_720p[1]:.1f}, x2:{bbox_720p[2]:.1f}, y2:{bbox_720p[3]:.1f}]")
            print(f"   Scaled 1080p:  [x1:{bbox_1080p_equiv[0]:.1f}, y1:{bbox_1080p_equiv[1]:.1f}, x2:{bbox_1080p_equiv[2]:.1f}, y2:{bbox_1080p_equiv[3]:.1f}]")
            print(f"   Size: {yolo_det['size'][0]:.1f}×{yolo_det['size'][1]:.1f}")
            print(f"   Conf: {yolo_det['confidence']:.2f}")
            
            print(f"\n📍 Raw Detection [{i}]:")
            print(f"   bbox: [x1:{raw_det['bbox'][0]:.1f}, y1:{raw_det['bbox'][1]:.1f}, x2:{raw_det['bbox'][2]:.1f}, y2:{raw_det['bbox'][3]:.1f}]")
            print(f"   Size: {raw_det['size'][0]:.1f}×{raw_det['size'][1]:.1f}")
            print(f"   Conf: {raw_det['confidence']:.2f}")
            
            shift = compare_bboxes(bbox_1080p_equiv, raw_det['bbox'], "YOLO 720p (scaled)", "Raw Detection")
            
            # Check if raw detection is stored correctly
            if abs(shift['shift_x1']) < 1 and abs(shift['shift_y1']) < 1:
                print(f"\n   ✅ RAW DETECTION IS CORRECT (minimal shift)")
            else:
                print(f"\n   ⚠️  RAW DETECTION HAS SHIFT - Check YOLO/storage logic")
        else:
            print(f"   ❌ No matching raw detection!")
    
    # Compare raw detections with canonical persons
    print(f"\n\n{'='*80}")
    print(f"📊 Step 2: Raw Detections → Canonical Persons")
    print(f"{'-'*80}\n")
    
    for person in canonical_persons:
        print(f"\n🔍 Person {person['person_id']}")
        print(f"-" * 80)
        
        # Find closest raw detection
        person_center = ((person['bbox'][0] + person['bbox'][2]) / 2,
                        (person['bbox'][1] + person['bbox'][3]) / 2)
        
        min_dist = float('inf')
        closest_raw = None
        
        for raw_det in raw_dets:
            raw_center = ((raw_det['bbox'][0] + raw_det['bbox'][2]) / 2,
                         (raw_det['bbox'][1] + raw_det['bbox'][3]) / 2)
            dist = np.sqrt((person_center[0] - raw_center[0])**2 + 
                          (person_center[1] - raw_center[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_raw = raw_det
        
        if closest_raw:
            shift = compare_bboxes(closest_raw['bbox'], person['bbox'], 
                                  "Raw Detection (closest)", f"Canonical Person {person['person_id']}")
            
            if abs(shift['shift_x1']) < 2 and abs(shift['shift_y1']) < 2:
                print(f"\n   ✅ MINOR ADJUSTMENT (grouping/aggregation)")
            elif abs(shift['width_pct']) > 10 or abs(shift['height_pct']) > 10:
                print(f"\n   🚨 MAJOR SIZE CHANGE - Likely merging multiple tracklets!")
            else:
                print(f"\n   ⚠️  SHIFT DETECTED - Check Stage 4b logic")
        else:
            print(f"   ❌ No matching raw detection!")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
