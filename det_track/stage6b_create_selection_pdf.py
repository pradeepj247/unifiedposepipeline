#!/usr/bin/env python3
"""
Stage 6b Alternative: Create Person Selection PDF with Table and Thumbnails

Creates a professional PDF document showing:
- Table of all persons with statistics (duration, start frame, end frame, etc.)
- Thumbnail crops for easy visual review
- Sortable by duration, person ID, or other metrics

Usage:
    python stage6b_create_selection_pdf.py --config configs/pipeline_config.yaml
"""

import argparse
import numpy as np
import pickle
import yaml
import re
import os
import cv2
from pathlib import Path
import time
from datetime import timedelta

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("⚠️  reportlab not found. Install with: pip install reportlab")


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    def resolve_string_once(s, vars_dict):
        if not isinstance(s, str):
            return s
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(vars_dict.get(m.group(1), m.group(0))),
            s
        )
    
    max_iterations = 10
    for _ in range(max_iterations):
        resolved_globals = {}
        changed = False
        for key, value in global_vars.items():
            if isinstance(value, str):
                resolved = resolve_string_once(value, global_vars)
                resolved_globals[key] = resolved
                if resolved != value:
                    changed = True
            else:
                resolved_globals[key] = value
        global_vars = resolved_globals
        if not changed:
            break
    
    def resolve_string(s):
        return re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(global_vars.get(m.group(1), m.group(0))),
            s
        )
    
    def resolve_recursive(obj):
        if isinstance(obj, dict):
            return {k: resolve_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_recursive(v) for v in obj]
        elif isinstance(obj, str):
            return resolve_string(s)
        return obj
    
    result = resolve_recursive(config)
    result['global'] = global_vars
    return result


def load_config(config_path):
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def get_best_crop_for_person(person, crops_cache):
    """Get highest-confidence crop for person"""
    if person.get('frame_numbers') is None or len(person['frame_numbers']) == 0:
        return None
    
    confidences = person['confidences']
    best_idx = np.argmax(confidences)
    best_frame = int(person['frame_numbers'][best_idx])
    
    if best_frame in crops_cache:
        crops_in_frame = crops_cache[best_frame]
        for crop_image in crops_in_frame.values():
            if crop_image is not None and isinstance(crop_image, np.ndarray):
                return crop_image
    
    return None


def save_crop_to_temp_png(crop_bgr, temp_path):
    """Save BGR crop to temporary PNG file"""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(temp_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))


def create_selection_pdf(canonical_file, crops_cache_file, fps, output_pdf, temp_dir):
    """Create PDF with table and thumbnails"""
    
    # Load data
    print(f"📂 Loading canonical persons...")
    data = np.load(canonical_file, allow_pickle=True)
    persons = list(data['persons'])
    persons.sort(key=lambda p: len(p['frame_numbers']), reverse=True)
    
    print(f"📂 Loading crops cache...")
    with open(crops_cache_file, 'rb') as f:
        crops_cache = pickle.load(f)
    
    # Prepare table data
    print(f"🎨 Preparing table data...")
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    table_data = [
        ['Rank', 'Person ID', 'Duration', 'Frames', 'Start', 'End', 'Avg Conf', 'Thumbnail']
    ]
    
    crop_temp_files = []
    
    for rank, person in enumerate(persons[:20], 1):  # Top 20 persons
        person_id = person['person_id']
        frames = person['frame_numbers']
        durations = len(frames)
        start_frame = int(frames[0])
        end_frame = int(frames[-1])
        avg_conf = np.mean(person['confidences'])
        
        duration_seconds = durations / fps if fps > 0 else durations / 25
        duration_str = str(timedelta(seconds=int(duration_seconds)))
        
        # Get crop
        crop = get_best_crop_for_person(person, crops_cache)
        
        if crop is not None:
            temp_file = temp_dir / f"person_{person_id:03d}.png"
            save_crop_to_temp_png(crop, temp_file)
            crop_temp_files.append(str(temp_file))
            
            # For reportlab, we need to create an Image object
            row = [
                str(rank),
                str(person_id),
                duration_str,
                str(durations),
                str(start_frame),
                str(end_frame),
                f'{avg_conf:.3f}',
                str(temp_file)  # Placeholder - will be replaced with Image object
            ]
        else:
            row = [
                str(rank),
                str(person_id),
                duration_str,
                str(durations),
                str(start_frame),
                str(end_frame),
                f'{avg_conf:.3f}',
                '(no crop)'
            ]
        
        table_data.append(row)
    
    # Create PDF if reportlab available
    if not HAS_REPORTLAB:
        print(f"⚠️  reportlab not installed. Cannot create PDF.")
        print(f"   Install with: pip install reportlab")
        return False
    
    print(f"📄 Creating PDF document...")
    
    doc = SimpleDocTemplate(str(output_pdf), pagesize=A4, rightMargin=10, leftMargin=10,
                           topMargin=20, bottomMargin=20)
    
    story = []
    
    # Title
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("Person Selection Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Convert table with images
    final_table_data = []
    final_table_data.append(table_data[0])  # Header
    
    for i, row in enumerate(table_data[1:], 1):
        new_row = list(row)
        
        # Replace image path with actual RLImage if crop exists
        if row[-1] != '(no crop)' and Path(row[-1]).exists():
            try:
                img = RLImage(row[-1], width=0.8*inch, height=1.0*inch)
                new_row[-1] = img
            except:
                pass
        
        final_table_data.append(new_row)
    
    # Create table
    table = Table(final_table_data, colWidths=[0.5*inch, 0.7*inch, 0.8*inch, 0.6*inch,
                                               0.6*inch, 0.6*inch, 0.7*inch, 1.0*inch])
    
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), TA_CENTER),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    
    story.append(table)
    
    # Build PDF
    doc.build(story)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Stage 6b: Create Person Selection PDF'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration YAML')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    canonical_file = config['stage4b_group_canonical']['output']['canonical_persons_file']
    crops_cache_file = config['stage4a_reid_recovery']['input']['crops_cache_file']
    output_pdf = Path(config.get('stage6b_create_selection_pdf', {}).get('output', {}).get(
        'selection_pdf', 
        str(Path(canonical_file).parent / 'person_selection_report.pdf')
    ))
    
    fps = config.get('global', {}).get('video_fps', 25)
    
    print(f"\n{'='*70}")
    print(f"📄 STAGE 6b: CREATE PERSON SELECTION PDF")
    print(f"{'='*70}\n")
    
    t_start = time.time()
    
    success = create_selection_pdf(
        canonical_file,
        crops_cache_file,
        fps,
        output_pdf,
        output_pdf.parent / 'temp_crops'
    )
    
    t_end = time.time()
    
    if success:
        size_mb = output_pdf.stat().st_size / (1024 * 1024) if output_pdf.exists() else 0
        print(f"\n✅ PDF created: {output_pdf.name} ({size_mb:.2f} MB)")
        print(f"⏱️  Time: {t_end - t_start:.2f}s")
        print(f"\n{'='*70}\n")
        return True
    else:
        print(f"\n❌ Failed to create PDF")
        print(f"{'='*70}\n")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
