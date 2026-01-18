#!/usr/bin/env python3
"""
Stage 0: Video Normalization & Validation

This is the FIRST stage of the pipeline - runs BEFORE YOLO detection.
Ensures all videos are in a consistent, optimal format for processing.

Purpose:
    - Validate user uploads (size, duration, format)
    - Normalize to canonical format (MP4, H.264, constant FPS, optimal GOP)
    - Enable fast seeking and consistent tracking

Input:
    - Any video format (MP4, MOV, AVI, etc.)
    - Any resolution, FPS, codec

Output:
    - Canonical video: MP4, H.264, ≤1080p, constant FPS, GOP=30
    - Metadata JSON with validation results

Usage:
    python stage0_normalize_video.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import json
import subprocess
import sys
import time
import re
from pathlib import Path

def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    # Iteratively resolve variables
    max_iterations = 10
    for _ in range(max_iterations):
        resolved_globals = {}
        changed = False
        for key, value in global_vars.items():
            if isinstance(value, str):
                resolved = re.sub(
                    r'\$\{(\w+)\}',
                    lambda m: str(global_vars.get(m.group(1), m.group(0))),
                    value
                )
                if resolved != value:
                    changed = True
                resolved_globals[key] = resolved
            else:
                resolved_globals[key] = value
        
        global_vars = resolved_globals
        if not changed:
            break
    
    # Apply to entire config
    def resolve_dict(d):
        if isinstance(d, dict):
            return {k: resolve_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [resolve_dict(item) for item in d]
        elif isinstance(d, str):
            return re.sub(
                r'\$\{(\w+)\}',
                lambda m: str(global_vars.get(m.group(1), m.group(0))),
                d
            )
        else:
            return d
    
    result = resolve_dict(config)
    result['global'] = global_vars
    return result


def load_config(config_path):
    """Load and resolve YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-extract current_video from video_file
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        config['global']['current_video'] = video_name
    
    return resolve_path_variables(config)


def get_video_metadata(filepath):
    """
    Extract video metadata using ffprobe.
    
    Returns dict with: width, height, fps, duration, codec, etc.
    """
    print(f"  Extracting metadata from: {filepath}")
    
    # Use ffprobe for reliable metadata
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(filepath)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Failed to read video file")
        print(f"     {e.stderr}")
        return None
    except json.JSONDecodeError:
        print(f"  [ERROR] Invalid ffprobe output")
        return None
    
    # Find video stream
    video_stream = None
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video':
            video_stream = stream
            break
    
    if not video_stream:
        print(f"  [ERROR] No video stream found")
        return None
    
    # Extract FPS (handle fraction like "25/1")
    fps_str = video_stream.get('r_frame_rate', '0/1')
    try:
        num, den = map(int, fps_str.split('/'))
        fps = num / den if den != 0 else 0
    except:
        fps = 0
    
    # Build metadata dict
    metadata = {
        'width': int(video_stream.get('width', 0)),
        'height': int(video_stream.get('height', 0)),
        'fps': fps,
        'duration': float(data.get('format', {}).get('duration', 0)),
        'codec': video_stream.get('codec_name', 'unknown'),
        'pix_fmt': video_stream.get('pix_fmt', 'unknown'),
        'bitrate': int(data.get('format', {}).get('bit_rate', 0)),
        'frame_count': int(video_stream.get('nb_frames', 0)) if 'nb_frames' in video_stream else None,
        'filesize_mb': Path(filepath).stat().st_size / (1024 * 1024)
    }
    
    # Check for variable frame rate
    avg_fps_str = video_stream.get('avg_frame_rate', fps_str)
    metadata['is_vfr'] = (avg_fps_str != fps_str)
    
    return metadata


def validate_metadata(metadata, config):
    """
    Check if video meets requirements.
    
    Returns: (is_valid, checks_list)
    """
    stage_config = config.get('stage0_normalize', {})
    limits = stage_config.get('limits', {})
    
    checks = []
    is_valid = True
    
    # Check duration
    max_duration = limits.get('max_duration_seconds', 120)
    if metadata['duration'] > max_duration:
        checks.append(('FAIL', f"Duration {metadata['duration']:.1f}s exceeds limit ({max_duration}s)"))
        is_valid = False
    else:
        checks.append(('PASS', f"Duration OK ({metadata['duration']:.1f}s / {max_duration}s max)"))
    
    # Check file size
    max_size_mb = limits.get('max_filesize_mb', 200)
    if metadata['filesize_mb'] > max_size_mb:
        checks.append(('FAIL', f"File size {metadata['filesize_mb']:.1f} MB exceeds limit ({max_size_mb} MB)"))
        is_valid = False
    else:
        checks.append(('PASS', f"File size OK ({metadata['filesize_mb']:.1f} MB / {max_size_mb} MB max)"))
    
    # Check resolution (warning, not failure)
    max_res = limits.get('max_resolution', [1920, 1080])
    if metadata['width'] > max_res[0] or metadata['height'] > max_res[1]:
        checks.append(('WARN', f"High resolution ({metadata['width']}x{metadata['height']}), will downscale to {max_res[0]}x{max_res[1]}"))
    else:
        checks.append(('PASS', f"Resolution OK ({metadata['width']}x{metadata['height']})"))
    
    # Check FPS (warning, not failure)
    target_fps = stage_config.get('normalization', {}).get('target_fps', 25)
    if metadata['fps'] > 60:
        checks.append(('WARN', f"Very high FPS ({metadata['fps']:.1f}), will resample to {target_fps}"))
    elif metadata['fps'] < 10:
        checks.append(('WARN', f"Very low FPS ({metadata['fps']:.1f}), may affect tracking quality"))
    else:
        checks.append(('PASS', f"FPS OK ({metadata['fps']:.1f})"))
    
    # Check for variable frame rate
    if metadata.get('is_vfr'):
        checks.append(('WARN', "Variable FPS detected, will convert to constant"))
    
    # Check codec
    common_codecs = ['h264', 'hevc', 'h265', 'vp8', 'vp9', 'av1']
    if metadata['codec'] in common_codecs:
        checks.append(('PASS', f"Codec OK ({metadata['codec']})"))
    else:
        checks.append(('INFO', f"Unusual codec ({metadata['codec']}), will transcode"))
    
    return is_valid, checks


def needs_normalization(metadata, config):
    """
    Determine if video needs normalization.
    
    Returns: (needs_norm, reasons)
    """
    stage_config = config.get('stage0_normalize', {})
    normalization = stage_config.get('normalization', {})
    limits = stage_config.get('limits', {})
    
    reasons = []
    
    # Check resolution
    max_res = limits.get('max_resolution', [1920, 1080])
    if metadata['width'] > max_res[0] or metadata['height'] > max_res[1]:
        reasons.append(f"resolution ({metadata['width']}x{metadata['height']} > {max_res[0]}x{max_res[1]})")
    
    # Check FPS
    target_fps = normalization.get('target_fps', 25)
    if abs(metadata['fps'] - target_fps) > 0.5:  # Allow small tolerance
        reasons.append(f"FPS ({metadata['fps']:.1f} != {target_fps})")
    
    # Check VFR
    if metadata.get('is_vfr'):
        reasons.append("variable FPS")
    
    # Check codec (always normalize to H.264 for consistency)
    if metadata['codec'] != 'h264':
        reasons.append(f"codec ({metadata['codec']} != h264)")
    
    # Always normalize for GOP structure (unless we can detect it's already good)
    # For simplicity, we normalize if any other reason exists
    
    return len(reasons) > 0, reasons


def normalize_video(input_path, output_path, metadata, config):
    """
    Normalize video to canonical format.
    
    Returns: (success, elapsed_time)
    """
    stage_config = config.get('stage0_normalize', {})
    normalization = stage_config.get('normalization', {})
    encoding = stage_config.get('encoding', {})
    limits = stage_config.get('limits', {})
    
    print(f"\n  Normalizing video...")
    print(f"    Input:  {input_path}")
    print(f"    Output: {output_path}")
    
    # Build ffmpeg command with GPU acceleration
    # Hardware acceleration for decode
    cmd = ['ffmpeg', '-y']
    
    # Check if GPU encoding is enabled
    use_gpu = encoding.get('use_gpu', True)
    if use_gpu:
        cmd += ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
        print(f"    GPU: CUDA hardware acceleration enabled")
    
    cmd += ['-i', str(input_path)]
    
    # Resolution normalization with GPU scaling
    target_res = normalization.get('target_resolution', [1280, 720])
    if use_gpu:
        # GPU-based scaling
        scale_filter = f"scale_cuda={target_res[0]}:{target_res[1]}"
        print(f"    Resolution: {metadata['width']}x{metadata['height']} → {target_res[0]}x{target_res[1]} (GPU)")
    else:
        # CPU-based scaling (fallback)
        scale_filter = f"scale={target_res[0]}:{target_res[1]}"
        print(f"    Resolution: {metadata['width']}x{metadata['height']} → {target_res[0]}x{target_res[1]} (CPU)")
    
    cmd += ['-vf', scale_filter]
    
    # Video codec and settings
    if use_gpu:
        # NVENC GPU encoding
        cmd += [
            '-c:v', 'h264_nvenc',
            '-preset', encoding.get('gpu_preset', 'p4'),  # p1-p7, p4=balanced
            '-pix_fmt', encoding.get('pix_fmt', 'yuv420p'),
        ]
        # All I-frames for optimal seeking (no P/B frames)
        cmd += ['-g', '1', '-bf', '0']
        print(f"    Encoder: h264_nvenc (GPU) with all I-frames")
    else:
        # CPU encoding (fallback)
        cmd += [
            '-c:v', encoding.get('codec', 'libx264'),
            '-preset', encoding.get('preset', 'veryfast'),
            '-profile:v', encoding.get('profile', 'main'),
            '-pix_fmt', encoding.get('pix_fmt', 'yuv420p'),
        ]
        # GOP structure for CPU
        keyint = encoding.get('keyframe_interval', 30)
        cmd += ['-x264-params', f'keyint={keyint}:scenecut=0']
        print(f"    Encoder: libx264 (CPU), GOP keyint={keyint}")
    
    # FPS normalization
    target_fps = normalization.get('target_fps', 25)
    if abs(metadata['fps'] - target_fps) > 0.5 or metadata.get('is_vfr'):
        cmd += ['-r', str(target_fps), '-vsync', 'cfr']
        print(f"    FPS: {metadata['fps']:.1f} → {target_fps} (constant)")
    
    # Audio (remove to save space)
    cmd += ['-an']
    
    # Output
    cmd += [str(output_path)]
    
    print(f"    Running ffmpeg...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"    ✅ Normalization complete ({elapsed:.2f}s)")
        return True, elapsed
    
    except subprocess.CalledProcessError as e:
        print(f"    [ERROR] ffmpeg failed")
        print(f"       {e.stderr}")
        return False, 0


def run_stage0_normalize(config):
    """Main function for Stage 0"""
    
    stage_config = config.get('stage0_normalize', {})
    
    # Check if stage is enabled
    if not stage_config.get('enabled', True):
        print("  ⏭️  Stage 0 disabled, skipping normalization")
        return None
    
    # Get input video path (combine video_dir + video_file)
    input_video = config['global']['video_dir'] + config['global']['video_file']
    if not Path(input_video).exists():
        print(f"  [ERROR] Input video not found: {input_video}")
        sys.exit(1)
    
    # Get output path
    output_video = stage_config['output']['canonical_video_file']
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    
    # Timing
    timing = {
        'stage': 'stage0_normalize',
        'input_video': str(input_video),
        'start_time': time.time()
    }
    
    print("\n" + "="*70)
    print("STAGE 0: VIDEO NORMALIZATION & VALIDATION")
    print("="*70)
    print(f"Input video: {input_video}")
    print("-"*70)
    
    # Step 1: Extract metadata
    print("\n📊 Step 1: Extracting video metadata...")
    metadata_start = time.time()
    metadata = get_video_metadata(input_video)
    metadata_time = time.time() - metadata_start
    timing['metadata_extraction_time'] = metadata_time
    
    if not metadata:
        print("[ERROR] FAILED: Could not extract video metadata")
        sys.exit(1)
    
    print(f"  ✅ Metadata extracted ({metadata_time:.2f}s)")
    print(f"     Resolution: {metadata['width']}x{metadata['height']}")
    print(f"     FPS: {metadata['fps']:.2f}")
    print(f"     Duration: {metadata['duration']:.1f}s")
    print(f"     Codec: {metadata['codec']}")
    print(f"     File size: {metadata['filesize_mb']:.1f} MB")
    
    # Step 2: Validate
    print("\n✅ Step 2: Validating video...")
    is_valid, checks = validate_metadata(metadata, config)
    timing['validation_time'] = time.time() - metadata_start - metadata_time
    
    for status, message in checks:
        icon = {'PASS': '✅', 'WARN': '⚠️', 'INFO': 'ℹ️', 'FAIL': '❌'}[status]
        print(f"  {icon} {message}")
    
    if not is_valid:
        print("\n[ERROR] VALIDATION FAILED: Video does not meet requirements")
        sys.exit(1)
    
    # Step 3: Check if normalization needed
    print("\n🔍 Step 3: Checking if normalization needed...")
    needs_norm, reasons = needs_normalization(metadata, config)
    
    if not needs_norm:
        print("  ✅ Video already in canonical format, no normalization needed")
        print(f"     Using original video as canonical: {input_video}")
        
        # Create symlink or copy
        if stage_config.get('output', {}).get('symlink_if_canonical', True):
            if output_video != input_video:
                # Create symlink (or copy on Windows)
                try:
                    if Path(output_video).exists():
                        Path(output_video).unlink()
                    Path(output_video).symlink_to(Path(input_video).resolve())
                    print(f"     Created symlink: {output_video}")
                except OSError:
                    # Symlink failed (Windows?), copy instead
                    import shutil
                    shutil.copy2(input_video, output_video)
                    print(f"     Copied file: {output_video}")
        
        timing['normalization_time'] = 0
        timing['normalization_needed'] = False
    
    else:
        print(f"  ⚙️  Normalization needed: {', '.join(reasons)}")
        
        # Step 4: Normalize
        print("\n🔄 Step 4: Normalizing video...")
        norm_start = time.time()
        success, norm_time = normalize_video(input_video, output_video, metadata, config)
        timing['normalization_time'] = norm_time
        timing['normalization_needed'] = True
        timing['normalization_reasons'] = reasons
        
        if not success:
            print("[ERROR] FAILED: Video normalization failed")
            sys.exit(1)
    
    # Get canonical video metadata
    print("\n📊 Verifying canonical video...")
    canonical_metadata = get_video_metadata(output_video)
    if canonical_metadata:
        print(f"  ✅ Canonical video created:")
        print(f"     Resolution: {canonical_metadata['width']}x{canonical_metadata['height']}")
        print(f"     FPS: {canonical_metadata['fps']:.2f}")
        print(f"     Codec: {canonical_metadata['codec']}")
        print(f"     File size: {canonical_metadata['filesize_mb']:.1f} MB")
    
    # Save timing
    timing['end_time'] = time.time()
    timing['total_time'] = timing['end_time'] - timing['start_time']
    timing['canonical_video'] = str(output_video)
    timing['original_metadata'] = metadata
    timing['canonical_metadata'] = canonical_metadata
    
    timing_file = stage_config['output'].get('timing_file', 
                                             str(Path(output_video).parent / 'stage0_timing.json'))
    Path(timing_file).parent.mkdir(parents=True, exist_ok=True)
    with open(timing_file, 'w') as f:
        json.dump(timing, f, indent=2)
    
    print("\n" + "="*70)
    print("STAGE 0 COMPLETE")
    print("="*70)
    print(f"Total time: {timing['total_time']:.2f}s")
    print(f"  Metadata extraction: {timing['metadata_extraction_time']:.2f}s")
    print(f"  Normalization: {timing['normalization_time']:.2f}s")
    print(f"\nCanonical video: {output_video}")
    print(f"Timing saved: {timing_file}")
    print("="*70)
    
    return output_video


def main():
    parser = argparse.ArgumentParser(description='Stage 0: Video Normalization')
    parser.add_argument('--config', required=True, help='Path to pipeline config YAML')
    args = parser.parse_args()
    
    config = load_config(args.config)
    canonical_video = run_stage0_normalize(config)
    
    if canonical_video:
        print(f"\n✅ Stage 0 successful: {canonical_video}")
    else:
        print("\n⏭️  Stage 0 skipped")


if __name__ == '__main__':
    main()
