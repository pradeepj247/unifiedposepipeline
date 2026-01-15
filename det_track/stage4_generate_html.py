#!/usr/bin/env python3
"""
Stage 4: Generate HTML Viewer with OSNet Clustering

Loads pre-extracted crops from Stage 3c and generates WebP animations with HTML viewer.
Includes OSNet clustering for ReID-based duplicate detection.

Key Changes (Phase 5):
- Now loads crops from final_crops.pkl (created by Stage 3c)
- No video extraction in Stage 4 (eliminates redundant video scanning)
- Uses quality metrics from Stage 3c for crop selection
- Much faster (~5s vs ~16s previously)

Algorithm:
1. Load final_crops.pkl from Stage 3c
2. Select best N crops per person using quality metrics
3. Extract OSNet features and compute similarity matrix
4. Generate WebP animations from all 50 crops
5. Create unified HTML viewer with similarity heatmap

Usage:
    python stage4_generate_html.py --config configs/pipeline_config.yaml
"""

import argparse
import yaml
import numpy as np
import time
import re
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Import the on-demand extraction module
from ondemand_crop_extraction import generate_webp_animations
from crop_utils import load_final_crops

# Import OSNet clustering
try:
    from osnet_clustering import create_similarity_matrix, save_similarity_results
    OSNET_AVAILABLE = True
except ImportError:
    OSNET_AVAILABLE = False


sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import PipelineLogger


def resolve_path_variables(config):
    """Recursively resolve ${variable} in config"""
    global_vars = config.get('global', {})
    
    def resolve_string_once(s, vars_dict):
        if not isinstance(s, str):
            return s
        pattern = re.compile(r'\$\{(\w+)\}')
        return pattern.sub(lambda m: str(vars_dict.get(m.group(1), m.group(0))), s)
    
    def resolve_dict(d, vars_dict):
        for key, value in d.items():
            if isinstance(value, dict):
                resolve_dict(value, vars_dict)
            elif isinstance(value, list):
                d[key] = [resolve_string_once(item, vars_dict) if isinstance(item, str) else item for item in value]
            elif isinstance(value, str):
                d[key] = resolve_string_once(value, vars_dict)
    
    # Multi-pass resolution
    for _ in range(5):
        old_config = str(config)
        resolve_dict(config, global_vars)
        resolve_dict(config, config.get('global', {}))
        if str(config) == old_config:
            break
    
    return config


def enhance_html_with_similarity(html_file: Path, clustering_result: dict, person_buckets: dict) -> None:
    """
    Enhance existing HTML viewer with similarity matrix heatmap visualization.
    
    Args:
        html_file: Path to existing viewer.html
        clustering_result: Dict from create_similarity_matrix() with 'similarity_matrix', 'person_ids', etc.
        person_buckets: Dict of person_id -> crops
    """
    import json
    
    if not html_file.exists():
        print(f"[ERROR] HTML file not found: {html_file}")
        return
    
    # Read existing HTML
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract data from clustering result
    similarity_matrix = clustering_result.get('similarity_matrix', [])
    person_ids = clustering_result.get('person_ids', [])
    high_pairs = clustering_result.get('high_similarity_pairs', [])
    
    if not person_ids or not len(similarity_matrix):
        return  # No data to add
    
    # Create JavaScript code to embed similarity data
    similarity_data_js = f"""
    <script type="application/json" id="similarity-data">
    {{
        "person_ids": {json.dumps(person_ids)},
        "similarity_matrix": {json.dumps(similarity_matrix.tolist() if hasattr(similarity_matrix, 'tolist') else similarity_matrix)},
        "high_similarity_pairs": {json.dumps(high_pairs)}
    }}
    </script>
    """
    
    # Create HTML section for similarity heatmap
    heatmap_html = """
    <div id="similarity-section" style="margin-top: 40px; padding: 20px; background: #2a2a2a; border-radius: 8px;">
        <h2 style="color: #4CAF50; margin-top: 0;">🔍 Person Similarity Matrix (ReID)</h2>
        <p style="color: #aaa; margin-bottom: 20px;">Cosine similarity between person embeddings. Higher values = more likely same person.</p>
        
        <div id="similarity-heatmap" style="overflow-x: auto; margin-bottom: 20px;">
            <canvas id="heatmap-canvas" style="border: 1px solid #444; display: block; margin: 0 auto;"></canvas>
        </div>
        
        <div id="high-similarity-pairs" style="background: #1a1a1a; padding: 15px; border-radius: 4px; margin-top: 20px;">
            <h3 style="color: #4CAF50; margin-top: 0;">⚠️ High Similarity Pairs (potential duplicates)</h3>
            <div id="pairs-list" style="color: #e0e0e0;"></div>
        </div>
    </div>
    
    <script>
    // Parse similarity data
    const simDataElement = document.getElementById('similarity-data');
    if (simDataElement) {
        const data = JSON.parse(simDataElement.textContent);
        const personIds = data.person_ids;
        const matrix = data.similarity_matrix;
        const highPairs = data.high_similarity_pairs;
        
        // Draw heatmap
        const canvas = document.getElementById('heatmap-canvas');
        if (canvas && matrix.length > 0) {
            const cellSize = Math.max(30, Math.min(80, 600 / matrix.length));
            const size = personIds.length;
            canvas.width = size * cellSize + 50;
            canvas.height = size * cellSize + 50;
            
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw cells
            for (let i = 0; i < size; i++) {
                for (let j = 0; j < size; j++) {
                    const value = matrix[i][j];
                    const hue = (1 - value) * 120; // Red (0°) to Green (120°)
                    ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
                    ctx.fillRect(50 + j * cellSize, 50 + i * cellSize, cellSize, cellSize);
                    
                    // Draw text for high values
                    if (value > 0.7) {
                        ctx.fillStyle = '#000';
                        ctx.font = 'bold 10px Arial';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(value.toFixed(2), 50 + j * cellSize + cellSize/2, 50 + i * cellSize + cellSize/2);
                    }
                }
            }
            
            // Draw labels
            ctx.fillStyle = '#aaa';
            ctx.font = '11px Arial';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (let i = 0; i < size; i++) {
                ctx.fillText(`P${personIds[i]}`, 45, 50 + i * cellSize + cellSize/2);
                ctx.textAlign = 'center';
                ctx.fillText(`P${personIds[i]}`, 50 + i * cellSize + cellSize/2, 40);
            }
        }
        
        // Display high similarity pairs
        const pairsList = document.getElementById('pairs-list');
        if (highPairs.length > 0) {
            pairsList.innerHTML = highPairs.map(([p1, p2, score]) => 
                `<div style="padding: 8px; margin: 5px 0; background: #333; border-left: 3px solid hsl(${(1-score)*120}, 100%, 50%); border-radius: 4px;">
                    <strong>Person ${p1} ↔ Person ${p2}</strong>: ${(score*100).toFixed(1)}% similar
                </div>`
            ).join('');
        } else {
            pairsList.innerHTML = '<div style="color: #888;">No high-similarity pairs found (threshold > 0.70)</div>';
        }
    }
    </script>
    """
    
    # Insert before closing body tag
    if '</body>' in html_content:
        html_content = html_content.replace('</body>', heatmap_html + '\n</body>')
    else:
        html_content += heatmap_html
    
    # Insert similarity data script before closing head or at the start of body
    if '</head>' in html_content:
        html_content = html_content.replace('</head>', similarity_data_js + '\n</head>')
    else:
        html_content = similarity_data_js + html_content
    
    # Write updated HTML
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description='Stage 4: Generate HTML Viewer')
    parser.add_argument('--config', type=str, required=True, help='Path to pipeline config YAML')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-extract current_video from video_file (needed for path resolution)
    video_file = config.get('global', {}).get('video_file', '')
    if video_file:
        import os
        video_name = os.path.splitext(video_file)[0]
        config['global']['current_video'] = video_name
    
    config = resolve_path_variables(config)
    
    # Extract configuration
    global_config = config.get('global', {})
    stage_config = config.get('stage4_generate_html', {})
    
    # Input/output paths
    video_path = stage_config.get('video_file')  # Use canonical video from stage config
    canonical_persons_file = stage_config.get('canonical_persons_file')
    output_dir = stage_config.get('output_dir')
    
    # Load final_crops.pkl from Stage 3c (NEW Phase 5)
    # Stage 3c saves to its own output directory (where primary_person.npz is), not to stage4's output_dir
    primary_person_file = config.get('stage3c_rank', {}).get('output', {}).get('primary_person_file', '')
    if primary_person_file:
        stage3c_output_dir = Path(primary_person_file).parent
        final_crops_path = stage3c_output_dir / 'final_crops.pkl'
    else:
        # Fallback: use stage4's output_dir
        final_crops_path = Path(output_dir) / 'final_crops.pkl'
    
    # Parameters
    resize_to = tuple(stage_config.get('resize_to', [256, 256]))
    webp_duration_ms = stage_config.get('webp_duration_ms', 100)
    
    # Clustering parameters
    clustering_config = stage_config.get('clustering', {})
    clustering_enabled = clustering_config.get('enabled', True)
    osnet_model_path = clustering_config.get('osnet_model', None)
    osnet_fallback_model_path = clustering_config.get('osnet_model_fallback', None)
    device = clustering_config.get('device', 'cuda')
    num_best_crops = clustering_config.get('num_best_crops', 16)
    similarity_threshold = clustering_config.get('similarity_threshold', 0.70)
    
    # Logging
    log_file = stage_config.get('log_file')
    verbose = stage_config.get('advanced', {}).get('verbose', False) or config.get('global', {}).get('verbose', False)
    logger = PipelineLogger("Stage 4: Generate HTML Viewer", verbose=verbose)
    
    logger.header()
    if verbose:
        logger.info(f"Final crops: {final_crops_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Configuration:")
        logger.info(f"  - Resize to: {resize_to}")
        logger.info(f"  - WebP duration: {webp_duration_ms}ms")
        logger.info(f"  - OSNet Clustering: {'ENABLED' if clustering_enabled and OSNET_AVAILABLE else 'DISABLED'}")
        if clustering_enabled and OSNET_AVAILABLE:
            logger.info(f"    - Best crops per person: {num_best_crops}")
            logger.info(f"    - Similarity threshold: {similarity_threshold:.0%}")
            logger.info(f"    - Device: {device}")
        print()
    
    # ==================== STAGE 4a: Load Crops ====================
    if verbose:
        logger.step("Loading crops from Stage 3c...")
    
    try:
        crops_data = load_final_crops(final_crops_path, verbose=verbose)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Error loading final_crops.pkl: {e}")
        return 1
    
    # Convert pickle format to person_buckets format
    person_buckets = {}
    person_metadata = {}
    
    for person_id in crops_data['person_ids']:
        crops = crops_data['crops'][person_id]
        metadata = crops_data['metadata'][person_id]
        
        # Convert numpy array of images to list of images
        person_buckets[person_id] = [crops[i] for i in range(crops.shape[0])]
        person_metadata[person_id] = metadata
    
    if verbose:
        total_crops = sum(len(c) for c in person_buckets.values())
        logger.info(f"Loaded {len(person_buckets)} persons, {total_crops} crops")
    else:
        total_crops = sum(len(c) for c in person_buckets.values())
        print(f"   Loaded {len(person_buckets)} persons, {total_crops} crops from final_crops.pkl")
    
    # ==================== STAGE 4b: Clustering (with Quality-Aware Crop Selection) ====================
    clustering_time = 0
    clustering_result = None
    
    if clustering_enabled and OSNET_AVAILABLE:
        if verbose:
            print()
            logger.step("Preparing crops for OSNet clustering (quality-aware selection)...")
        
        clustering_start = time.time()
        
        try:
            # Select best N crops per person using quality metrics
            best_crops_buckets = {}
            
            for person_id, metadata_list in person_metadata.items():
                crops = person_buckets[person_id]
                
                # Sort by quality_rank (lower rank = better quality)
                ranked_indices = sorted(
                    range(len(metadata_list)),
                    key=lambda i: metadata_list[i].get('quality_rank', i)
                )
                
                # Take top num_best_crops
                best_indices = ranked_indices[:min(num_best_crops, len(ranked_indices))]
                best_crops_buckets[person_id] = [crops[i] for i in best_indices]
                
                if verbose:
                    logger.verbose_info(
                        f"Person {person_id}: Selected {len(best_indices)} crops "
                        f"(quality ranks: {[metadata_list[i].get('quality_rank', '?') for i in best_indices[:5]]}...)"
                    )
            
            # Extract OSNet features and compute similarity
            clustering_result = create_similarity_matrix(
                buckets=best_crops_buckets,
                osnet_model_path=osnet_model_path,
                osnet_fallback_model_path=osnet_fallback_model_path,
                device=device,
                num_best_crops=num_best_crops,
                similarity_threshold=similarity_threshold,
                verbose=verbose
            )
            clustering_time = time.time() - clustering_start
            
            # Save results
            if verbose:
                logger.step("Saving similarity matrix and features...")
            output_path = Path(output_dir)
            save_similarity_results(
                results=clustering_result,
                output_dir=output_path,
                verbose=verbose
            )
            
            # Report clustering details
            if verbose:
                logger.timing("OSNet clustering", clustering_time)
                model_type = clustering_result.get('model_type', 'unknown')
                num_pairs = len(clustering_result.get('high_similarity_pairs', []))
                logger.info(f"  Model: {model_type} | Persons: {len(best_crops_buckets)} | High-similarity pairs: {num_pairs}")
                logger.info(f"High-similarity pairs (>{similarity_threshold}):")
                for p1, p2, score in clustering_result['high_similarity_pairs'][:10]:
                    logger.info(f"  - Person {p1} & {p2}: {score:.3f}")
            else:
                logger.info(f"OSNet clustering completed in {clustering_time:.2f}s")
                timing_breakdown = clustering_result.get('timing', {})
                if timing_breakdown:
                    logger.info(f"  - Model loading: {timing_breakdown.get('load_model', 0):.2f}s")
                    logger.info(f"  - Feature extraction: {timing_breakdown.get('extract_features', 0):.2f}s")
                    logger.info(f"  - Similarity computation: {timing_breakdown.get('similarity', 0):.2f}s")
        
        except Exception as e:
            logger.warning(f"OSNet clustering failed (non-fatal): {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            clustering_enabled = False
    
    elif clustering_enabled and not OSNET_AVAILABLE:
        logger.warning("OSNet clustering requested but module not available")
        clustering_enabled = False
    
    # ==================== STAGE 4c: Generate WebPs ====================
    if verbose:
        print()
        logger.step("Generating WebP animations from all 50 crops...")
    
    webp_start = time.time()
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generate_webp_animations(
            person_buckets=person_buckets,
            output_dir=output_path,
            metadata=None,  # No metadata needed since crops are already loaded
            resize_to=resize_to,
            duration_ms=webp_duration_ms,
            verbose=verbose
        )
        webp_time = time.time() - webp_start
        if verbose:
            logger.timing("WebP generation", webp_time)
    except Exception as e:
        logger.error(f"Error during WebP generation: {e}")
        return 1
    
    # ==================== STAGE 4d: Create HTML ====================
    if verbose:
        print()
        logger.step("Creating HTML viewer with similarity heatmap...")
    
    if clustering_enabled and clustering_result:
        # Enhance HTML with similarity heatmap
        html_file = output_path / "viewer.html"
        enhance_html_with_similarity(html_file, clustering_result, person_buckets)
        if verbose:
            logger.info("Similarity matrix embedded in HTML viewer")
    
    # ==================== Summary ====================
    total_time = webp_time + clustering_time
    
    if verbose:
        print()
        print("=" * 70)
        logger.info(f"Stage 4 Timing breakdown:")
        logger.info(f"  - WebP generation: {webp_time:.2f}s")
        if clustering_enabled:
            logger.info(f"  - OSNet clustering: {clustering_time:.2f}s")
        logger.info(f"  - Total: {total_time:.2f}s")
        print()
        logger.info(f"Output:")
        logger.info(f"  - WebP files: {output_path}")
        logger.info(f"  - HTML viewer: {output_path / 'viewer.html'}")
        if clustering_enabled:
            logger.info(f"  - Similarity matrix: {output_path / 'similarity_matrix.json'}")
        print()
        logger.verbose_info(f"Architecture improvement:")
        logger.verbose_info(f"  - Stage 3c: Extracts crops + saves final_crops.pkl")
        logger.verbose_info(f"  - Stage 4: Loads crops, no video scanning (69% faster)")
        print()
    
    logger.success()
    
    # Save timing sidecar for run_pipeline.py
    output_path = Path(output_dir)
    sidecar_path = output_path / 'stage4.timings.json'
    try:
        sidecar_data = {
            'stage': 'stage4',
            'approach': 'Phase 5: Load from final_crops.pkl',
            'webp_time': float(webp_time),
            'clustering_time': float(clustering_time) if clustering_enabled else 0.0,
            'total_time': float(total_time),
            'num_persons': len(person_buckets),
            'total_crops': sum(len(c) for c in person_buckets.values()),
            'clustering_enabled': clustering_enabled,
            'high_similarity_pairs': len(clustering_result['high_similarity_pairs']) if clustering_result else 0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar_data, f, indent=2)
    except Exception:
        pass  # Non-fatal
    
    return 0


if __name__ == '__main__':
    exit(main())
