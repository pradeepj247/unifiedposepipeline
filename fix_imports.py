"""
Quick fix script to update MotionAGFormer imports from absolute to relative
Run this in Colab if git pull fails
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parent

def fix_file(filepath, old_import, new_import):
    """Replace imports in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {filepath.name}")
            return True
        else:
            print(f"⏭️  Skipped: {filepath.name} (already fixed or not found)")
            return False
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("🔧 Fixing MotionAGFormer Import Errors")
    print("="*70)
    
    fixes = [
        {
            'file': REPO_ROOT / 'lib' / 'motionagformer' / 'model' / 'MotionAGFormer.py',
            'replacements': [
                ('from model.modules.attention import Attention', 'from .modules.attention import Attention'),
                ('from model.modules.graph import GCN', 'from .modules.graph import GCN'),
                ('from model.modules.mlp import MLP', 'from .modules.mlp import MLP'),
                ('from model.modules.tcn import MultiScaleTCN', 'from .modules.tcn import MultiScaleTCN'),
            ]
        },
        {
            'file': REPO_ROOT / 'lib' / 'motionagformer' / 'model' / 'modules' / 'metaformer.py',
            'replacements': [
                ('from model.modules.attention import Attention', 'from .attention import Attention'),
                ('from model.modules.ctrgc import CTRGCBlock', 'from .ctrgc import CTRGCBlock'),
                ('from model.modules.graph import GCN', 'from .graph import GCN'),
                ('from model.modules.mlp import MLP', 'from .mlp import MLP'),
                ('from model.modules.tcn import MultiScaleTCN', 'from .tcn import MultiScaleTCN'),
            ]
        }
    ]
    
    fixed_count = 0
    for fix_info in fixes:
        filepath = fix_info['file']
        print(f"\n📝 Processing: {filepath.name}")
        
        if not filepath.exists():
            print(f"   ❌ File not found: {filepath}")
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            modified = False
            for old_import, new_import in fix_info['replacements']:
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    modified = True
                    print(f"   ✓ Replaced: {old_import[:40]}...")
            
            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_count += 1
                print(f"   ✅ Saved changes")
            else:
                print(f"   ⏭️  No changes needed")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)
    if fixed_count > 0:
        print(f"✅ Fixed {fixed_count} file(s)")
        print("\nNow run: python test_3d_lifting.py")
    else:
        print("ℹ️  All files already fixed")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
