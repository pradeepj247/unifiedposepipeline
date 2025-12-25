"""
Explore BoxMOT v16.0.4 API to understand:
1. What's available in the module
2. How to access tracker classes
3. What ReID models are supported
4. Proper initialization patterns
"""

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

print("\n" + "="*70)
print("Exploring boxmot module...")
print("="*70)

try:
    import boxmot
    print(f"✅ boxmot imported successfully!")
    print(f"   Location: {boxmot.__file__}")
    print(f"   Version: {boxmot.__version__}")
    
    # List all public attributes
    print(f"\n📦 Available attributes in boxmot:")
    all_attrs = [name for name in dir(boxmot) if not name.startswith('_')]
    for attr in sorted(all_attrs):
        obj = getattr(boxmot, attr)
        obj_type = type(obj).__name__
        print(f"   - {attr:25s} ({obj_type})")
    
    # Try to find tracker classes
    print(f"\n🔍 Looking for tracker classes:")
    tracker_classes = []
    for name in all_attrs:
        obj = getattr(boxmot, name)
        if isinstance(obj, type):  # It's a class
            print(f"   ✅ Found class: {name}")
            tracker_classes.append(name)
    
    # Check if trackers submodule exists
    print(f"\n📁 Checking for trackers submodule:")
    try:
        from boxmot import trackers
        print(f"   ✅ trackers submodule found!")
        print(f"   Location: {trackers.__file__}")
        
        tracker_names = [name for name in dir(trackers) if not name.startswith('_')]
        print(f"\n   Available in trackers submodule:")
        for name in sorted(tracker_names):
            print(f"      - {name}")
    except (ImportError, AttributeError) as e:
        print(f"   ⚠️  No trackers submodule: {e}")
    
    # Check for appearance (ReID) models
    print(f"\n🎭 Checking for appearance/ReID support:")
    try:
        from boxmot import appearance
        print(f"   ✅ appearance submodule found!")
        print(f"   Location: {appearance.__file__}")
        
        reid_attrs = [name for name in dir(appearance) if not name.startswith('_')]
        print(f"\n   Available in appearance submodule:")
        for name in sorted(reid_attrs):
            print(f"      - {name}")
    except (ImportError, AttributeError) as e:
        print(f"   ⚠️  No appearance submodule: {e}")
    
    # Try to instantiate available trackers
    print(f"\n🧪 Testing tracker instantiation:")
    
    test_trackers = [
        ('BotSort', 'botsort'),
        ('ByteTrack', 'bytetrack'),
        ('BoostTrack', 'boosttrack'),
        ('StrongSORT', 'strongsort'),
        ('OCSORT', 'ocsort'),
        ('DeepOCSORT', 'deepocsort'),
        ('HybridSORT', 'hybridsort'),
    ]
    
    for class_name, tracker_name in test_trackers:
        try:
            tracker_class = getattr(boxmot, class_name)
            print(f"   ✅ {class_name:15s} - Class accessible")
            
            # Try to get signature
            import inspect
            sig = inspect.signature(tracker_class.__init__)
            params = [p for p in sig.parameters.keys() if p != 'self']
            print(f"      Init params: {', '.join(params[:5])}")  # Show first 5 params
            
        except AttributeError:
            print(f"   ⚠️  {class_name:15s} - Not found in boxmot")
        except Exception as e:
            print(f"   ❌ {class_name:15s} - Error: {e}")
    
    # Check for ReID weights
    print(f"\n💾 Checking ReID model support:")
    try:
        # Try to find where ReID models are stored/referenced
        if hasattr(boxmot, 'get_model_path'):
            print(f"   ✅ get_model_path function found")
        
        # Check for common ReID model names
        common_reid_models = [
            'osnet_x0_25_msmt17',
            'osnet_x1_0_msmt17',
            'lmbn_n_duke',
            'mobilenetv2_x1_4_msmt17',
            'clip_market1501',
        ]
        
        print(f"\n   Common ReID models mentioned in README:")
        for model in common_reid_models:
            print(f"      - {model}")
        
    except Exception as e:
        print(f"   ⚠️  Error checking ReID: {e}")
    
    print("\n" + "="*70)
    print("✅ Exploration complete!")
    print("="*70)
    
except ImportError as e:
    print(f"❌ Failed to import boxmot: {e}")
    sys.exit(1)
