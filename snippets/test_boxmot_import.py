#!/usr/bin/env python3
"""
Quick test to verify boxmot installation and imports
"""

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}")

print("\n" + "="*70)
print("Testing boxmot import...")
print("="*70)

try:
    import boxmot
    print(f"✅ boxmot imported successfully!")
    print(f"   Location: {boxmot.__file__}")
    print(f"   Version: {boxmot.__version__ if hasattr(boxmot, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"❌ Failed to import boxmot: {e}")
    print("\nTry installing: pip install boxmot")
    sys.exit(1)

print("\nTesting individual tracker imports...")
available_trackers = []
failed_trackers = []

trackers_to_test = [
    ('BotSort', 'botsort'),
    ('ByteTrack', 'bytetrack'),
    ('StrongSort', 'strongsort'),      # Fixed: StrongSort, not StrongSORT!
    ('OcSort', 'ocsort'),              # Fixed: OcSort, not OCSORT!
    ('BoostTrack', 'boosttrack'),
    ('DeepOcSort', 'deepocsort'),      # Fixed: DeepOcSort, not DeepOCSORT!
    ('HybridSort', 'hybridsort'),      # Fixed: HybridSort, not HybridSORT!
]

for class_name, tracker_name in trackers_to_test:
    try:
        tracker_class = getattr(boxmot, class_name)
        available_trackers.append(tracker_name)
        print(f"   ✅ {class_name} available")
    except AttributeError:
        failed_trackers.append(tracker_name)
        print(f"   ⚠️  {class_name} not available")

if available_trackers:
    print(f"\n✅ Available trackers ({len(available_trackers)}): {', '.join(available_trackers)}")
else:
    print(f"\n❌ No trackers available!")
    sys.exit(1)

if failed_trackers:
    print(f"⚠️  Unavailable trackers ({len(failed_trackers)}): {', '.join(failed_trackers)}")

print("\n" + "="*70)
print("✅ BoxMOT is ready to use!")
print(f"   Use any of these trackers: {', '.join(available_trackers)}")
print("="*70)
