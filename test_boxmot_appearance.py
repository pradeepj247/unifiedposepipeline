#!/usr/bin/env python3
"""Quick test to see if boxmot.appearance is available"""

try:
    from boxmot.appearance.reid_auto_backend import ReidAutoBackend
    print("✅ boxmot.appearance.reid_auto_backend is available!")
    print(f"ReidAutoBackend: {ReidAutoBackend}")
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")

# Also try listing what's in boxmot
try:
    import boxmot
    print(f"\n📦 boxmot location: {boxmot.__file__}")
    print(f"boxmot attributes: {dir(boxmot)}")
except Exception as e:
    print(f"Error: {e}")
