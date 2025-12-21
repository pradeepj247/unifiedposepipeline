# Files to Clean Up

## ✅ Cleanup Complete - Already Removed

The following files were identified as redundant but were already cleaned up before commit:

### 1. `verify_environment.py`
- **Reason**: Redundant - superseded by comprehensive `verify.py`
- **Status**: ✅ Already removed (not in repo)

### 2. `setup_environment.py`
- **Reason**: Redundant - all functionality now in `setup_unified.py`
- **Details**: Old setup script (505 lines), replaced by newer unified version
- **Status**: ✅ Already removed (not in repo)

### 3. `setup_demo_data.py`
- **Reason**: Redundant - demo data setup is now Step 8 in `setup_unified.py`
- **Details**: Standalone script for demo data, now integrated
- **Status**: ✅ Already removed (not in repo)

### 4. `download_models.py`
- **Reason**: Redundant - model downloads are now Step 7 in `setup_unified.py`
- **Details**: Has outdated OneDrive links for ViTPose (we now use GitHub releases)
- **Status**: ✅ Already removed (not in repo)

### 5. `udp.py`
- **Reason**: Redundant - replaced by `udp_image.py` and `udp_video.py`
- **Details**: Old general-purpose demo script (451 lines), now split into focused scripts
- **Status**: ✅ Already removed (not in repo)

---

## 🤔 Files to REVIEW

### 1. `setup.py`
- **Type**: Standard Python package setup file (setuptools)
- **Purpose**: For installing as a pip package (`pip install -e .`)
- **Decision**: KEEP if you want pip install capability, DELETE if not needed
- **Recommendation**: **KEEP** - useful for development mode installation
- **Status**: ⚠️ Review needed

### 2. `run.py`
- **Type**: Helper launcher script
- **Purpose**: Shortcuts like `python run.py setup`, `python run.py demo image`
- **Decision**: KEEP if you want convenience commands, DELETE if direct calls are fine
- **Recommendation**: **KEEP** - provides nice user-friendly shortcuts
- **Status**: ⚠️ Review needed

---

## ✅ Files to KEEP (Core functionality)

### Main Scripts
- `setup_unified.py` - Main setup script (9 stages, complete)
- `verify.py` - Comprehensive verification
- `udp_image.py` - Quick image demo
- `udp_video.py` - Comprehensive video demo

### Configuration
- `configs/*.yaml` - All YAML config files
- `.gitignore` - Git exclusions

### Documentation
- `README*.md` - All documentation files
- `*.md` - Guide files

### Library Code
- `lib/vitpose/` - ViTPose library
- `lib/rtmlib/` - RTMLib library

---

## 📊 Summary
- **To Delete**: 5 files (verified redundant)
- **To Review**: 2 files (user preference)
- **To Keep**: Core files + configs + docs + libs

---

*Last updated: After reviewing all setup/verify/demo scripts*
