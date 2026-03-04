#!/usr/bin/env python3
"""
SIMPEL TEST SCRIPT
==================
Zet dit in dezelfde folder als je andere scripts en run:

    python test_setup.py

Dit checkt of alles goed is ingesteld voordat je integrated_workflow.py runt.
"""

import sys
import os
from pathlib import Path

print("\n" + "="*70)
print("SENSOR FUSION SETUP CHECK")
print("="*70)

# Stap 1: Check Python versie
print("\n[1/5] Python versie...")
print(f"  Python {sys.version.split()[0]}")
if sys.version_info < (3, 7):
    print("  ❌ Python 3.7+ vereist!")
    sys.exit(1)
print("  ✓ OK")

# Stap 2: Check bestandsstructuur
print("\n[2/5] Controleer bestandsstructuur...")

required_files = {
    'main.py': 'GPR script',
    'mag_upload_AS.py': 'Magnetometer script',
    'sensor_fusion.py': 'Sensor fusion library',
    'integrated_workflow.py': 'Integrated workflow',
    'helpers.py': 'Helpers (voor main.py)',
    'models.py': 'Models (voor main.py)',
}

missing = []
for filename, description in required_files.items():
    if os.path.exists(filename):
        print(f"  ✓ {filename:30s} ({description})")
    else:
        print(f"  ❌ {filename:30s} NIET GEVONDEN")
        missing.append(filename)

if missing:
    print(f"\n  ⚠️  Ontbrekende bestanden: {', '.join(missing)}")
    print("     Download ze en zet ze in dezelfde folder")
    sys.exit(1)

# Stap 3: Check data bestanden
print("\n[3/5] Controleer data-bestanden...")

data_files = {
    'gprdata': 'GPR data folder',
    'mag_testdata.npz': 'Magnetometer testdata',
}

data_ok = True
for filename, description in data_files.items():
    if os.path.exists(filename):
        print(f"  ✓ {filename:30s} ({description})")
    else:
        print(f"  ⚠️  {filename:30s} NIET GEVONDEN (optioneel)")
        data_ok = False

if not data_ok:
    print("\n  Opmerking: data-bestanden zijn optioneel voor nu")

# Stap 4: Check imports
print("\n[4/5] Controleer Python modules...")

required_modules = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'matplotlib': 'Matplotlib',
    'h5py': 'HDF5',
}

missing_modules = []
for module_name, description in required_modules.items():
    try:
        __import__(module_name)
        print(f"  ✓ {description:20s} ({module_name})")
    except ImportError:
        print(f"  ❌ {description:20s} NIET GEÏNSTALLEERD")
        missing_modules.append(module_name)

if missing_modules:
    print(f"\n  Install met: pip install {' '.join(missing_modules)}")
    sys.exit(1)

# Stap 5: Try imports van jouw scripts
print("\n[5/5] Controleer jouw scripts...")

try:
    from main import detect_reflections, visualize_results
    print(f"  ✓ main.py imports OK")
except Exception as e:
    print(f"  ❌ main.py import failed: {e}")
    sys.exit(1)

try:
    from mag_upload_AS import laad_mag_data, vind_object_locatie
    print(f"  ✓ mag_upload_AS.py imports OK")
except Exception as e:
    print(f"  ❌ mag_upload_AS.py import failed: {e}")
    sys.exit(1)

try:
    from sensor_fusion import match_detections, extract_gpr_detections
    print(f"  ✓ sensor_fusion.py imports OK")
except Exception as e:
    print(f"  ❌ sensor_fusion.py import failed: {e}")
    sys.exit(1)

try:
    import integrated_workflow
    print(f"  ✓ integrated_workflow.py imports OK")
except Exception as e:
    print(f"  ❌ integrated_workflow.py import failed: {e}")
    sys.exit(1)

# Alles OK!
print("\n" + "="*70)
print("✓ ALLE CHECKS GESLAAGD!")
print("="*70)

print("""
Je bent klaar om te runnen!

Volgende stap:

    python integrated_workflow.py

Dit zal:
  1. Je GPR data analyseren
  2. Je magnetometer data analyseren
  3. Ze samenvoegen (sensor fusion)
  4. 3 output-bestanden genereren:
     - visual_gpr_data.png
     - sensor_fusion_result.png
     - fusion_results.csv

Vragen? Check QUICK_START.txt of SETUP_INSTRUCTIES.txt
""")
