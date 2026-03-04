"""
GEÏNTEGREERD WORKFLOW
=====================
Dit script runt je GPR-script, magnetometer-script EN sensor fusion
allemaal achter elkaar in één go.

Voorbereiding:
1. Plaats je gpr_visualization.py, magnetic_analysis.py en sensor_fusion.py
   in dezelfde folder
2. Zorg dat je data-bestanden aanwezig zijn
3. Run dit script
"""

import numpy as np
import sys
from pathlib import Path

# Importeer je bestaande modules
from sensor_fusion import (
    Detection,
    extract_gpr_detections,
    extract_mag_detections,
    match_detections,
    print_fusion_report,
    visualize_fusion,
    gpr_naar_wereld,
)

# Je eigen scripts (dit volgt uit je upload)
# from gpr_visualization import detect_reflections, visualize_results
# from magnetic_analysis import laad_mag_data, vind_object_locatie, analyze_mag_data_oud


def run_integrated_workflow(
    gpr_filename: str,
    mag_filename: str,
    gpr_settings: dict,
    fusion_max_distance: float = 0.5,
    trace_spacing: float = 0.1,
    sample_spacing: float = 0.05,
):
    """
    Voer het volledige workflow uit: GPR → Magnetometer → Fusion
    
    Args:
        gpr_filename: Pad naar GPR data (.out of .mat)
        mag_filename: Pad naar magnetometer data (.npz)
        gpr_settings: Settings dict voor GPR detectie
        fusion_max_distance: max afstand (meters) voor een match
        trace_spacing: fysieke afstand per trace
        sample_spacing: fysieke afstand per sample
    """
    
    print("\n" + "=" * 70)
    print("GEÏNTEGREERD WORKFLOW STARTER")
    print("=" * 70)
    
    # =========================================================================
    # STAP 1: GPR DETECTIE
    # =========================================================================
    print("\n[STAP 1] GPR ANALYSE STARTEN...")
    print("-" * 70)
    
    try:
        # Import hier zodat het optional is
        from main import detect_reflections, visualize_results
        
        reflections, candidates, data_gained, binary_mask, ground_cutoff, col_profile_smooth = (
            detect_reflections(gpr_filename, gpr_settings)
        )
        
        if data_gained is None:
            print("❌ GPR-script kon geen data laden")
            return None
        
        # Visualiseer GPR alleen
        print("\n[GPR] Genereer visualisatie...")
        visualize_results(
            reflections,
            candidates,
            data_gained,
            binary_mask,
            ground_cutoff,
        )
        
        # Converteer naar Detection objecten
        gpr_detections = extract_gpr_detections(
            reflections,
            trace_spacing=trace_spacing,
            sample_spacing=sample_spacing,
        )
        print(f"✓ GPR: {len(gpr_detections)} detecties geconverteerd")
        
    except ImportError:
        print("⚠️  gpr_visualization.py niet gevonden. Slaat GPR-stap over.")
        gpr_detections = []
    except Exception as e:
        print(f"❌ Fout in GPR-script: {e}")
        return None
    
    # =========================================================================
    # STAP 2: MAGNETOMETER DETECTIE
    # =========================================================================
    print("\n[STAP 2] MAGNETOMETER ANALYSE STARTEN...")
    print("-" * 70)
    
    try:
        from mag_upload_AS import laad_mag_data, vind_object_locatie, analyze_mag_data_oud
        
        rx_locs, dpred = laad_mag_data(mag_filename)

        # Als het bestand niet bestaat, probeer te genereren (optioneel)
        if rx_locs is None:
            try:
                print("   Bestand niet gevonden. Probeer test-bestand te genereren via generator...")
                from mag_upload_AS import genereer_test_bestand
                genereer_test_bestand(mag_filename)
                rx_locs, dpred = laad_mag_data(mag_filename)
            except Exception as gen_e:
                print(f"   Geen test-bestand kunnen genereren: {gen_e}")

        if rx_locs is None:
            print("❌ Magnetometer-script kon geen data laden")
            mag_detections = []
        else:
            # Nieuwe methode (Analytisch Signaal)
            mag_x, mag_y = vind_object_locatie(rx_locs, dpred)
            mag_new = [{'x': mag_x, 'y': mag_y, 'strength': 1.0}]

            # Oude methode (ter vergelijking)
            mag_old = analyze_mag_data_oud(rx_locs, dpred)

            # Combineer (oude methode als fallback)
            mag_all = mag_new + mag_old

            # Converteer naar Detection objecten
            mag_detections = extract_mag_detections(mag_all)
            print(f"✓ Magnetometer: {len(mag_detections)} detecties geconverteerd")
        
    except ImportError as e:
        print(f"⚠️  Magnetometer module import failed: {e}. Slaat magnetometer-stap over.")
        mag_detections = []
    except Exception as e:
        print(f"❌ Fout in magnetometer-script: {e}")
        mag_detections = []
    
    # =========================================================================
    # STAP 3: SENSOR FUSION (AND-GATE)
    # =========================================================================
    print("\n[STAP 3] SENSOR FUSION STARTEN...")
    print("-" * 70)
    
    if not gpr_detections or not mag_detections:
        print("⚠️  Een of beide sensoren hebben geen detecties.")
        print("    Kan geen fusion uitvoeren.")
        if gpr_detections:
            print(f"    (GPR: {len(gpr_detections)}, Mag: {len(mag_detections)})")
        return None
    
    fused_detections = match_detections(
        gpr_detections,
        mag_detections,
        max_distance=fusion_max_distance,
        use_distance_weighting=True,
    )
    
    print(f"✓ Fusion voltooid: {len(fused_detections)} geverifieerde objecten")
    
    # =========================================================================
    # STAP 4: RAPPORTAGE
    # =========================================================================
    print("\n[STAP 4] RAPPORTAGE EN VISUALISATIE...")
    print("-" * 70)
    
    print_fusion_report(gpr_detections, mag_detections, fused_detections)
    
    # Bepaal grid bounds van alle detecties
    if gpr_detections or mag_detections:
        all_x = [d.x for d in gpr_detections + mag_detections]
        all_y = [d.y for d in gpr_detections + mag_detections]
        x_min, x_max = min(all_x) - 1, max(all_x) + 1
        y_min, y_max = min(all_y) - 1, max(all_y) + 1
        grid_bounds = (x_min, x_max, y_min, y_max)
    else:
        grid_bounds = (-5, 5, -5, 5)
    
    visualize_fusion(gpr_detections, mag_detections, fused_detections, grid_bounds)
    
    # =========================================================================
    # STAP 5: EXPORTEER RESULTATEN
    # =========================================================================
    print("\n[STAP 5] EXPORTEER RESULTATEN...")
    print("-" * 70)
    
    export_results(fused_detections)
    
    print("\n" + "=" * 70)
    print("✓ WORKFLOW VOLTOOID")
    print("=" * 70)
    
    return fused_detections


def export_results(fused_detections, output_file: str = "fusion_results.csv"):
    """Exporteer fusion-resultaten naar CSV"""
    
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Object #',
            'X (m)',
            'Y (m)',
            'GPR Sterkte',
            'Mag Sterkte',
            'Fusie Score',
            'Afstand Apart (m)',
        ])
        
        for idx, obj in enumerate(sorted(fused_detections, 
                                         key=lambda x: x.fusion_score, 
                                         reverse=True), 1):
            writer.writerow([
                idx,
                f"{obj.x:.3f}",
                f"{obj.y:.3f}",
                f"{obj.gpr_strength:.3g}",
                f"{obj.mag_strength:.3g}",
                f"{obj.fusion_score:.1%}",
                f"{obj.distance_apart:.3f}",
            ])
    
    print(f"✓ Resultaten geëxporteerd naar '{output_file}'")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    
    # CONFIGURATIE
    # ============
    
    GPR_FILE = "gprdata/Chelton_sand_targets.mat"  # Je GPR bestand
    MAG_FILE = "gprdata/mag_testdata.npz"                   # Je magnetometer bestand
    
    # GPR settings (dezelfde als in je gpr_visualization.py)
    GPR_SETTINGS = {
        "filename": GPR_FILE,
        "sigma": 5,
        "min_trace": 50,
        "max_trace": 400,
        "intensity_T1": None,
        "min_trace_width": 5,
        "max_fit_error": 1000,
        "ground_search_fraction": 0.25,
        "ground_margin": 5,
    }
    
    # Fusion parameters
    FUSION_MAX_DISTANCE = 0.5  # meter - aanpassen naar je noden!
    
    # Fysieke scaling
    TRACE_SPACING = 0.1        # meter per trace-index
    SAMPLE_SPACING = 0.05      # meter per sample-index
    
    # RUN
    # ===
    
    results = run_integrated_workflow(
        gpr_filename=GPR_FILE,
        mag_filename=MAG_FILE,
        gpr_settings=GPR_SETTINGS,
        fusion_max_distance=FUSION_MAX_DISTANCE,
        trace_spacing=TRACE_SPACING,
        sample_spacing=SAMPLE_SPACING,
    )
    
    if results is not None:
        print(f"\n✓ Gevonden objecten: {len(results)}")
        for i, obj in enumerate(sorted(results, key=lambda x: x.fusion_score, reverse=True), 1):
            print(f"  {i}. ({obj.x:.2f}, {obj.y:.2f}) - Score: {obj.fusion_score:.1%}")
