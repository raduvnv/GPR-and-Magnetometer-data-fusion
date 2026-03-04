"""
SENSOR FUSION: GPR + Magnetometer
==================================
Dit script combineert detecties van twee sensoren:
- GPR (Ground Penetrating Radar): geeft (trace, sample) coördinaten
- Magnetometer: geeft (x, y) coördinaten (grid-based)

Aanpak: AND-gate (alleen detecties waar beide sensoren akkoord gaan)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Detection:
    """Een enkele detectie van één sensor"""
    x: float
    y: float
    amplitude: float
    source: str  # 'gpr' of 'magnetometer'
    confidence: float = 1.0


@dataclass
class FusedDetection:
    """Een bevestigde detectie (beide sensoren stemmen akkoord)"""
    x: float
    y: float
    gpr_strength: float
    mag_strength: float
    fusion_score: float  # Hoe sterk stemmen ze overeen? (0-1)
    distance_apart: float  # Afstand tussen de twee detecties


# ==============================================================================
# STAP 1: COORDINAAT-TRANSFORMATIE (GPR → wereldcoördinaten)
# ==============================================================================

def gpr_naar_wereld(
    center_x_trace: float,
    center_y_sample: float,
    trace_spacing: float = 0.1,  # meter per trace
    sample_spacing: float = 0.05,  # meter per sample
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> Tuple[float, float]:
    """
    Converteer GPR indices naar wereldcoördinaten.
    
    Args:
        center_x_trace: trace index (diepte/oostwaarts richting)
        center_y_sample: sample index (zijwaarts/noordwaarts richting)
        trace_spacing: fysieke afstand per trace-index
        sample_spacing: fysieke afstand per sample-index
        origin_x, origin_y: waar het GPR-scan begon
    
    Returns:
        (wereld_x, wereld_y)
    """
    wereld_x = origin_x + center_x_trace * trace_spacing
    wereld_y = origin_y + center_y_sample * sample_spacing
    return wereld_x, wereld_y


def wereld_naar_gpr(
    wereld_x: float,
    wereld_y: float,
    trace_spacing: float = 0.1,
    sample_spacing: float = 0.05,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> Tuple[float, float]:
    """Omgekeerde transformatie"""
    center_x_trace = (wereld_x - origin_x) / trace_spacing
    center_y_sample = (wereld_y - origin_y) / sample_spacing
    return center_x_trace, center_y_sample


# ==============================================================================
# STAP 2: EXTRACTIE VAN DETECTIES UIT BEIDE SCRIPTS
# ==============================================================================

def extract_gpr_detections(
    reflections,
    trace_spacing: float = 0.1,
    sample_spacing: float = 0.05,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> List[Detection]:
    """
    Pak de gevalideerde reflecties uit het GPR-script.
    
    Input: reflections (list van Candidate objecten uit detect_reflections)
    Output: list van Detection objecten in wereldcoördinaten
    """
    detections = []
    for refl in reflections:
        # refl.apex_x = trace index van het midden
        # refl.apex_y = sample index van het midden
        # refl.max_amplitude = sterkte van de reflectie
        
        wereld_x, wereld_y = gpr_naar_wereld(
            refl.apex_x,
            refl.apex_y,
            trace_spacing=trace_spacing,
            sample_spacing=sample_spacing,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        
        detections.append(Detection(
            x=wereld_x,
            y=wereld_y,
            amplitude=refl.max_amplitude,
            source='gpr',
            confidence=1.0 - (refl.fit_error / 1000.0)  # Gebruik fit_error als vertrouwen
        ))
    
    return detections


def extract_mag_detections(
    mag_locations: List[Dict],
) -> List[Detection]:
    """
    Pak de detecties uit het magnetometer-script.
    
    Input: lijst van dicts met 'x', 'y' keys
    Output: list van Detection objecten
    """
    detections = []
    for loc in mag_locations:
        detections.append(Detection(
            x=loc['x'],
            y=loc['y'],
            amplitude=loc.get('strength', 1.0),
            source='magnetometer',
            confidence=loc.get('confidence', 0.8)
        ))
    return detections


# ==============================================================================
# STAP 3: MATCHING ALGORITME (Welke detecties horen bij elkaar?)
# ==============================================================================

def match_detections(
    gpr_detections: List[Detection],
    mag_detections: List[Detection],
    max_distance: float = 0.5,  # meter (aanpassingsparameter!)
    use_distance_weighting: bool = True,
) -> List[FusedDetection]:
    """
    AND-gate sensorfusie: match GPR en magnetometer detecties.
    
    Strategie:
    1. Voor elke GPR detectie, zoek de dichtsbijzijnde magnetometer detectie
    2. Als die dichterbij is dan max_distance, dan is het een match
    3. Bereken een fusion_score op basis van afstand en amplitudes
    
    Args:
        gpr_detections: lijst van GPR Detection objecten
        mag_detections: lijst van Magnetometer Detection objecten
        max_distance: maximale afstand (meters) voor een match
        use_distance_weighting: of we afstand meenemen in score
    
    Returns:
        lijst van FusedDetection objecten
    """
    fused = []
    used_mag_indices = set()
    
    for gpr_det in gpr_detections:
        # Vind dichtsbijzijnde magnetometer detectie
        best_mag_idx = None
        best_distance = float('inf')
        
        for mag_idx, mag_det in enumerate(mag_detections):
            if mag_idx in used_mag_indices:
                continue
            
            distance = np.sqrt(
                (gpr_det.x - mag_det.x) ** 2 +
                (gpr_det.y - mag_det.y) ** 2
            )
            
            if distance < best_distance:
                best_distance = distance
                best_mag_idx = mag_idx
        
        # Check of match dicht genoeg is
        if best_mag_idx is not None and best_distance <= max_distance:
            mag_det = mag_detections[best_mag_idx]
            used_mag_indices.add(best_mag_idx)
            
            # Fusion score: hoe goed stemmen ze overeen?
            # 1. Afstand penalty
            distance_score = 1.0 - (best_distance / max_distance)
            
            # 2. Amplitude consensus (beide moeten relatief sterk zijn)
            # Normaliseer amplitudes (0-1 schaal binnen hun sensor)
            gpr_norm = gpr_det.amplitude / max([d.amplitude for d in gpr_detections] or [1.0])
            mag_norm = mag_det.amplitude / max([d.amplitude for d in mag_detections] or [1.0])
            amplitude_score = (gpr_norm + mag_norm) / 2.0
            
            # 3. Gecombineerde score
            if use_distance_weighting:
                fusion_score = 0.6 * distance_score + 0.4 * amplitude_score
            else:
                fusion_score = amplitude_score
            
            # Centrum van beide detecties
            fused_x = (gpr_det.x + mag_det.x) / 2.0
            fused_y = (gpr_det.y + mag_det.y) / 2.0
            
            fused.append(FusedDetection(
                x=fused_x,
                y=fused_y,
                gpr_strength=gpr_det.amplitude,
                mag_strength=mag_det.amplitude,
                fusion_score=fusion_score,
                distance_apart=best_distance,
            ))
    
    return fused


# ==============================================================================
# STAP 4: RAPPORTAGE EN VISUALISATIE
# ==============================================================================

def print_fusion_report(
    gpr_detections: List[Detection],
    mag_detections: List[Detection],
    fused_detections: List[FusedDetection],
):
    """Pretty-print van de fusie resultaten"""
    print("\n" + "=" * 70)
    print("SENSOR FUSION RAPPORT (AND-GATE)")
    print("=" * 70)
    
    print(f"\nInput:")
    print(f"  • GPR detecties:         {len(gpr_detections)}")
    print(f"  • Magnetometer detecties: {len(mag_detections)}")
    
    print(f"\nOutput:")
    print(f"  • Geverifieerde objecten: {len(fused_detections)}")
    
    if fused_detections:
        print(f"\nDetails (gesorteerd op fusie-score):")
        print("-" * 70)
        
        sorted_fused = sorted(fused_detections, key=lambda f: f.fusion_score, reverse=True)
        
        for idx, obj in enumerate(sorted_fused, 1):
            print(f"\nObject #{idx}")
            print(f"  Locatie:          X={obj.x:+.2f}m, Y={obj.y:+.2f}m")
            print(f"  Fusie-score:      {obj.fusion_score:.1%} (betrouwbaarheid)")
            print(f"  GPR sterkte:      {obj.gpr_strength:.3g}")
            print(f"  Mag sterkte:      {obj.mag_strength:.3g}")
            print(f"  Afstand apart:    {obj.distance_apart:.3f}m")
    
    print("\n" + "=" * 70)


def visualize_fusion(
    gpr_detections: List[Detection],
    mag_detections: List[Detection],
    fused_detections: List[FusedDetection],
    grid_bounds: Tuple[float, float, float, float] = (-5, 5, -5, 5),
):
    """
    Visualiseer de sensorfusie-resultaten.
    
    Args:
        grid_bounds: (x_min, x_max, y_min, y_max) voor de plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x_min, x_max, y_min, y_max = grid_bounds
    
    # --- PANEL 1: Alle detecties + matches ---
    ax = axes[0]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot GPR detecties
    if gpr_detections:
        gpr_x = [d.x for d in gpr_detections]
        gpr_y = [d.y for d in gpr_detections]
        ax.scatter(gpr_x, gpr_y, c='red', s=100, marker='s', 
                  label='GPR', alpha=0.7, edgecolors='darkred', linewidth=2)
    
    # Plot magnetometer detecties
    if mag_detections:
        mag_x = [d.x for d in mag_detections]
        mag_y = [d.y for d in mag_detections]
        ax.scatter(mag_x, mag_y, c='blue', s=100, marker='^', 
                  label='Magnetometer', alpha=0.7, edgecolors='darkblue', linewidth=2)
    
    # Plot geverifieerde objecten + verbindingslijnen
    for obj in fused_detections:
        ax.scatter(obj.x, obj.y, c='lime', s=300, marker='*', 
                  label='Verified' if obj == fused_detections[0] else '',
                  edgecolors='darkgreen', linewidth=2, zorder=5)
    
    ax.set_xlabel('X (meter)', fontsize=11)
    ax.set_ylabel('Y (meter)', fontsize=11)
    ax.set_title('Sensor Fusion: GPR + Magnetometer Detecties', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    
    # --- PANEL 2: Fusie-scores ---
    ax = axes[1]
    
    if fused_detections:
        sorted_fused = sorted(fused_detections, key=lambda f: f.fusion_score, reverse=True)
        object_ids = [f"Obj {i+1}" for i in range(len(sorted_fused))]
        scores = [f.fusion_score for f in sorted_fused]
        distances = [f.distance_apart for f in sorted_fused]
        
        # Dual-axis bar chart
        ax2 = ax.twinx()
        
        bars = ax.bar(object_ids, scores, color='green', alpha=0.7, label='Fusie-score')
        line = ax2.plot(object_ids, distances, 'ro-', linewidth=2, markersize=8, label='Afstand apart')
        
        ax.set_ylabel('Fusie-score (0-1)', fontsize=11, color='green')
        ax2.set_ylabel('Afstand (meter)', fontsize=11, color='red')
        ax.set_ylim(0, 1.0)
        ax.set_title('Kwaliteit van Matchings', fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='green')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        # Geen fusies: toon een duidelijke melding in het rechter paneel
        ax.axis('off')
        ax.set_title('Kwaliteit van Matchings', fontsize=12, fontweight='bold')
        ax.text(0.5, 0.5, 'Geen geverifieerde matches\n(geen overlapping tussen sensoren)',
                ha='center', va='center', fontsize=12, color='gray')
    
    plt.tight_layout()
    plt.savefig('sensor_fusion_result.png', dpi=150)
    print("\nVisualisatie opgeslagen als 'sensor_fusion_result.png'")
    plt.show()


# ==============================================================================
# VOORBEELD: INTEGRATIE MET JE BESTAANDE SCRIPTS
# ==============================================================================

def example_integration():
    """
    Voorbeeld hoe je dit integreert in je bestaande code.
    """
    
    # === UIT JE GPR-SCRIPT ===
    # (dit roep je vanuit je gpr_visualization.py)
    # reflections, candidates, data_gained, binary_mask, ground_cutoff, col_profile_smooth = \
    #     detect_reflections(SETTINGS["filename"], SETTINGS)
    
    # === UIT JE MAGNETOMETER-SCRIPT ===
    # (dit roep je vanuit je magnetic_analysis.py)
    # oude_detecties = analyze_mag_data_oud(rx_locs, dpred)
    # exacte_x, exacte_y = vind_object_locatie(rx_locs, dpred)
    
    # === SENSOR FUSION ===
    # Voorbeeld data (vervang met echte data van bovenstaande)
    example_gpr = [
        {'apex_x': 45.0, 'apex_y': 120.0, 'max_amplitude': 0.85, 'fit_error': 50.0},
        {'apex_x': 78.0, 'apex_y': 200.0, 'max_amplitude': 0.72, 'fit_error': 120.0},
    ]
    
    example_mag = [
        {'x': 4.5, 'y': 6.0, 'strength': 250.0},
        {'x': 7.8, 'y': 10.0, 'strength': 180.0},
    ]
    
    # Converteer naar Detection objecten
    gpr_dets = []
    for gpr in example_gpr:
        x, y = gpr_naar_wereld(gpr['apex_x'], gpr['apex_y'])
        gpr_dets.append(Detection(x, y, gpr['max_amplitude'], 'gpr'))
    
    mag_dets = extract_mag_detections(example_mag)
    
    # Voer fusie uit
    fused = match_detections(gpr_dets, mag_dets, max_distance=1.0)
    
    # Rapporteer
    print_fusion_report(gpr_dets, mag_dets, fused)
    visualize_fusion(gpr_dets, mag_dets, fused)


if __name__ == "__main__":
    example_integration()
