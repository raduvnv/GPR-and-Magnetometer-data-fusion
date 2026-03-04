import numpy as np
import matplotlib.pyplot as plt
import os

# --- SimPEG imports (alleen nodig als je synthetische data wilt GENEREREN) ---
# De SimPEG-gerelateerde imports worden pas gemaakt binnen de generatorfunctie
# zodat het script ook kan worden geïmporteerd zonder dat SimPEG aanwezig is.

# --- Imports voor detectie ---
from scipy.interpolate import griddata
from scipy.ndimage import label, center_of_mass

# ==============================================================================
# STAP 1: DATA GENERATOR (OPTIONEEL)
# Draai dit alleen om een test-bestand te maken als je nog geen echte data hebt.
# ==============================================================================
def genereer_test_bestand(bestandsnaam="mag_testdata.npz"):
    print(f"--- GENERATOR: Bezig met aanmaken van {bestandsnaam} ---")
    # 1. Setup Grid & Model
    try:
        from discretize import TensorMesh
        from simpeg import maps, utils
        from simpeg.potential_fields import magnetics
    except Exception:
        print("   [FOUT] SimPEG niet gevonden. Generator vereist SimPEG-pakketten.")
        raise

    h = [0.1, 0.1, 0.1]
    n = [100, 100, 30]
    mesh = TensorMesh([np.ones(n[0]) * h[0], np.ones(n[1]) * h[1], np.ones(n[2]) * h[2]], origin='CCC')
    
    # Start with 0 susceptibility everywhere (air/background)
    model_array = np.zeros(mesh.nC)

    # Create a sphere anomaly (gebaseerd op TM-62M volume-equivalent bol)
    sphere_radius = 0.135
    sphere_depth = -0.5

    # object 1 (in het midden)
    sphere_ind1 = utils.model_builder.get_indices_sphere([0, 0, sphere_depth], sphere_radius, mesh.gridCC)
    model_array[sphere_ind1] = 100  #susceptibility

    # object 2 (2 meter naar het noordoosten)
    sphere_ind2 = utils.model_builder.get_indices_sphere([2, 2, sphere_depth], sphere_radius, mesh.gridCC)
    model_array[sphere_ind2] = 100  #susceptibility

    # 2. Survey (De Drone Vlucht)
    xr = np.linspace(-5, 5, 50)
    yr = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(xr, yr)
    Z = np.ones_like(X) * 1.0 
    rx_locs = np.c_[X.flatten(), Y.flatten(), Z.flatten()]

    receiver_list = [magnetics.receivers.Point(rx_locs, components=["tmi"])]
    source_field = magnetics.sources.UniformBackgroundField(
        receiver_list=receiver_list,
        amplitude=49300, inclination=60, declination=2.5
    )
    survey = magnetics.survey.Survey(source_field)
    
    # 3. Bereken Data
    simulation = magnetics.simulation.Simulation3DIntegral(
        mesh=mesh, survey=survey, chiMap=maps.IdentityMap(nP=mesh.nC)
    )
    dpred = simulation.dpred(model_array)
    
    # 4. OPSLAAN
    # We slaan de locaties (rx_locs) en de meetwaarden (dpred) op in één bestand
    np.savez(bestandsnaam, rx_locs=rx_locs, dpred=dpred)
    print(f"   -> Bestand '{bestandsnaam}' is aangemaakt en opgeslagen.\n")

# ==============================================================================
# STAP 2: DATA LADEN (De "Upload" Functie)
# Hier haal je de data binnen, of het nu gesimuleerd is of van een echte drone.
# ==============================================================================
def laad_mag_data(bestandsnaam):
    print(f"--- LADEN: Bezig met inlezen van {bestandsnaam} ---")
    try:
        # Laad het .npz bestand (NumPy Zip)
        data = np.load(bestandsnaam)
        rx_locs = data['rx_locs']
        dpred = data['dpred']
        print(f"   -> Succesvol geladen: {len(dpred)} meetpunten.")
        return rx_locs, dpred
    except FileNotFoundError:
        print(f"   [FOUT] Bestand '{bestandsnaam}' niet gevonden.")
        return None, None

# ==============================================================================
# STAP 3: ANALYSE ALGORITMES (Analytisch Signaal)
# ==============================================================================

def vind_object_locatie(rx_locs, dpred):
    """
    Gebruikt Analytisch Signaal (Total Horizontal Gradient) om de locatie te vinden.
    Corrigeert voor de dipool-afwijking.
    """
    # 1. Grid de data
    x = rx_locs[:, 0]
    y = rx_locs[:, 1]
    xi = np.linspace(min(x), max(x), 50) 
    yi = np.linspace(min(y), max(y), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), dpred, (Xi, Yi), method='cubic')

    # 2. Bereken Gradiënten (dx, dy)
    dy, dx = np.gradient(Zi)
    
    # 3. Bereken Analytisch Signaal (THG)
    # Dit maakt van de plus/min dipool één enkele positieve bult
    as_signal = np.sqrt(dx**2 + dy**2) 
    
    # 4. Vind de piek
    max_idx = np.unravel_index(np.argmax(as_signal), as_signal.shape)
    
    gevonden_x = xi[max_idx[1]]
    gevonden_y = yi[max_idx[0]]
    
    return gevonden_x, gevonden_y

def analyze_mag_data_oud(rx_locs, dpred, threshold_percentile=95):
    """(Oude methode: Kijkt puur naar max waarde/rode vlek)"""
    x = rx_locs[:, 0]
    y = rx_locs[:, 1]
    xi = np.linspace(min(x), max(x), 50) 
    yi = np.linspace(min(y), max(y), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), dpred, (Xi, Yi), method='cubic')
    
    signal_strength = np.abs(Zi) 
    thresh = np.percentile(signal_strength, threshold_percentile)
    binary_mask = signal_strength > thresh
    labeled_array, num_features = label(binary_mask)
    
    detecties = []
    for i in range(1, num_features + 1):
        cy, cx = center_of_mass(binary_mask * signal_strength, labeled_array, i)
        real_x = xi[0] + cx * (xi[1] - xi[0])
        real_y = yi[0] + cy * (yi[1] - yi[0])
        detecties.append({'x': real_x, 'y': real_y})
        
    return detecties

# ==============================================================================
# HOOFD PROGRAMMA (MAIN)
# ==============================================================================

if __name__ == "__main__":
    # NAAM VAN HET BESTAND
    # Dit is het bestand dat je "uploadt" of wilt analyseren
    mijn_bestand = "mag_testdata.npz"

    # A. Check of bestand bestaat. Zo niet, genereer het (voor deze test).
    if not os.path.exists(mijn_bestand):
        print("Geen bestand gevonden. Er wordt een synthetische test-set gegenereerd...")
        genereer_test_bestand(mijn_bestand)
    
    # B. Laad de data
    rx_locs, dpred = laad_mag_data(mijn_bestand)

    if rx_locs is not None:
        # C. Voer Detectie uit (Nieuwe Methode)
        exacte_x, exacte_y = vind_object_locatie(rx_locs, dpred)
        print(f"\n--- RESULTAAT ANALYTISCH SIGNAAL ---")
        print(f"Gedetecteerde locatie: X={exacte_x:.2f}, Y={exacte_y:.2f}")

        # D. Voer Detectie uit (Oude Methode - ter vergelijking)
        oude_detecties = analyze_mag_data_oud(rx_locs, dpred)

        # E. Visualiseer
        plt.figure(figsize=(10, 8))
        
        # Plot de ruwe data (SimPEG functie, of contourf als SimPEG niet aanwezig is)
        try:
            utils.plot2Ddata(rx_locs, dpred, contourOpts={"cmap": "jet"})
        except:
            # Fallback als utils niet werkt (voor pure data visualisatie)
            plt.tricontourf(rx_locs[:,0], rx_locs[:,1], dpred, 20, cmap="jet")
            plt.colorbar(label="TMI (nT)")

        # Plot resultaat NIEUWE methode (Groen)
        plt.scatter(exacte_x, exacte_y, c='lime', marker='*', s=300, edgecolors='black', label='Analytisch Signaal (Correct)')

        # Plot resultaat OUDE methode (Wit)
        for obj in oude_detecties:
            plt.scatter(obj['x'], obj['y'], c='white', marker='x', s=100, linewidth=2, label='Max Waarde (Verschoven)')

        plt.title(f"Analyse van bestand: {mijn_bestand}")
        plt.xlabel("Oostwaarts (m)")
        plt.ylabel("Noordwaarts (m)")
        plt.legend()
        plt.show()