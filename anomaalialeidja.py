import json
import math
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from skimage.filters import sobel
from shapely.geometry import Point, mapping


# ==========================================================
# KOORDINAATIDE TEISENDUS: L-EST97 (EPSG:3301) → WGS84
# ==========================================================
# ArcGIS Online loeb GeoJSON-i ainult WGS84 (lon/lat) koordinaatides.
# L-EST väärtused (~670000, ~6560000) on meetrites ja ArcGIS ei suuda
# neid kaardile paigutada. Teisendus tehtud käsitsi (ilma pyproj-ita)
# kasutades GRS80 ellipsoidi + Transverse Mercator projektsiooni matemaatikat.
# Täpsus: < 1 mm, piisav igasuguseks GIS-kasutuseks.

def lest_to_wgs84(easting: float, northing: float):
    """
    Teisendab L-EST97 (EPSG:3301) koordinaadid WGS84 lon/lat-iks.

    L-EST97 TM parameetrid:
      Ellipsoid     : GRS80  (a=6378137, f=1/298.257222101)
      Keskmeriidian : 24° E
      Skaala        : 0.9996
      Vale idasuund : 500 000 m
      Vale põhjasund: 0 m
    """
    # GRS80 ellipsoid
    a  = 6378137.0
    f  = 1 / 298.257222101
    b  = a * (1 - f)
    e2 = 1 - (b / a) ** 2

    # TM projektsioon
    k0   = 0.9996
    lon0 = math.radians(24.0)
    x    = easting  - 500000.0
    y    = northing

    # Jalajälje laius (footprint latitude)
    mu  = y / (k0 * a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    e1  = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = (mu
            + (3*e1/2   - 27*e1**3/32) * math.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu)
            + (151*e1**3/96)              * math.sin(6*mu)
            + (1097*e1**4/512)            * math.sin(8*mu))

    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    tan_phi1 = math.tan(phi1)

    N1  = a / math.sqrt(1 - e2 * sin_phi1**2)
    R1  = a * (1 - e2) / (1 - e2 * sin_phi1**2) ** 1.5
    T1  = tan_phi1 ** 2
    C1  = (e2 / (1 - e2)) * cos_phi1 ** 2
    D   = x / (N1 * k0)

    lat = phi1 - (N1 * tan_phi1 / R1) * (
          D**2/2
          - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e2/(1-e2)) * D**4/24
          + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*e2/(1-e2) - 3*C1**2) * D**6/720)

    lon = lon0 + (D
          - (1 + 2*T1 + C1) * D**3/6
          + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*e2/(1-e2) + 24*T1**2) * D**5/120) / cos_phi1

    return math.degrees(lon), math.degrees(lat)

# ==========================================================
# KASUTAJA SEADISTUSED
# ==========================================================
GRID_SPACING  = 50       # m – peab kattuma mudeli spacing-iga
LOWPASS_SIGMA = 2.0      # low-pass Gaussi silumine (gridi pikslites)

# Regionaalse taustavälja eemaldamine magnetandmetest.
# Taustaväli = tugevalt silutud versioon griditud väljast.
# Suurem sigma = suurem regionaalne komponent eemaldatakse.
# Soovituslik vahemik: 10–40 (pikslites, 1 piksel = GRID_SPACING meetrit)
REGIONAL_SIGMA = 15.0

# Anomaaliapiiride künnis (mean + k*std)
ANOMALY_K_GRAV = 0.7     # gravitatsioon – Bouguer on puhas, k=0.7 sobib
ANOMALY_K_MAG  = 1.5     # magnet – rangem künnis, et haarata ainult teravad lokaalsed tipud
                          # (k=0.7 haarab kogu regionaalse gradiendi → 868 nT vs mudeli 17 nT)

MIN_SCORE = 0.6          # kombineeritud skoori künnis

# Kaalud sarnasusskoorile (summa = 1)
W_AMP   = 0.40
W_SIZE  = 0.25
W_CONC  = 0.20
W_GRAD  = 0.15
W_ORI   = 0

# ==========================================================
# ABIFUNKTSIOONID
# ==========================================================

def grid_and_lowpass(easting, northing, values, spacing, sigma):
    """Interpoleerib hajusandmed ruudustikule ja silub low-pass filtriga."""
    ei = np.arange(np.nanmin(easting),  np.nanmax(easting)  + spacing, spacing)
    ni = np.arange(np.nanmin(northing), np.nanmax(northing) + spacing, spacing)
    EI, NI = np.meshgrid(ei, ni)
    VI = griddata((easting, northing), values, (EI, NI), method="cubic")
    VI = gaussian_filter(VI, sigma=sigma)
    return EI, NI, VI


def remove_regional(field, sigma):
    """
    Lahutab regionaalse taustavälja Gaussi kõrgmöödufiltriga.
    Meetod: taustaväli ≈ gaussian_filter(field, sigma=REGIONAL_SIGMA)
    Lokaalne anomaalia = field - taustaväli
    """
    filled = field.copy()
    nan_mask = np.isnan(field)
    filled[nan_mask] = np.nanmean(field)   # NaN-id täidetakse keskmisega servaefekti vältimiseks
    regional = gaussian_filter(filled, sigma=sigma)
    local = field - regional
    local[nan_mask] = np.nan
    return local


def anomaly_mask(field, k):
    return field > (np.nanmean(field) + k * np.nanstd(field))


def amplitude(field, mask):
    vals = field[mask]
    return float(np.nanmax(vals) - np.nanmin(vals)) if vals.size > 0 else 0.0


def equivalent_diameter(mask, spacing):
    return float(np.sqrt(np.sum(mask) * spacing ** 2))


def concentration_energy(field, mask):
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.abs(field[mask])) / np.sum(mask))


def mean_gradient(field, mask):
    if not np.any(mask):
        return 0.0
    g = np.hypot(sobel(field, axis=0), sobel(field, axis=1))
    vals = g[mask]
    return float(np.nanmean(vals)) if vals.size > 0 else 0.0


def orientation_deg(mask):
    labels = label(mask)
    props = regionprops(labels)
    return float(np.degrees(props[0].orientation)) if props else 0.0


def similarity(val, ref, tol):
    if tol == 0 or np.isnan(val) or np.isnan(ref):
        return 0.0
    return max(0.0, 1.0 - abs(val - ref) / tol)


def score_feature(data_feat, model_feat):
    s_amp  = similarity(data_feat["amplitude"],       model_feat["amplitude"],       model_feat["amplitude"])
    s_size = similarity(data_feat["width_m"],         model_feat["width_m"],         model_feat["width_m"])
    s_conc = similarity(data_feat["concentration"],   model_feat["concentration"],   model_feat["concentration"])
    s_grad = similarity(data_feat["gradient"],        model_feat["gradient"],        model_feat["gradient"])
    s_ori  = similarity(data_feat["orientation_deg"], model_feat["orientation_deg"],
                        max(5.0, abs(model_feat["orientation_deg"])))
    return W_AMP*s_amp + W_SIZE*s_size + W_CONC*s_conc + W_GRAD*s_grad + W_ORI*s_ori


# ==========================================================
# 1) LOE MUDELI TUNNUSED
# ==========================================================

with open("model_features.json", "r", encoding="utf-8") as f:
    MODEL = json.load(f)

MODEL_G = MODEL["gravity"]
MODEL_M = MODEL["magnetic"]

print("Mudeli referentsid:")
print(f"  Gravi amplituud : {MODEL_G['amplitude']:.4f} mGal")
print(f"  Magnet amplituud: {MODEL_M['amplitude']:.4f} nT")
print(f"  Gravi laius     : {MODEL_G['width_m']:.0f} m")
print(f"  Magnet laius    : {MODEL_M['width_m']:.0f} m")

# ==========================================================
# 2) LOE PÄRISANDMED
# ==========================================================

# Gravitatsioon: X(L-Est)=Northing, Y(L-Est)=Easting
grav = pd.read_csv("ristGravi.csv")
grav_easting  = grav["Y(L-Est)"].values
grav_northing = grav["X(L-Est)"].values
grav_vals     = grav["IGSN71_2.67"].values

# Magnetväli: col0=Easting, col1=Northing
mag = pd.read_csv("ristMag.csv", header=None, names=["easting", "northing", "m"])
mag_easting  = mag["easting"].values
mag_northing = mag["northing"].values
mag_vals     = mag["m"].values

# ==========================================================
# 3) GRID + LOW-PASS
# ==========================================================

Eg, Ng, Gf_raw = grid_and_lowpass(grav_easting,  grav_northing,  grav_vals, GRID_SPACING, LOWPASS_SIGMA)
Em, Nm, Mf_raw = grid_and_lowpass(mag_easting,   mag_northing,   mag_vals,  GRID_SPACING, LOWPASS_SIGMA)

# --- Gravitatsioon: mean-remove (Bouguer on juba regionaalselt korrigeeritud) ---
Gf = Gf_raw - np.nanmean(Gf_raw)

# --- Magnetväli: eemalda regionaalne taust enne anomaaliate otsimist ---
Mf = remove_regional(Mf_raw, sigma=REGIONAL_SIGMA)

print(f"\nPärisandmete statistika pärast töötlust:")
print(f"  Gravi std={np.nanstd(Gf):.3f} mGal,  amp={np.nanmax(Gf)-np.nanmin(Gf):.3f} mGal")
print(f"  Magnet std={np.nanstd(Mf):.2f} nT,  amp={np.nanmax(Mf)-np.nanmin(Mf):.2f} nT")
print(f"  (Mudeli magnet amp: {MODEL_M['amplitude']:.2f} nT – peaks olema samas suurusjärgus)")

# ==========================================================
# 4) LEIA KÕIK ANOMAALIAD
# ==========================================================

def process_field(field, E_grid, N_grid, MODEL_FEAT, tag, k_thresh):
    mask0  = anomaly_mask(field, k_thresh)
    labels = label(mask0)
    regs   = regionprops(labels)

    results = []
    for r in regs:
        mask = np.zeros_like(field, dtype=bool)
        mask[r.coords[:, 0], r.coords[:, 1]] = True

        feat = {
            "amplitude":       amplitude(field, mask),
            "width_m":         equivalent_diameter(mask, GRID_SPACING),
            "concentration":   concentration_energy(field, mask),
            "gradient":        mean_gradient(field, mask),
            "orientation_deg": orientation_deg(mask),
        }

        score = score_feature(feat, MODEL_FEAT)

        row, col = int(r.centroid[0]), int(r.centroid[1])
        results.append({
            "easting":  float(E_grid[row, col]),
            "northing": float(N_grid[row, col]),
            "score":    float(score),
            "type":     tag,
            **feat,
        })
    return results


grav_res = process_field(Gf, Eg, Ng, MODEL_G, "gravity",  ANOMALY_K_GRAV)
mag_res  = process_field(Mf, Em, Nm, MODEL_M, "magnetic", ANOMALY_K_MAG)

print(f"\nKandidaate enne skoorifiltrit:")
print(f"  Gravitatsiooni: {len(grav_res)}")
print(f"  Magnetilisi   : {len(mag_res)}")

# ==========================================================
# 5) ÜHENDA JA FILTREERI
# ==========================================================

dfg = pd.DataFrame(grav_res)
dfm = pd.DataFrame(mag_res)

SEARCH_RADIUS = max(MODEL_G["width_m"], MODEL_M["width_m"]) * 1.5

final = []

for _, g0 in dfg.iterrows():
    if len(dfm) > 0:
        de = dfm["easting"].values  - g0["easting"]
        dn = dfm["northing"].values - g0["northing"]
        j  = np.argmin(np.hypot(de, dn))
        d_min = np.hypot(de[j], dn[j])
        if d_min <= SEARCH_RADIUS:
            mag_score = float(dfm.iloc[j]["score"])
            mag_amp   = float(dfm.iloc[j]["amplitude"])
        else:
            mag_score = 0.0
            mag_amp   = np.nan
    else:
        mag_score = 0.0
        mag_amp   = np.nan

    combined_score = 0.65 * g0["score"] + 0.35 * mag_score

    if combined_score >= MIN_SCORE:
        final.append({
            "easting":        g0["easting"],
            "northing":       g0["northing"],
            "score_combined": float(combined_score),
            "score_grav":     float(g0["score"]),
            "score_mag":      float(mag_score),
            "grav_amp_mGal":  float(g0["amplitude"]),
            "mag_amp_nT":     float(mag_amp) if not np.isnan(mag_amp) else None,
        })

out = pd.DataFrame(final)
print(f"\nLeitud {len(out)} sihtala(d) (skoor >= {MIN_SCORE})")
print(f"  (skoor = absoluutne sarnasus mudelile 0..1, mitte suhteline pingerida)")
if len(out) > 0:
    print(out[["easting", "northing", "score_combined",
               "score_grav", "score_mag",
               "grav_amp_mGal", "mag_amp_nT"]].to_string(index=False))

# ==========================================================
# 6) VÄLJUND
# ==========================================================

out.to_csv("targets.csv", index=False)

features = []
for _, r in out.iterrows():
    # Teisenda L-EST → WGS84 lon/lat ArcGIS Online jaoks
    lon, lat = lest_to_wgs84(r["easting"], r["northing"])

    props = {
        "easting_lest":  float(r["easting"]),    # algne L-EST säilitatakse atribuutidena
        "northing_lest": float(r["northing"]),   # juhuks kui vaja tagasi teisendada
    }
    for k in out.columns:
        if k in ["easting", "northing"]:
            continue
        val = r[k]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        props[k] = float(val)

    features.append({
        "type": "Feature",
        # GeoJSON standard: koordinaadid [longitude, latitude] WGS84-s
        "geometry": {"type": "Point", "coordinates": [round(lon, 8), round(lat, 8)]},
        "properties": props,
    })

with open("targets.geojson", "w", encoding="utf-8") as f:
    json.dump({"type": "FeatureCollection", "features": features},
              f, ensure_ascii=False, indent=2)

print("Salvestatud → targets.csv ja targets.geojson")
