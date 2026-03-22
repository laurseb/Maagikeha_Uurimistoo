import numpy as np
import json
from skimage.filters import sobel
from skimage.measure import label, regionprops

GRID_SPACING = 50   # peab kattuma maagikeha3d.py spacing-iga

MODEL_GRAV = np.load("model_grav.npy")   # mGal
MODEL_MAG  = np.load("model_mag.npy")    # nT


def amplitude(field):
    return float(np.nanmax(field) - np.nanmin(field))


def anomaly_mask(field, k=0.5):
    """Kõik pikslid, mis ületavad keskmise + k*std."""
    thr = np.nanmean(field) + k * np.nanstd(field)
    return field > thr


def equivalent_diameter(mask, spacing):
    """Ruutjuur pindalast ligikaudne laius meetrites."""
    area = np.sum(mask) * spacing ** 2
    return float(np.sqrt(area))


def concentration_energy(field, mask):
    """Keskmine absoluutväärtus anomaalia sees."""
    return float(np.sum(np.abs(field[mask])) / np.sum(mask))


def mean_gradient(field, mask):
    """Sobeliga arvutatud keskmine gradient anomaalia piirialal."""
    grad = np.hypot(sobel(field, axis=0), sobel(field, axis=1))
    return float(np.nanmean(grad[mask]))


def orientation_deg(mask):
    """Peatelge nurk kraadides (regionprops.orientation)."""
    labels = label(mask)
    props = regionprops(labels)
    if not props:
        return 0.0
    return float(np.degrees(props[0].orientation))


def extract_model_features(field, spacing):
    mask = anomaly_mask(field)
    if not np.any(mask):
        raise ValueError("Mudeli anomaaliamaski alla ei jää ühtegi pikslit – "
                         "kontrolli spacing ja k parameetreid.")
    return {
        "amplitude":       amplitude(field),
        "width_m":         equivalent_diameter(mask, spacing),
        "concentration":   concentration_energy(field, mask),
        "gradient":        mean_gradient(field, mask),
        "orientation_deg": orientation_deg(mask),
        "area_m2":         float(np.sum(mask) * spacing ** 2),
    }


model_grav_feat = extract_model_features(MODEL_GRAV, GRID_SPACING)
model_mag_feat  = extract_model_features(MODEL_MAG,  GRID_SPACING)

print("Mudeli gravitatsiooni tunnused:")
for k, v in model_grav_feat.items():
    print(f"  {k}: {v:.4f}")

print("\nMudeli magnetilised tunnused:")
for k, v in model_mag_feat.items():
    print(f"  {k}: {v:.4f}")

with open("model_features.json", "w", encoding="utf-8") as f:
    json.dump(
        {"gravity": model_grav_feat, "magnetic": model_mag_feat},
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\nSalvestatud: model_features.json")
