import numpy as np
import matplotlib.pyplot as plt
import harmonica as hm
from scipy.spatial.transform import Rotation as R

# ==========================================================
# MUUTUJAD (vaba voli)
# ==========================================================

# -------- omadused --------
rho_host = 2850          # aluskord kg/m^3
rho_body = 4600          # maagikeha kg/m^3
chi_host = 0.004         # aluskorra magnetiline läbitavus (SI)
chi_body = 0.1           # keha magnetiline vastuvõtlikkus (SI)

# -------- keha omadused --------
length = 2000      # keha pikkus (m)
width = 200        # keha laius (m)
thickness = 50     # keha kõrgus (ülemisest servast alla) (m)
top_depth = 200    # sügavus (ülemisest servast) (m)

dip_deg = 0        # DIP ANGLE
strike_deg = 0     # suund (0 = N-S)

# -------- uuritav ala --------
area = (-3000, 3000, -3000, 3000)
spacing = 50       # joonise jaotis (täpsus) – peab kattuma anomaalialeidja GRID_SPACING-iga
observation_height = 0

# ==========================================================
# muud arvud (mitte muuta)
# ==========================================================

earth_field_nT = 50000
delta_rho = (rho_body - rho_host)/2
delta_chi = (chi_body - chi_host)/2
mu0 = 4 * np.pi * 1e-7
H = earth_field_nT * 1e-9 / mu0
magnetization = delta_chi * H

bottom_depth = top_depth + thickness

# ==========================================================
# uuritav ala
# ==========================================================

xmin, xmax, ymin, ymax = area

easting  = np.arange(xmin, xmax + spacing, spacing)
northing = np.arange(ymin, ymax + spacing, spacing)

easting, northing = np.meshgrid(easting, northing)
upward = np.zeros_like(easting) + observation_height

# ==========================================================
# prisma definitsioon
# ==========================================================

x1 = -width / 2
x2 =  width / 2
y1 = -length / 2
y2 =  length / 2
z1 = -top_depth
z2 = -bottom_depth

prism  = np.array([[x1, x2, y1, y2, z2, z1]])
Mprism = np.array([[x1, x2, y1, y2, z2 - 100, z1 - 100]])
# ==========================================================
# keha suund ja kallak
# ==========================================================

if dip_deg != 0:
    rotation_axis = [0, 1, 0]
    rot = R.from_rotvec(np.radians(dip_deg) * np.array(rotation_axis))
    corners = np.array([
        [x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],
    ])
    rotated = rot.apply(corners)
    x_min, y_min, z_min = rotated.min(axis=0)
    x_max, y_max, z_max = rotated.max(axis=0)
    z_min = min(rotated[:, 2])
    z_max = max(rotated[:, 2])
    prism = np.array([[x_min, x_max, y_min, y_max, z_min, z_max]])

# ==========================================================
# GRAVITATSIOON MUDEL
# ==========================================================

gz = hm.prism_gravity(
    coordinates=(easting, northing, upward),
    prisms=prism,
    density=delta_rho,
    field="g_z"
)

gz_mgal = gz - np.nanmean(gz)

# ==========================================================
# MAGNET MUDEL (Induced)
# ==========================================================

inclination_deg = 70
declination_deg = 10

inc = np.radians(inclination_deg)
dec = np.radians(declination_deg)

Fx = np.cos(inc) * np.sin(dec)
Fy = np.cos(inc) * np.cos(dec)
Fz = np.sin(inc)

Mx = magnetization * Fx
My = magnetization * Fy
Mz = magnetization * Fz

b_e = hm.prism_magnetic((easting, northing, upward), Mprism, magnetization=(Mx, My, Mz), field="b_e")
b_n = hm.prism_magnetic((easting, northing, upward), Mprism, magnetization=(Mx, My, Mz), field="b_n")
b_u = hm.prism_magnetic((easting, northing, upward), Mprism, magnetization=(Mx, My, Mz), field="b_u")

mag_nT = b_e * Fx + b_n * Fy + b_u * Fz
mag_nT = mag_nT - np.nanmean(mag_nT)

# ==========================================================
# INFO
# ==========================================================

print(f"Gravi amplituud : {np.max(gz_mgal) - np.min(gz_mgal):.4f} mGal")
print(f"Magnet amplituud: {np.max(mag_nT)  - np.min(mag_nT):.4f} nT")
print(f"Gravi max       : {np.max(gz_mgal):.4f} mGal")
print(f"Magnet max      : {np.max(mag_nT):.4f} nT")

# ==========================================================
# SALVESTA – normaliseerimata väljad (raw anomaalia)
# FIX: salvestatakse normaliseerimata andmed, et anomaaliateandmeextract.py
#      saaks ise arvutada tegelikud amplituudid, gradiendid jne.
#      Varem salvestati 0..1 normaliseeritud väljad, mis moonutasid
#      amplituudi- ja gradiendivõrdlust pärisandmetega.
# ==========================================================

np.save("model_grav.npy", gz_mgal)   # mGal, float64
np.save("model_mag.npy",  mag_nT)    # nT,   float64

# ==========================================================
# KUJUTAMINE
# ==========================================================

center_x = 0
center_y = 0

# Leia gridis sellele lähim rida
row = np.argmin(np.abs(northing[:,0] - center_y))

profile_x = easting[row, :]
grav_profile = gz_mgal[row, :]
mag_profile  = mag_nT[row, :]

plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.title("Raskusjõu anomaalia (mGal)")
plt.pcolormesh(easting, northing, gz_mgal, shading="auto", cmap="RdBu_r")
plt.colorbar(label="mGal")
plt.xlabel("W-E (m)")
plt.ylabel("S-N (m)")

plt.subplot(2, 1, 2)
plt.title("Magnetvälja anomaalia (nT)")
plt.pcolormesh(easting, northing, mag_nT, shading="auto", cmap="RdBu_r")
plt.colorbar(label="nT")
plt.xlabel("W-E (m)")
plt.ylabel("S-N (m)")

plt.tight_layout()
#plt.savefig("3d100lai200s05x.png", dpi=150)
#plt.show()

fig, ax1 = plt.subplots(figsize=(10,5))

# gravitatsioon
ax1.plot(profile_x, grav_profile, color="blue", label="Gravity (mGal)")
ax1.set_xlabel("W-E (m)")
ax1.set_ylabel("Raskusjõu anomaalia (mGal)", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

# teine telg magnetile
ax2 = ax1.twinx()
ax2.plot(profile_x, mag_profile, color="red", label="Magnetic (nT)")
ax2.set_ylabel("Magnetvälja anomaalia (nT)", color="red")
ax2.tick_params(axis='y', labelcolor="red")

# keha asukoha joon
ax1.axvline(0, color="black", linestyle="--", alpha=0.6)

plt.title("Anomaalia läbilõige keha keskpunktist")
plt.grid(True)

plt.tight_layout()
#plt.savefig("2d100lai200s05x.png", dpi=150)
#plt.show()