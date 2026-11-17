# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from bbospdc.config import SIM_PARAMS

def plot_type2_rings():
    """
    Type-II BBO kristalindeki sinyal (o-ray) ve idler (e-ray) emisyon konilerinin
    kamera sensörüne (uzaysal düzleme) düşen teorik izdüşümünü çizer.
    """
    theta = np.linspace(0, 2*np.pi, 500)
    
    radius = 3.0       # Yaklaşık koni açısı (derece)
    walk_off = 2.4     # Walk-off kayması (derece)
    
    # Sinyal Konisi (Ordinary) - Yukarı kaymış
    sig_x = radius * np.cos(theta)
    sig_y = radius * np.sin(theta) + (walk_off / 2)
    
    # Idler Konisi (Extraordinary) - Aşağı kaymış
    idl_x = radius * np.cos(theta)
    idl_y = radius * np.sin(theta) - (walk_off / 2)
    
    # Kesişim noktaları
    intersect_x = np.sqrt(radius**2 - (walk_off/2)**2)
    
    plt.figure(figsize=(7, 7))
    plt.plot(sig_x, sig_y, label="Signal Cone (o-ray, 810nm)", color="blue", linewidth=2)
    plt.plot(idl_x, idl_y, label="Idler Cone (e-ray, 810nm)", color="red", linewidth=2)
    
    plt.scatter([intersect_x, -intersect_x], [0, 0], color="purple", zorder=5, s=120, 
                label="Entangled Photon Intersections\n(Fiber coupling locations)")
    
    plt.title("Spatial Distribution of Type-II SPDC")
    plt.xlabel("Horizontal Emission Angle (Degrees)")
    plt.ylabel("Vertical Emission Angle (Degrees)")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.axis('equal')
    
    plt.show()

# %%
plot_type2_rings()