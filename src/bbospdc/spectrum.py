import sys
from pathlib import Path
# Python'a 'src' klasörünün yerini gösteriyoruz
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from bbospdc.bbo import no, ne_theta
from bbospdc.config import SIM_PARAMS

def calculate_delta_k(signal_wl_nm):
    """
    Verilen sinyal dalga boyu için momentum uyuşmazlığını (Delta k) hesaplar.
    Enerji korunumu: 1/lambda_p = 1/lambda_s + 1/lambda_i
    """
    pump_wl_um = SIM_PARAMS["pump"]["wavelength_nm"] / 1000.0
    signal_wl_um = signal_wl_nm / 1000.0
    
    # Enerji korunumundan idler dalga boyunu bul
    idler_wl_um = 1.0 / ((1.0 / pump_wl_um) - (1.0 / signal_wl_um))
    
    # Kristal açısını YAML'dan çek ve radyana çevir
    theta_deg = SIM_PARAMS["crystal"]["cut_angle_deg"]
    theta_rad = np.radians(theta_deg)
    
    # Kırılma indislerini hesapla (Type-II: e -> o + e)
    n_p = ne_theta(pump_wl_um, theta_rad)
    n_s = no(signal_wl_um)
    n_i = ne_theta(idler_wl_um, theta_rad)
    
    # Dalga vektörlerini (k = 2 * pi * n / lambda) hesapla
    # Birim: um^-1
    k_p = 2 * np.pi * n_p / pump_wl_um
    k_s = 2 * np.pi * n_s / signal_wl_um
    k_i = 2 * np.pi * n_i / idler_wl_um
    
    return k_p - k_s - k_i

def phase_matching_function(signal_wl_nm):
    """sinc^2(Delta k * L / 2) fonksiyonunu hesaplar."""
    delta_k = calculate_delta_k(signal_wl_nm)
    L_um = SIM_PARAMS["crystal"]["thickness_mm"] * 1000.0
    
    # numpy.sinc(x) fonksiyonu sin(pi * x) / (pi * x) hesaplar
    arg = (delta_k * L_um / 2.0) / np.pi
    return np.sinc(arg)**2

if __name__ == "__main__":
    # 790 nm ile 830 nm arasındaki dalga boylarını tarayalım
    wavelengths = np.linspace(790, 830, 500)
    pmf_values = [phase_matching_function(wl) for wl in wavelengths]
    
    # Grafiği çiz
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, pmf_values, color='blue', linewidth=2)
    plt.axvline(810, color='red', linestyle='--', alpha=0.5, label='Degenerate (810 nm)')
    plt.title(f"SPDC Spektrumu (Type-II BBO, L = {SIM_PARAMS['crystal']['thickness_mm']} mm)")
    plt.xlabel("Sinyal Dalga Boyu (nm)")
    plt.ylabel("Uretim Ihtimali (Sinc^2)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Grafiği results klasörüne kaydet ve ekranda göster
    plt.savefig("results/spdc_spectrum.png", dpi=300)
    print("Grafik 'results/spdc_spectrum.png' olarak kaydedildi!")
    plt.show()