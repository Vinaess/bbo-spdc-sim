# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from bbospdc.bbo import no, ne_theta
from bbospdc.config import SIM_PARAMS

def calculate_delta_k(signal_wl_nm):
    pump_wl_um = SIM_PARAMS["pump"]["wavelength_nm"] / 1000.0
    signal_wl_um = signal_wl_nm / 1000.0
    idler_wl_um = 1.0 / ((1.0 / pump_wl_um) - (1.0 / signal_wl_um))
    
    theta_rad = np.radians(SIM_PARAMS["crystal"]["cut_angle_deg"])
    
    n_p = ne_theta(pump_wl_um, theta_rad)
    n_s = no(signal_wl_um)
    n_i = ne_theta(idler_wl_um, theta_rad)
    
    k_p = 2 * np.pi * n_p / pump_wl_um
    k_s = 2 * np.pi * n_s / signal_wl_um
    k_i = 2 * np.pi * n_i / idler_wl_um
    
    return k_p - k_s - k_i

def phase_matching_function(signal_wl_nm):
    delta_k = calculate_delta_k(signal_wl_nm)
    L_um = SIM_PARAMS["crystal"]["thickness_mm"] * 1000.0
    arg = (delta_k * L_um / 2.0) / np.pi
    return np.sinc(arg)**2

def calculate_fwhm():
    """Bant genişliğini (FWHM) sayısal olarak tam hassasiyetle hesaplar."""
    def root_func(wl):
        return phase_matching_function(wl) - 0.5

    # Zirve noktası 810 nm. Sol ve sağ %50 (0.5) ihtimal noktalarını bul
    left_root = brentq(root_func, 800, 810)
    right_root = brentq(root_func, 810, 820)
    
    fwhm = right_root - left_root
    return fwhm, left_root, right_root

def plot_spectrum_with_fwhm():
    """FWHM hesaplamalı ve İngilizce etiketli gelişmiş spektrum çizimi."""
    wavelengths = np.linspace(790, 830, 500)
    pmf_values = [phase_matching_function(wl) for wl in wavelengths]
    
    # FWHM Hesabı ve Terminal Çıktısı
    fwhm, left_wl, right_wl = calculate_fwhm()
    print("\n" + "="*50)
    print(f"FWHM (Bant Genişliği): {fwhm:.3f} nm")
    print(f"Alt Sınır (50%): {left_wl:.3f} nm")
    print(f"Üst Sınır (50%): {right_wl:.3f} nm")
    print("="*50 + "\n")
    
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, pmf_values, color='blue', linewidth=2, label="SPDC Spectrum")
    plt.axvline(810, color='red', linestyle='--', alpha=0.5, label='Degenerate (810 nm)')
    
    # FWHM göstergesini grafiğe ekle
    plt.hlines(0.5, left_wl, right_wl, color='green', linewidth=2, label=f'FWHM = {fwhm:.2f} nm')
    plt.scatter([left_wl, right_wl], [0.5, 0.5], color='green', zorder=5)
    
    plt.title(f"SPDC Spectrum (Type-II BBO, L = {SIM_PARAMS['crystal']['thickness_mm']} mm)")
    plt.xlabel("Signal Wavelength (nm)")
    plt.ylabel("Probability (Sinc^2)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("results/spdc_spectrum_fwhm.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # --- ESKİ VERSİYON (Sadece temel grafiği çizen kod - Tarihçe için saklandı) ---
    """
    wavelengths = np.linspace(790, 830, 500)
    pmf_values = [phase_matching_function(wl) for wl in wavelengths]
    
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, pmf_values, color='blue', linewidth=2)
    plt.axvline(810, color='red', linestyle='--', alpha=0.5, label='Degenerate (810 nm)')
    plt.title(f"SPDC Spektrumu (Type-II BBO, L = {SIM_PARAMS['crystal']['thickness_mm']} mm)")
    plt.xlabel("Sinyal Dalga Boyu (nm)")
    plt.ylabel("Üretim İhtimali (Sinc^2)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    
    # --- YENİ VERSİYON (Aktif olan kod) ---
    plot_spectrum_with_fwhm()