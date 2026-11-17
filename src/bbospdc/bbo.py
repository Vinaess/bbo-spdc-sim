import numpy as np
from scipy.optimize import brentq

# BBO Sellmeier Katsayıları (Kaynak: Eimerl et al., 1987)
def no(wavelength_um):
    """BBO için Ordinary (o-ray) kırılma indisini hesaplar."""
    l_sq = wavelength_um**2
    n_sq = 2.7359 + 0.01878 / (l_sq - 0.01822) - 0.01354 * l_sq
    return np.sqrt(n_sq)

def ne(wavelength_um):
    """BBO için Principal Extraordinary (e-ray) kırılma indisini hesaplar."""
    l_sq = wavelength_um**2
    n_sq = 2.3753 + 0.01224 / (l_sq - 0.01667) - 0.01516 * l_sq
    return np.sqrt(n_sq)

def ne_theta(wavelength_um, theta_rad):
    """Belirli bir açıda (theta) efektif Extraordinary kırılma indisini hesaplar."""
    n_o_val = no(wavelength_um)
    n_e_val = ne(wavelength_um)
    inv_n_sq = (np.cos(theta_rad)**2 / n_o_val**2) + (np.sin(theta_rad)**2 / n_e_val**2)
    return np.sqrt(1.0 / inv_n_sq)

def type2_phase_matching_angle(pump_wl, signal_wl, idler_wl):
    """
    Type-II (e -> o + e) SPDC için momentum korunumunu sağlayan faz uyumu açısını hesaplar.
    Formül: k_pump = k_signal + k_idler
    """
    def momentum_mismatch(theta):
        n_p = ne_theta(pump_wl, theta)
        n_s = no(signal_wl)
        n_i = ne_theta(idler_wl, theta)
        
        return (n_p / pump_wl) - (n_s / signal_wl) - (n_i / idler_wl)

    # 0 ile 90 derece (pi/2) arasında momentum farkını sıfır yapan açıyı bulur
    theta_pm_rad = brentq(momentum_mismatch, 0, np.pi/2)
    return np.degrees(theta_pm_rad)

if __name__ == "__main__":
    # --- AŞAMA 1: TEMEL İNDİS TESTİ (Başarıyla tamamlandı) ---
    # İlk olarak Sellmeier denklemlerimizin doğru çalışıp çalışmadığını test ettik.
    # print("--- BBO Temel Kırılma İndisleri ---")
    # print(f"405 nm n_o: {no(0.405):.5f}")
    # print(f"405 nm n_e: {ne(0.405):.5f}")
    # print(f"810 nm n_o: {no(0.810):.5f}")
    # print(f"810 nm n_e: {ne(0.810):.5f}")
    
    # --- AŞAMA 2: FAZ UYUMU AÇISI HESABI (Aktif Test) ---
    # Şimdi momentum korunumundan yola çıkarak Thorlabs'ın kesim açısını bulmaya çalışıyoruz.
    angle = type2_phase_matching_angle(0.405, 0.810, 0.810)
    print(f"Hesaplanan Type-II Faz Uyumu Acisi: {angle:.2f} derece")
    print(f"Thorlabs Katalog Acisi (Karsilastirma): 41.80 derece")