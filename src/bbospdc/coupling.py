import numpy as np
import matplotlib.pyplot as plt
from bbospdc.bbo import no, ne, ne_theta
from bbospdc.config import SIM_PARAMS


def walkoff_angle_rad(wavelength_um, theta_rad):
    """
    Tek eksenli negatif BBO kristali için olağanüstü (e) ışının
    enine yürüme açısını (walk-off angle, radyan) hesaplar.
    """
    n_o = no(wavelength_um)
    n_e = ne(wavelength_um)
    num = (n_o**2 - n_e**2) * np.sin(theta_rad) * np.cos(theta_rad)
    den = (n_o**2) * (np.cos(theta_rad)**2) + (n_e**2) * (np.sin(theta_rad)**2)
    return np.arctan(num / den)


def gaussian_coupling_efficiency(w_p_um, w_f_um, L_mm=None, theta_deg=None):
    """
    Type-II BBO (e -> o + e) için pompa ışın beli (w_p) ve kristal düzlemine
    görüntülenen fiber toplama modu (w_f) arasındaki uzaysal örtüşme integralini
    (walk-off + kırınım dahil) hesaplar.

    Döndürür:
        eta_herald: Koşullu kuplaj verimliliği (heralding / mode-matching efficiency)
        pair_yield: Bağıl çift toplama verimi (normalize edilmiş)
    """
    if L_mm is None:
        L_mm = SIM_PARAMS["crystal"]["thickness_mm"]
    if theta_deg is None:
        theta_deg = SIM_PARAMS["crystal"]["cut_angle_deg"]

    L_um = L_mm * 1000.0
    theta_rad = np.radians(theta_deg)

    wl_p = SIM_PARAMS["pump"]["wavelength_nm"] / 1000.0
    wl_s = SIM_PARAMS["spdc"]["signal_wavelength_nm"] / 1000.0
    wl_i = SIM_PARAMS["spdc"]["idler_wavelength_nm"] / 1000.0

    # Kırılma indisleri ve dalga sayıları (1/µm)
    n_p = ne_theta(wl_p, theta_rad)
    n_s = no(wl_s)
    n_i = ne_theta(wl_i, theta_rad)

    k_p = 2 * np.pi * n_p / wl_p
    k_s = 2 * np.pi * n_s / wl_s
    k_i = 2 * np.pi * n_i / wl_i

    # Walk-off açıları (pompa ve idler e-ray, sinyal o-ray)
    rho_p = walkoff_angle_rad(wl_p, theta_rad)
    rho_i = walkoff_angle_rad(wl_i, theta_rad)

    # Kristal boyunca z integrasyonu (-L/2 -> +L/2)
    n_z = 200
    z = np.linspace(-L_um / 2.0, L_um / 2.0, n_z)
    dz = z[1] - z[0]

    # Karmaşık Gaussian odaklama parametreleri (difraksiyon / Rayleigh etkisi)
    # q(z) = 1 / (w^2 * (1 + i * z / z_R))
    zR_p = 0.5 * k_p * (w_p_um**2)
    zR_s = 0.5 * k_s * (w_f_um**2)
    zR_i = 0.5 * k_i * (w_f_um**2)

    a_p = 1.0 / (w_p_um**2 * (1.0 + 1j * z / zR_p))
    a_s = 1.0 / (w_f_um**2 * (1.0 - 1j * z / zR_s))
    a_i = 1.0 / (w_f_um**2 * (1.0 - 1j * z / zR_i))

    # Walk-off merkez kaymaları (x ekseni boyunca)
    x_p = rho_p * z
    x_s = 0.0
    x_i = rho_i * z

    A = a_p + a_s + a_i
    x_bar = (a_p * x_p + a_s * x_s + a_i * x_i) / A

    # 3-Gaussian çarpımının x ve y üzerinden analitik integrali
    exponent = -(a_p * (x_p - x_bar)**2 + a_s * (x_s - x_bar)**2 + a_i * (x_i - x_bar)**2)
    overlap_z = (2.0 / (w_p_um * (w_f_um**2))) * (1.0 / A) * np.exp(exponent)

    # Çift (coincidence) genliğinin karesi
    amp_pair = np.sum(overlap_z) * dz
    prob_pair = (np.abs(amp_pair)**2) / (L_um**2)

    # Tekil (single) foton toplama olasılığı (walk-off ve odaklama normalizasyonu)
    # İdeal ince kristal limitinde (L -> 0, w_p = w_f) oran maksimuma yaklaşır
    w_eff_p = w_p_um * np.sqrt(1.0 + (L_um / (2.0 * zR_p))**2)
    w_eff_f = w_f_um * np.sqrt(1.0 + (L_um / (2.0 * zR_s))**2)
    single_norm = 1.0 / (w_eff_p**2 + w_eff_f**2)

    # Normalize edilmiş kuplaj verimliliği (0 - 1 arası)
    eta_herald = np.clip(prob_pair / (single_norm + 1e-18), 0.0, 1.0)
    return float(eta_herald), float(prob_pair)
def plot_coupling_heatmap():
    """
    Pompa (w_p) ve Fiber (w_f) ışın bellerine göre 2D Fiber Kuplaj Verimliliği
    ve Bağıl Çift Verimi (Brightness) haritalarını yan yana çizer.
    """
    L_mm = SIM_PARAMS["crystal"]["thickness_mm"]
    theta_rad = np.radians(SIM_PARAMS["crystal"]["cut_angle_deg"])
    wl_p = SIM_PARAMS["pump"]["wavelength_nm"] / 1000.0
    wl_i = SIM_PARAMS["spdc"]["idler_wavelength_nm"] / 1000.0

    rho_p_deg = np.degrees(walkoff_angle_rad(wl_p, theta_rad))
    rho_i_rad = walkoff_angle_rad(wl_i, theta_rad)
    w_walkoff_um = rho_i_rad * (L_mm * 1000.0)

    wp_vals = np.linspace(20, 350, 60)
    wf_vals = np.linspace(20, 350, 60)

    eta_grid = np.zeros((len(wf_vals), len(wp_vals)))
    yield_grid = np.zeros((len(wf_vals), len(wp_vals)))

    for i, wf in enumerate(wf_vals):
        for j, wp in enumerate(wp_vals):
            eta_grid[i, j], yield_grid[i, j] = gaussian_coupling_efficiency(wp, wf)

    # Bağıl parlaklığı %0 - %100 arasına normalize et
    yield_norm = (yield_grid / np.max(yield_grid)) * 100.0

    # Optimum noktalar
    max_eta_idx = np.unravel_index(np.argmax(eta_grid), eta_grid.shape)
    opt_wf_eta, opt_wp_eta = wf_vals[max_eta_idx[0]], wp_vals[max_eta_idx[1]]

    max_y_idx = np.unravel_index(np.argmax(yield_grid), yield_grid.shape)
    opt_wf_y, opt_wp_y = wf_vals[max_y_idx[0]], wp_vals[max_y_idx[1]]

    print("=" * 60)
    print(f"BBO Walk-off Açısı (405 nm): {rho_p_deg:.2f} derece | w_w = {w_walkoff_um:.1f} µm")
    print(f"[1] Maks. Heralding Verimi : %{eta_grid[max_eta_idx]*100:.1f}  (w_p={opt_wp_eta:.1f} µm, w_f={opt_wf_eta:.1f} µm)")
    print(f"[2] Maks. Çift Verimi (100%): w_p={opt_wp_y:.1f} µm, w_f={opt_wf_y:.1f} µm")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Grafik 1: Heralding / Mode-Matching Efficiency
    im1 = ax1.imshow(
        eta_grid * 100,
        origin="lower",
        aspect="auto",
        extent=[wp_vals[0], wp_vals[-1], wf_vals[0], wf_vals[-1]],
        cmap="viridis",
    )
    fig.colorbar(im1, ax=ax1, label="Heralding / Mode-Matching Efficiency (%)")
    ax1.scatter([opt_wp_eta], [opt_wf_eta], color="red", s=90, marker="*",
                label=f"Optimum ({opt_wp_eta:.0f} µm, {opt_wf_eta:.0f} µm)")
    ax1.axvline(w_walkoff_um, color="white", linestyle="--", alpha=0.6, label=f"w_w = {w_walkoff_um:.0f} µm")
    ax1.axhline(w_walkoff_um, color="white", linestyle="--", alpha=0.6)
    ax1.set_title(f"Mode-Matching Efficiency (L = {L_mm} mm)")
    ax1.set_xlabel("Pump Beam Waist w_p (µm)")
    ax1.set_ylabel("Collection Mode Waist w_f (µm)")
    ax1.legend(loc="upper right")

    # Grafik 2: Relative Pair Yield (Brightness)
    im2 = ax2.imshow(
        yield_norm,
        origin="lower",
        aspect="auto",
        extent=[wp_vals[0], wp_vals[-1], wf_vals[0], wf_vals[-1]],
        cmap="inferno",
    )
    fig.colorbar(im2, ax=ax2, label="Relative Pair Yield / Brightness (%)")
    ax2.scatter([opt_wp_y], [opt_wf_y], color="cyan", s=90, marker="*",
                label=f"Max Yield ({opt_wp_y:.0f} µm, {opt_wf_y:.0f} µm)")
    ax2.axvline(w_walkoff_um, color="white", linestyle="--", alpha=0.6, label=f"w_w = {w_walkoff_um:.0f} µm")
    ax2.axhline(w_walkoff_um, color="white", linestyle="--", alpha=0.6)
    ax2.set_title(f"Normalized Pair Yield / Brightness (L = {L_mm} mm)")
    ax2.set_xlabel("Pump Beam Waist w_p (µm)")
    ax2.set_ylabel("Collection Mode Waist w_f (µm)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
