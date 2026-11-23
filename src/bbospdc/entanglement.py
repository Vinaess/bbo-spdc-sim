import numpy as np
import matplotlib.pyplot as plt
from bbospdc.config import SIM_PARAMS


def temporal_walkoff_fs(L_mm, dispersion_fs_per_mm=200.0):
    """
    Type-II BBO kristalinde o ve e fotonları arasındaki ortalama
    boyuna zaman gecikmesini (fs) hesaplar (~200 fs/mm @ 810 nm).
    """
    return 0.5 * L_mm * dispersion_fs_per_mm


def coherence_time_fs(wavelength_nm, bandwidth_nm):
    """
    Gaussian spektral filtre için koherens süresini (FWHM, fs) hesaplar.
    """
    c_nm_per_fs = 299.792458  # nm / fs
    return (2.0 * np.log(2) / np.pi) * (wavelength_nm**2) / (c_nm_per_fs * bandwidth_nm)


def visibility_vs_delay(delay_fs, wavelength_nm=810.0, bandwidth_nm=10.0, v_max=0.985):
    """
    Zaman gecikmesine (delay_fs) bağlı olarak 45 derece bazındaki
    çift-foton girişim görünürlüğünü (Visibility) hesaplar.
    """
    tau_c = coherence_time_fs(wavelength_nm, bandwidth_nm)
    return v_max * np.exp(-2.0 * np.log(2) * (delay_fs / tau_c) ** 2)


def coincidence_probability(theta1_deg, theta2_deg, visibility=0.98, state="psi_minus"):
    """
    Werner benzeri gürültülü Bell durumu için iki polarizör açısına (theta1, theta2)
    bağlı çakışma (coincidence) olasılığını hesaplar.
    """
    t1 = np.radians(theta1_deg)
    t2 = np.radians(theta2_deg)
    if state == "psi_minus":
        # P(t1, t2) = 1/4 * [1 - V * cos(2*(t1 - t2))]
        return 0.25 * (1.0 - visibility * np.cos(2.0 * (t1 - t2)))
    elif state == "psi_plus":
        return 0.25 * (1.0 - visibility * np.cos(2.0 * (t1 + t2)))
    else:
        raise ValueError("Desteklenen durumlar: 'psi_minus' veya 'psi_plus'")


def plot_entanglement_analysis(compensated=True):
    """
    Faz 5:
    [Sol] H/V (0 deg) ve Diagonal (45 deg) bazlarında polarizasyon korelasyon saçakları.
    [Sağ] Zaman gecikmesine (walk-off) bağlı olarak 45 deg görünürlük eğrisi.
    """
    L_mm = SIM_PARAMS["crystal"]["thickness_mm"]
    wl_s = SIM_PARAMS["spdc"]["signal_wavelength_nm"]
    bw_nm = SIM_PARAMS["filters"]["signal_bandwidth_nm"]

    tau_c = coherence_time_fs(wl_s, bw_nm)
    tau_w = temporal_walkoff_fs(L_mm)

    # Telafi varsa gecikme ~0 fs, yoksa walk-off kadar (~200 fs ortalama)
    eff_delay = 0.0 if compensated else tau_w
    v_hv = 0.992
    v_45 = visibility_vs_delay(eff_delay, wl_s, bw_nm, v_max=0.985)

    print("=" * 60)
    print(f"Kristal Kalınlığı (L)      : {L_mm:.2f} mm")
    print(f"Filtre Bant Genişliği      : {bw_nm} nm (Koherens Süresi: {tau_c:.1f} fs)")
    print(f"Walk-off Telafisi (HWP+cc) : {'AKTİF' if compensated else 'YOK (Telafisiz)'}")
    print(f"Kalan Zaman Gecikmesi      : {eff_delay:.1f} fs")
    print(f"H/V Bazı Görünürlük (V_0)  : %{v_hv * 100:.1f}")
    print(f"45° Bazı Görünürlük (V_45) : %{v_45 * 100:.1f}")
    print(f"CHSH S-Değeri Tahmini      : {2 * np.sqrt(2) * v_45:.3f} (Klasik sınır: 2.000)")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # Grafik 1: Polarizasyon Korelasyon Saçakları
    theta2 = np.linspace(0, 180, 360)
    p_hv = coincidence_probability(0.0, theta2, visibility=v_hv)
    p_45 = coincidence_probability(45.0, theta2, visibility=v_45)

    ax1.plot(theta2, p_hv, label=f"H/V Basis (θ₁=0°, V={v_hv*100:.1f}%)", lw=2, color="tab:blue")
    ax1.plot(theta2, p_45, label=f"Diagonal Basis (θ₁=45°, V={v_45*100:.1f}%)", lw=2, linestyle="--", color="tab:red")
    ax1.set_title("Polarization Entanglement Fringes (|Ψ⁻⟩)")
    ax1.set_xlabel("Polarizer 2 Angle θ₂ (degrees)")
    ax1.set_ylabel("Normalized Coincidence Probability")
    ax1.set_xlim(0, 180)
    ax1.set_ylim(0, 0.55)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Grafik 2: Zaman Gecikmesine Bağlı Görünürlük (Walk-off Etkisi)
    delays = np.linspace(-300, 300, 400)
    vis_curve = visibility_vs_delay(delays, wl_s, bw_nm, v_max=0.985) * 100.0

    ax2.plot(delays, vis_curve, color="tab:purple", lw=2.5, label=f"Δλ = {bw_nm} nm Filter")
    ax2.axvline(0, color="green", linestyle=":", label="Compensated (Δt = 0 fs)")
    ax2.axvline(tau_w, color="orange", linestyle="--", label=f"Uncompensated 2mm BBO ({tau_w:.0f} fs)")
    ax2.axhline(70.7, color="red", linestyle="-.", alpha=0.7, label="Bell/CHSH Threshold (70.7%)")
    ax2.scatter([eff_delay], [v_45 * 100], color="black", zorder=5, s=70, label="Current Setup")

    ax2.set_title("45° Visibility vs. Temporal Walk-off Delay")
    ax2.set_xlabel("Relative Time Delay Δt (fs)")
    ax2.set_ylabel("Visibility V₄₅° (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()
    plt.show()