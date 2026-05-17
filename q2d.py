# ============================================================
# ACTU-F4002 - Modèles financiers en temps continu
# Question 2d : Prix théorique Lookback Put - Formule BS
#               + biais de discrétisation
#               + correction de Broadie-Glasserman-Kou (1997)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------
# Paramètres globaux
# ---------------------------
r     = 0.02
S0    = 90
sigma = 0.20
T     = 0.5
N     = 26
n_sim = 50000
SEED  = 42

# Constante de Broadie-Glasserman-Kou
# beta = -zeta(1/2) / sqrt(2*pi) ≈ 0.5826
BETA = 0.5826

# ---------------------------
# Fonctions
# ---------------------------
def black_scholes_lookback_put(S0, r, sigma, T):
    """
    Prix théorique exact du Lookback Put à strike flottant
    dans le modèle de Black-Scholes.
    """
    c1 = (r + 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
    c2 = (r - 0.5 * sigma**2) * T / (sigma * np.sqrt(T))

    A = np.exp(-r * T) * norm.cdf(-c2)
    B = norm.cdf(-c1)
    C = (sigma**2 / (2 * r)) * (norm.cdf(c1) - np.exp(-r * T) * norm.cdf(-c2))

    price = S0 * (A - B + C)
    return price, c1, c2, A, B, C


def monte_carlo_lookback_put_biased(S0, r, sigma, T, N, n_sim, seed=42):
    """
    Monte Carlo Lookback Put SANS correction :
    maximum discret observé sur la grille seulement.
    """
    np.random.seed(seed)
    dt = T / N

    paths = np.zeros((n_sim, N + 1))
    paths[:, 0] = S0

    for t in range(1, N + 1):
        Z = np.random.standard_normal(n_sim)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    M_T = np.max(paths, axis=1)
    S_T = paths[:, -1]

    payoffs = M_T - S_T
    discount = np.exp(-r * T)

    discounted_payoffs = discount * payoffs
    price = np.mean(discounted_payoffs)
    std_err = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sim)

    return price, std_err


def monte_carlo_lookback_put_corrected(S0, r, sigma, T, N, n_sim, seed=42):
    """
    Monte Carlo Lookback Put AVEC correction BGK (1997).
    Le maximum discret est corrigé vers le haut :
        M_corrected = M_discrete * exp(beta * sigma * sqrt(dt))
    """
    np.random.seed(seed)
    dt = T / N

    paths = np.zeros((n_sim, N + 1))
    paths[:, 0] = S0

    for t in range(1, N + 1):
        Z = np.random.standard_normal(n_sim)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    M_T_discrete = np.max(paths, axis=1)
    correction_factor = np.exp(BETA * sigma * np.sqrt(dt))
    M_T_corrected = M_T_discrete * correction_factor

    S_T = paths[:, -1]
    payoffs = M_T_corrected - S_T
    discount = np.exp(-r * T)

    discounted_payoffs = discount * payoffs
    price = np.mean(discounted_payoffs)
    std_err = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sim)

    return price, std_err


def theoretical_bias(S0, sigma, T, N):
    """
    Approximation théorique du biais BGK :
        biais ≈ beta * sigma * S0 * sqrt(T/N)
    """
    return BETA * sigma * S0 * np.sqrt(T / N)


# ============================================================
# CALCULS PRINCIPAUX
# ============================================================
lp_bs, c1, c2, A, B, C = black_scholes_lookback_put(S0, r, sigma, T)

# MC sans correction pour N=26
lp_mc_bias, lp_std_bias = monte_carlo_lookback_put_biased(
    S0, r, sigma, T, N, n_sim, seed=SEED
)
IC_low_bias  = lp_mc_bias - 1.96 * lp_std_bias
IC_high_bias = lp_mc_bias + 1.96 * lp_std_bias

# MC avec correction BGK pour N=26
lp_mc_corr, lp_std_corr = monte_carlo_lookback_put_corrected(
    S0, r, sigma, T, N, n_sim, seed=SEED
)
IC_low_corr  = lp_mc_corr - 1.96 * lp_std_corr
IC_high_corr = lp_mc_corr + 1.96 * lp_std_corr

# Biais théorique / réel pour N=26
bias_real = lp_bs - lp_mc_bias
bias_theo = theoretical_bias(S0, sigma, T, N)

print("=" * 75)
print("  Question 2d : Prix théorique du Lookback Put")
print("  + biais de discrétisation + correction BGK (1997)")
print("=" * 75)

print(f"\n  Paramètres : S0={S0}, r={r}, sigma={sigma}, T={T}, N={N}, n_sim={n_sim}")

print(f"\n  --- Prix théorique Black-Scholes ---")
print(f"  c1                      : {c1:.6f}")
print(f"  c2                      : {c2:.6f}")
print(f"  Phi(c1)                 : {norm.cdf(c1):.6f}")
print(f"  Phi(-c1)                : {norm.cdf(-c1):.6f}")
print(f"  Phi(-c2)                : {norm.cdf(-c2):.6f}")
print(f"  e^(-rT)                 : {np.exp(-r*T):.6f}")
print(f"  sigma²/(2r)             : {sigma**2/(2*r):.6f}")
print(f"  Prix BS exact           : {lp_bs:.4f} €")

print(f"\n  --- Monte Carlo SANS correction ---")
print(f"  Prix MC biaisé          : {lp_mc_bias:.4f} €")
print(f"  Erreur standard         : {lp_std_bias:.4f} €")
print(f"  IC 95%                  : [{IC_low_bias:.4f}, {IC_high_bias:.4f}]")
print(f"  Écart absolu à BS       : {abs(lp_bs - lp_mc_bias):.4f} €")
print(f"  Écart relatif à BS      : {abs(lp_bs - lp_mc_bias)/lp_bs*100:.2f} %")
print(f"  BS dans IC 95% ?        : {'✓ OUI' if IC_low_bias <= lp_bs <= IC_high_bias else '✗ NON'}")

print(f"\n  --- Monte Carlo AVEC correction BGK ---")
print(f"  Prix MC corrigé         : {lp_mc_corr:.4f} €")
print(f"  Erreur standard         : {lp_std_corr:.4f} €")
print(f"  IC 95%                  : [{IC_low_corr:.4f}, {IC_high_corr:.4f}]")
print(f"  Écart absolu à BS       : {abs(lp_bs - lp_mc_corr):.4f} €")
print(f"  Écart relatif à BS      : {abs(lp_bs - lp_mc_corr)/lp_bs*100:.2f} %")
print(f"  BS dans IC 95% ?        : {'✓ OUI' if IC_low_corr <= lp_bs <= IC_high_corr else '✗ NON'}")

print(f"\n  --- Analyse du biais pour N={N} ---")
print(f"  Biais réel (BS - MC biaisé)      : {bias_real:.4f} €")
print(f"  Biais théorique BGK approx.      : {bias_theo:.4f} €")
print(f"  Réduction grâce à la correction  : {abs(lp_bs-lp_mc_bias)-abs(lp_bs-lp_mc_corr):.4f} €")

# Comparaison complémentaire N=26 vs N=180
p26_bias, s26_bias = monte_carlo_lookback_put_biased(S0, r, sigma, T, 26, n_sim, seed=SEED)
p26_corr, s26_corr = monte_carlo_lookback_put_corrected(S0, r, sigma, T, 26, n_sim, seed=SEED)
p180_bias, s180_bias = monte_carlo_lookback_put_biased(S0, r, sigma, T, 180, n_sim, seed=SEED)
p180_corr, s180_corr = monte_carlo_lookback_put_corrected(S0, r, sigma, T, 180, n_sim, seed=SEED)

print(f"\n{'=' * 70}")
print("  Comparaison des approches : N=26 vs N=180")
print(f"{'=' * 70}")
print(f"  {'Méthode':<35} {'Prix':<10} {'Écart BS':<10} {'Écart %'}")
print("-" * 70)
print(f"  {'MC N=26 (sans correction)':<35} {p26_bias:<10.4f} {abs(p26_bias-lp_bs):<10.4f} {abs(p26_bias-lp_bs)/lp_bs*100:.2f}%")
print(f"  {'MC N=26 (avec correction)':<35} {p26_corr:<10.4f} {abs(p26_corr-lp_bs):<10.4f} {abs(p26_corr-lp_bs)/lp_bs*100:.2f}%")
print(f"  {'MC N=180 (sans correction)':<35} {p180_bias:<10.4f} {abs(p180_bias-lp_bs):<10.4f} {abs(p180_bias-lp_bs)/lp_bs*100:.2f}%")
print(f"  {'MC N=180 (avec correction)':<35} {p180_corr:<10.4f} {abs(p180_corr-lp_bs):<10.4f} {abs(p180_corr-lp_bs)/lp_bs*100:.2f}%")
print(f"  {'BS exact (référence)':<35} {lp_bs:<10.4f} {'0.0000':<10} 0.00%")
print("=" * 70)

# ============================================================
# DÉTAIL POUR N = 180 AVEC CORRECTION BGK
# ============================================================
IC_low_180_corr  = p180_corr - 1.96 * s180_corr
IC_high_180_corr = p180_corr + 1.96 * s180_corr

print(f"\n  --- Monte Carlo AVEC correction BGK (N=180) ---")
print(f"  Prix MC corrigé         : {p180_corr:.4f} €")
print(f"  Erreur standard         : {s180_corr:.4f} €")
print(f"  IC 95%                  : [{IC_low_180_corr:.4f}, {IC_high_180_corr:.4f}]")
print(f"  Écart absolu à BS       : {abs(lp_bs - p180_corr):.4f} €")
print(f"  Écart relatif à BS      : {abs(lp_bs - p180_corr)/lp_bs*100:.2f} %")
print(f"  BS dans IC 95% ?        : {'✓ OUI' if IC_low_180_corr <= lp_bs <= IC_high_180_corr else '✗ NON'}")

# ============================================================
# GRAPHIQUES
# ============================================================

# ---------------------------
# Graphique 1 : Biais réel et théorique en fonction de N
# ---------------------------
N_values = [6, 13, 26, 52, 90, 180, 365]
biases_real = []
biases_theo = []
prices_bias = []
prices_corr = []

for N_val in N_values:
    p_bias, _ = monte_carlo_lookback_put_biased(S0, r, sigma, T, N_val, n_sim, seed=SEED)
    p_corr, _ = monte_carlo_lookback_put_corrected(S0, r, sigma, T, N_val, n_sim, seed=SEED)

    biases_real.append(lp_bs - p_bias)
    biases_theo.append(theoretical_bias(S0, sigma, T, N_val))
    prices_bias.append(p_bias)
    prices_corr.append(p_corr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(N_values, biases_real, 'o-',
             color='red', linewidth=2, markersize=8,
             label='Biais réel (BS - MC biaisé)')
axes[0].plot(N_values, biases_theo, 's--',
             color='darkorange', linewidth=2, markersize=8,
             label=r'Biais théorique BGK ($\beta \sigma S_0 \sqrt{T/N}$)')
axes[0].axvline(x=26, color='green', linestyle=':',
                linewidth=1.5, label='N=26')
axes[0].axvline(x=180, color='purple', linestyle=':',
                linewidth=1.5, label='N=180')
axes[0].set_xlabel("Nombre de pas N")
axes[0].set_ylabel("Biais (€)")
axes[0].set_title("Biais de discrétisation\nen fonction de N")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

N_arr = np.array(N_values, dtype=float)
scale = biases_real[0] * np.sqrt(N_values[0])

axes[1].loglog(N_values, biases_real, 'o-',
               color='red', linewidth=2, markersize=8,
               label='Biais réel')
axes[1].loglog(N_values, scale / np.sqrt(N_arr), 'k--',
               linewidth=1.5, label=r'$C/\sqrt{N}$')
axes[1].axvline(x=26, color='green', linestyle=':',
                linewidth=1.5, label='N=26')
axes[1].axvline(x=180, color='purple', linestyle=':',
                linewidth=1.5, label='N=180')
axes[1].set_xlabel("N (échelle log)")
axes[1].set_ylabel("Biais (€) (échelle log)")
axes[1].set_title(r"Décroissance du biais en $1/\sqrt{N}$")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2d : Biais de discrétisation du Lookback Put",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2d_biais_discretisation.png", dpi=150)
plt.show()

# ---------------------------
# Graphique 2 : Comparaison biaisé / corrigé / BS en fonction de N
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(N_values, prices_bias, 'o-',
             color='red', linewidth=2, markersize=8,
             label='MC sans correction')
axes[0].plot(N_values, prices_corr, 's-',
             color='green', linewidth=2, markersize=8,
             label='MC avec correction BGK')
axes[0].axhline(y=lp_bs, color='blue', linestyle='--',
                linewidth=2, label=f'BS exact = {lp_bs:.4f} €')
axes[0].axvline(x=180, color='gray', linestyle=':',
                linewidth=1.5, label='N=180')
axes[0].set_xlabel("Nombre de pas N")
axes[0].set_ylabel("Prix du Lookback Put (€)")
axes[0].set_title("Prix MC biaisé vs corrigé vs BS\nen fonction de N")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

errors_bias = [abs(p - lp_bs) for p in prices_bias]
errors_corr = [abs(p - lp_bs) for p in prices_corr]

axes[1].plot(N_values, errors_bias, 'o-',
             color='red', linewidth=2, markersize=8,
             label='|Erreur| sans correction')
axes[1].plot(N_values, errors_corr, 's-',
             color='green', linewidth=2, markersize=8,
             label='|Erreur| avec correction BGK')
axes[1].axvline(x=180, color='gray', linestyle=':',
                linewidth=1.5, label='N=180')
axes[1].set_xlabel("Nombre de pas N")
axes[1].set_ylabel("Erreur absolue vs BS (€)")
axes[1].set_title("Réduction de l'erreur\ngrâce à la correction BGK")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2d : Effet de la correction BGK",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2d_correction_BGK.png", dpi=150)
plt.show()

# ---------------------------
# Graphique 3 : BS vs MC biaisé vs MC corrigé
#               selon S0 et selon sigma
# ---------------------------
S0_range  = np.linspace(70, 140, 100)
sig_range = np.linspace(0.05, 0.50, 60)

# En fonction de S0
bs_s0 = [black_scholes_lookback_put(s, r, sigma, T)[0] for s in S0_range]
mc_s0_bias = [monte_carlo_lookback_put_biased(s, r, sigma, T, 26, n_sim, seed=SEED)[0]
              for s in S0_range]
mc_s0_corr = [monte_carlo_lookback_put_corrected(s, r, sigma, T, 26, n_sim, seed=SEED)[0]
              for s in S0_range]

# En fonction de sigma
bs_sig = [black_scholes_lookback_put(S0, r, s, T)[0] for s in sig_range]
mc_sig_bias = [monte_carlo_lookback_put_biased(S0, r, s, T, 26, n_sim, seed=SEED)[0]
               for s in sig_range]
mc_sig_corr = [monte_carlo_lookback_put_corrected(S0, r, s, T, 26, n_sim, seed=SEED)[0]
               for s in sig_range]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Panneau gauche : selon S0
axes[0].plot(S0_range, bs_s0, 'b-', linewidth=2, label='BS exact')
axes[0].plot(S0_range, mc_s0_bias, 'r--', linewidth=1.8, label='MC sans correction')
axes[0].plot(S0_range, mc_s0_corr, 'g-.', linewidth=1.8, label='MC avec correction BGK')
axes[0].axvline(x=S0, color='gray', linestyle=':', linewidth=1.5,
                label=f'S0 = {S0}')
axes[0].set_xlabel("Valeur initiale S(0) (€)")
axes[0].set_ylabel("Prix du Lookback Put (€)")
axes[0].set_title("Comparaison selon S(0)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Panneau droit : selon sigma
axes[1].plot(sig_range * 100, bs_sig, 'b-', linewidth=2, label='BS exact')
axes[1].plot(sig_range * 100, mc_sig_bias, 'r--', linewidth=1.8, label='MC sans correction')
axes[1].plot(sig_range * 100, mc_sig_corr, 'g-.', linewidth=1.8, label='MC avec correction BGK')
axes[1].axvline(x=sigma * 100, color='gray', linestyle=':', linewidth=1.5,
                label=f'σ = {sigma*100:.0f}%')
axes[1].set_xlabel("Volatilité σ (%)")
axes[1].set_ylabel("Prix du Lookback Put (€)")
axes[1].set_title("Comparaison selon σ")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2d : BS vs MC biaisé vs MC corrigé",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2d_S0_sigma_comparaison.png", dpi=150)
plt.show()