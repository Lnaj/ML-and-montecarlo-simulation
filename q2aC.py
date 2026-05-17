# ============================================================
# ACTU-F4002 - Modèles financiers en temps continu
# Biais de discrétisation - Lookback Put
# + Correction de Broadie-Glasserman-Kou (1997)
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
n_sim = 50000
SEED  = 42

# Constante de Broadie-Glasserman-Kou
# beta = -zeta(1/2) / sqrt(2*pi) ≈ 0.5826
BETA = 0.5826

# ---------------------------
# Fonctions
# ---------------------------
def black_scholes_lookback_put(S0, r, sigma, T):
    """Prix théorique exact (référence)."""
    c1    = (r + 0.5*sigma**2)*T / (sigma*np.sqrt(T))
    c2    = (r - 0.5*sigma**2)*T / (sigma*np.sqrt(T))
    A     = np.exp(-r*T) * norm.cdf(-c2)
    B     = norm.cdf(-c1)
    C     = (sigma**2/(2*r)) * (norm.cdf(c1) - np.exp(-r*T)*norm.cdf(-c2))
    return S0 * (A - B + C)

def monte_carlo_lookback_put_biased(S0, r, sigma, T, N, n_sim, seed=42):
    """
    MC Lookback Put SANS correction → souffre du biais de discrétisation.
    Maximum discret = max observé aux N points seulement.
    """
    np.random.seed(seed)
    dt    = T / N
    paths = np.zeros((n_sim, N+1))
    paths[:, 0] = S0

    for t in range(1, N+1):
        Z = np.random.standard_normal(n_sim)
        paths[:, t] = paths[:, t-1] * np.exp(
            (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        )

    # Maximum discret (biaisé)
    M_T      = np.max(paths, axis=1)
    S_T      = paths[:, -1]
    payoffs  = np.maximum(M_T - S_T, 0)
    discount = np.exp(-r*T)
    price    = discount * np.mean(payoffs)
    std_err  = discount * np.std(payoffs) / np.sqrt(n_sim)

    return price, std_err

def monte_carlo_lookback_put_corrected(S0, r, sigma, T, N, n_sim, seed=42):
    """
    MC Lookback Put AVEC correction BGK (1997).
    On ajuste le maximum discret vers le haut :
    M_corrected = M_discret * exp(beta * sigma * sqrt(dt))
    """
    np.random.seed(seed)
    dt    = T / N
    paths = np.zeros((n_sim, N+1))
    paths[:, 0] = S0

    for t in range(1, N+1):
        Z = np.random.standard_normal(n_sim)
        paths[:, t] = paths[:, t-1] * np.exp(
            (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        )

    # Maximum discret
    M_T_discrete = np.max(paths, axis=1)

    # Correction BGK : ajustement vers le haut
    correction   = np.exp(BETA * sigma * np.sqrt(dt))
    M_T_corrected = M_T_discrete * correction

    S_T      = paths[:, -1]
    payoffs  = np.maximum(M_T_corrected - S_T, 0)
    discount = np.exp(-r*T)
    price    = discount * np.mean(payoffs)
    std_err  = discount * np.std(payoffs) / np.sqrt(n_sim)

    return price, std_err

def theoretical_bias(S0, sigma, T, N):
    """
    Biais théorique approximatif selon BGK :
    Biais ≈ beta * sigma * S0 * sqrt(T/N)
    """
    return BETA * sigma * S0 * np.sqrt(T / N)

# ============================================================
# PARTIE 1 : Calcul du biais pour nos paramètres
# ============================================================
lp_bs    = black_scholes_lookback_put(S0, r, sigma, T)
N_values = [6, 13, 26, 52, 90, 180, 365]

print("=" * 75)
print("  Biais de discrétisation du Lookback Put")
print("  (Prix BS exact = référence)")
print("=" * 75)
print(f"  Prix BS exact : {lp_bs:.4f} €\n")
print(f"  {'N':<6} {'dt(j)':<8} {'MC biaisé':<12} {'MC corrigé':<12} "
      f"{'Biais réel':<12} {'Biais théo':<12} {'Biais %'}")
print("-" * 75)

biases_real  = []
biases_theo  = []
prices_bias  = []
prices_corr  = []

for N in N_values:
    p_bias, s_bias = monte_carlo_lookback_put_biased(
        S0, r, sigma, T, N, n_sim, seed=SEED
    )
    p_corr, s_corr = monte_carlo_lookback_put_corrected(
        S0, r, sigma, T, N, n_sim, seed=SEED
    )
    b_real = lp_bs - p_bias        # biais réel (sous-estimation)
    b_theo = theoretical_bias(S0, sigma, T, N)
    dt_days = (T/N)*365

    biases_real.append(b_real)
    biases_theo.append(b_theo)
    prices_bias.append(p_bias)
    prices_corr.append(p_corr)

    print(f"  {N:<6} {dt_days:<8.1f} {p_bias:<12.4f} {p_corr:<12.4f} "
          f"{b_real:<12.4f} {b_theo:<12.4f} {b_real/lp_bs*100:.2f}%")

print("=" * 75)

# Résumé pour N=26 (notre choix)
idx_26 = N_values.index(26)
print(f"\n  ⚠️  Avec N=26 (notre choix) :")
print(f"     Biais réel     = {biases_real[idx_26]:.4f} € "
      f"({biases_real[idx_26]/lp_bs*100:.2f}% du prix exact)")
print(f"     Biais théorique = {biases_theo[idx_26]:.4f} €")
print(f"     Prix biaisé     = {prices_bias[idx_26]:.4f} €")
print(f"     Prix corrigé    = {prices_corr[idx_26]:.4f} €")
print(f"     Prix BS exact   = {lp_bs:.4f} €")

# ============================================================
# PARTIE 2 : Comparaison finale N=26 vs N=180 vs correction
# ============================================================
print(f"\n{'=' * 65}")
print(f"  Comparaison des approches")
print(f"{'=' * 65}")
configs = [
    ("MC N=26 (sans correction)",  26,  False),
    ("MC N=26 (avec correction)",  26,  True),
    ("MC N=180 (sans correction)", 180, False),
    ("MC N=180 (avec correction)", 180, True),
]
print(f"  {'Méthode':<35} {'Prix':<10} {'Écart BS':<10} {'Écart %'}")
print("-" * 65)
for label, N, corrected in configs:
    if corrected:
        p, s = monte_carlo_lookback_put_corrected(
            S0, r, sigma, T, N, n_sim, seed=SEED
        )
    else:
        p, s = monte_carlo_lookback_put_biased(
            S0, r, sigma, T, N, n_sim, seed=SEED
        )
    print(f"  {label:<35} {p:<10.4f} {abs(p-lp_bs):<10.4f} "
          f"{abs(p-lp_bs)/lp_bs*100:.2f}%")
print(f"  {'BS exact (référence)':<35} {lp_bs:<10.4f} {'0.0000':<10} 0.00%")
print("=" * 65)

# ============================================================
# GRAPHIQUES
# ============================================================

# ---------------------------
# Graphique 1 : Biais réel et théorique en fonction de N
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(N_values, biases_real, 'o-',
              color='red', linewidth=2, markersize=8,
              label='Biais réel (BS - MC biaisé)')
axes[0].plot(N_values, biases_theo, 's--',
              color='darkorange', linewidth=2, markersize=8,
              label=r'Biais théorique BGK ($\beta\sigma S_0\sqrt{T/N}$)')
axes[0].axvline(x=26,  color='green',  linestyle=':',
                 linewidth=1.5, label='N=26 (notre choix)')
axes[0].axvline(x=180, color='purple', linestyle=':',
                 linewidth=1.5, label='N=180 (journalier)')
axes[0].set_xlabel("Nombre de pas N")
axes[0].set_ylabel("Biais (€)")
axes[0].set_title("Biais de discrétisation\nen fonction de N")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Graphique log-log pour voir la décroissance en 1/sqrt(N)
N_arr   = np.array(N_values, dtype=float)
b_arr   = np.array(biases_real)
scale   = biases_real[0] * np.sqrt(N_values[0])

axes[1].loglog(N_values, biases_real, 'o-',
                color='red', linewidth=2, markersize=8,
                label='Biais réel')
axes[1].loglog(N_values, scale/np.sqrt(N_arr), 'k--',
                linewidth=1.5,
                label=r'$C/\sqrt{N}$ (théorique)')
axes[1].axvline(x=26,  color='green',  linestyle=':',
                 linewidth=1.5, label='N=26')
axes[1].axvline(x=180, color='purple', linestyle=':',
                 linewidth=1.5, label='N=180')
axes[1].set_xlabel("N (échelle log)")
axes[1].set_ylabel("Biais (€) (échelle log)")
axes[1].set_title(r"Décroissance du biais en $1/\sqrt{N}$")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Biais de discrétisation du Lookback Put",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("biais_discretisation.png", dpi=150)
plt.show()



# ---------------------------
# Graphique 2 : Comparaison biaisé / corrigé / BS
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(N_values, prices_bias, 'o-',
              color='red', linewidth=2, markersize=8,
              label='MC sans correction (biaisé)')
axes[0].plot(N_values, prices_corr, 's-',
              color='green', linewidth=2, markersize=8,
              label='MC avec correction BGK')
axes[0].axhline(y=lp_bs, color='blue', linestyle='--',
                 linewidth=2, label=f'BS exact = {lp_bs:.4f}€')
axes[0].axvline(x=26,  color='gray', linestyle=':',
                 linewidth=1.5, label='N=26 (notre choix)')
axes[0].set_xlabel("Nombre de pas N")
axes[0].set_ylabel("Prix du Lookback Put (€)")
axes[0].set_title("Prix MC biaisé vs corrigé vs BS\nen fonction de N")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Erreur résiduelle après correction
errors_bias = [abs(p - lp_bs) for p in prices_bias]
errors_corr = [abs(p - lp_bs) for p in prices_corr]

axes[1].plot(N_values, errors_bias, 'o-',
              color='red', linewidth=2, markersize=8,
              label='|Erreur| sans correction')
axes[1].plot(N_values, errors_corr, 's-',
              color='green', linewidth=2, markersize=8,
              label='|Erreur| avec correction BGK')
axes[1].axvline(x=26,  color='gray', linestyle=':',
                 linewidth=1.5, label='N=26')
axes[1].set_xlabel("Nombre de pas N")
axes[1].set_ylabel("Erreur absolue vs BS (€)")
axes[1].set_title("Réduction de l'erreur\ngrâce à la correction BGK")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Effet de la correction BGK sur le biais de discrétisation",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("biais_correction_BGK.png", dpi=150)
plt.show()