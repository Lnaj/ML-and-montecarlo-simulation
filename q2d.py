# ============================================================
# ACTU-F4002 - Modèles financiers en temps continu
# Question 2d : Prix théorique Lookback Put - Formule BS (0.1)
#               + Comparaison avec Monte Carlo
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

# ---------------------------
# Fonctions
# ---------------------------
def monte_carlo_lookback_put(S0, r, sigma, T, N, n_sim, seed=42):
    np.random.seed(seed)
    dt    = T / N
    paths = np.zeros((n_sim, N+1))
    paths[:, 0] = S0

    for t in range(1, N+1):
        Z = np.random.standard_normal(n_sim)
        paths[:, t] = paths[:, t-1] * np.exp(
            (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        )

    M_T      = np.max(paths, axis=1)
    S_T      = paths[:, -1]
    payoffs  = np.maximum(M_T - S_T, 0)
    discount = np.exp(-r*T)
    price    = discount * np.mean(payoffs)
    std_err  = discount * np.std(payoffs) / np.sqrt(n_sim)

    return price, std_err

def black_scholes_lookback_put(S0, r, sigma, T):
    """
    Prix théorique Lookback Put via formule Black-Scholes (0.1)

    LP(0,T) = S(0) * [e^{-rT}*Phi(-c2) - Phi(-c1)
              + sigma²/(2r) * (Phi(c1) - e^{-rT}*Phi(-c2))]

    avec :
    c1 = (r + sigma²/2)*T / (sigma*sqrt(T))
    c2 = (r - sigma²/2)*T / (sigma*sqrt(T))
    """
    c1 = (r + 0.5*sigma**2)*T / (sigma*np.sqrt(T))
    c2 = (r - 0.5*sigma**2)*T / (sigma*np.sqrt(T))

    A = np.exp(-r*T) * norm.cdf(-c2)
    B = norm.cdf(-c1)
    C = (sigma**2 / (2*r)) * (norm.cdf(c1) - np.exp(-r*T)*norm.cdf(-c2))

    price = S0 * (A - B + C)

    return price, c1, c2, A, B, C

# ============================================================
# CALCUL DU PRIX THEORIQUE
# ============================================================
lp_bs, c1, c2, A, B, C = black_scholes_lookback_put(S0, r, sigma, T)
lp_mc, lp_std           = monte_carlo_lookback_put(
    S0, r, sigma, T, N, n_sim, seed=SEED
)

IC_low  = lp_mc - 1.96 * lp_std
IC_high = lp_mc + 1.96 * lp_std

print("=" * 60)
print("  Question 2d : Prix théorique Lookback Put (BS)")
print("=" * 60)
print(f"\n  Paramètres : S0={S0}, r={r}, sigma={sigma}, T={T}")

print(f"\n  --- Calcul détaillé des paramètres ---")
print(f"  c1                      : {c1:.6f}")
print(f"  c2                      : {c2:.6f}")
print(f"  Phi(c1)                 : {norm.cdf(c1):.6f}")
print(f"  Phi(-c1)                : {norm.cdf(-c1):.6f}")
print(f"  Phi(-c2)                : {norm.cdf(-c2):.6f}")
print(f"  e^(-rT)                 : {np.exp(-r*T):.6f}")
print(f"  sigma²/(2r)             : {sigma**2/(2*r):.6f}")

print(f"\n  --- Décomposition du prix ---")
print(f"  Terme A = e^(-rT)*Phi(-c2)            : {A:.6f}")
print(f"  Terme B = Phi(-c1)                    : {B:.6f}")
print(f"  Terme C = sigma²/(2r)*(Phi(c1)        ")
print(f"            - e^(-rT)*Phi(-c2))         : {C:.6f}")
print(f"  S0 * (A - B + C)                      : {S0*(A-B+C):.6f}")

print(f"\n  --- Résultats ---")
print(f"  Prix BS Lookback Put    : {lp_bs:.4f} €")
print(f"  Prix MC Lookback Put    : {lp_mc:.4f} €")
print(f"  Erreur standard MC      : {lp_std:.4f} €")
print(f"  IC 95% MC               : [{IC_low:.4f}, {IC_high:.4f}]")
print(f"  Différence absolue      : {abs(lp_bs-lp_mc):.4f} €")
print(f"  Différence relative     : {abs(lp_bs-lp_mc)/lp_bs*100:.2f} %")
print(f"  BS dans IC 95% MC ?     : "
      f"{'✓ OUI' if IC_low <= lp_bs <= IC_high else '✗ NON'}")
print("=" * 60)

# ============================================================
# GRAPHIQUES
# ============================================================

# ---------------------------
# Graphique 1 : Prix BS vs MC en fonction de S0
# ---------------------------
S0_range    = np.linspace(70, 140, 100)
bs_prices   = [black_scholes_lookback_put(s, r, sigma, T)[0] for s in S0_range]
mc_prices   = [monte_carlo_lookback_put(s, r, sigma, T, N, n_sim)[0]
               for s in S0_range]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(S0_range, bs_prices, 'b-',  linewidth=2,
              label='Black-Scholes (exact)')
axes[0].plot(S0_range, mc_prices, 'r--', linewidth=1.5,
              label='Monte Carlo')
axes[0].axvline(x=S0, color='green', linestyle=':',
                 linewidth=1.5, label=f'S0={S0}')
axes[0].set_xlabel("Valeur initiale S(0) (€)")
axes[0].set_ylabel("Prix du Lookback Put (€)")
axes[0].set_title("Prix Lookback Put\nBlack-Scholes vs Monte Carlo")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ---------------------------
# Graphique 2 : Prix BS vs MC en fonction de sigma
# ---------------------------
sig_range = np.linspace(0.05, 0.50, 60)
bs_sig    = [black_scholes_lookback_put(S0, r, s, T)[0] for s in sig_range]
mc_sig    = [monte_carlo_lookback_put(S0, r, s, T, N, n_sim)[0]
             for s in sig_range]

axes[1].plot(sig_range*100, bs_sig, 'b-',  linewidth=2,
              label='Black-Scholes (exact)')
axes[1].plot(sig_range*100, mc_sig, 'r--', linewidth=1.5,
              label='Monte Carlo')
axes[1].axvline(x=sigma*100, color='green', linestyle=':',
                 linewidth=1.5, label=f'σ={sigma*100}%')
axes[1].set_xlabel("Volatilité σ (%)")
axes[1].set_ylabel("Prix du Lookback Put (€)")
axes[1].set_title("Sensibilité à σ\nBlack-Scholes vs Monte Carlo")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2d : Prix théorique BS vs Monte Carlo",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2d_BS_vs_MC.png", dpi=150)
plt.show()

# ---------------------------
# Graphique 3 : Décomposition de la formule BS
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Termes A, B, C en fonction de S0
A_vals = [black_scholes_lookback_put(s, r, sigma, T)[3] * s for s in S0_range]
B_vals = [black_scholes_lookback_put(s, r, sigma, T)[4] * s for s in S0_range]
C_vals = [black_scholes_lookback_put(s, r, sigma, T)[5] * s for s in S0_range]

axes[0].plot(S0_range, A_vals, 'b-',  linewidth=2, label='S0 × A')
axes[0].plot(S0_range, B_vals, 'r-',  linewidth=2, label='S0 × B')
axes[0].plot(S0_range, C_vals, 'g-',  linewidth=2, label='S0 × C')
axes[0].plot(S0_range, bs_prices, 'k--', linewidth=2,
              label='Prix total = S0×(A-B+C)')
axes[0].axvline(x=S0, color='gray', linestyle=':',
                 linewidth=1.5, label=f'S0={S0}')
axes[0].set_xlabel("Valeur initiale S(0) (€)")
axes[0].set_ylabel("Valeur (€)")
axes[0].set_title("Décomposition de la formule BS\nen fonction de S(0)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Termes A, B, C en fonction de sigma
A_sig = [black_scholes_lookback_put(S0, r, s, T)[3] * S0 for s in sig_range]
B_sig = [black_scholes_lookback_put(S0, r, s, T)[4] * S0 for s in sig_range]
C_sig = [black_scholes_lookback_put(S0, r, s, T)[5] * S0 for s in sig_range]

axes[1].plot(sig_range*100, A_sig, 'b-',  linewidth=2, label='S0 × A')
axes[1].plot(sig_range*100, B_sig, 'r-',  linewidth=2, label='S0 × B')
axes[1].plot(sig_range*100, C_sig, 'g-',  linewidth=2, label='S0 × C')
axes[1].plot(sig_range*100, bs_sig, 'k--', linewidth=2,
              label='Prix total = S0×(A-B+C)')
axes[1].axvline(x=sigma*100, color='gray', linestyle=':',
                 linewidth=1.5, label=f'σ={sigma*100}%')
axes[1].set_xlabel("Volatilité σ (%)")
axes[1].set_ylabel("Valeur (€)")
axes[1].set_title("Décomposition de la formule BS\nen fonction de σ")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2d : Décomposition de la formule BS du Lookback Put",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2d_decomposition.png", dpi=150)
plt.show()