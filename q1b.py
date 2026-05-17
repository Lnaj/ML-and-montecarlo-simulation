# ============================================================
# ACTU-F4002 - Modèles financiers en temps continu
# Question 1b : Prix d'un European Put - Formule Black-Scholes
#               + Comparaison avec Monte Carlo (Q1a)
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
K     = 110
T     = 0.5
N     = 26
n_sim = 50000

# ---------------------------
# Fonction Black-Scholes - European Put
# ---------------------------
def black_scholes_put(S0, K, r, sigma, T):
    """
    Calcule le prix exact d'un European Put via Black-Scholes.
    
    Paramètres :
    ------------
    S0    : valeur initiale du sous-jacent
    K     : strike
    r     : taux d'intérêt
    sigma : volatilité
    T     : maturité (en années)
    
    Retourne :
    ----------
    price : prix théorique du put
    d1    : paramètre d1
    d2    : paramètre d2
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    return price, d1, d2

# ---------------------------
# Fonction Monte Carlo (reprise de Q1a)
# ---------------------------
def monte_carlo_european_put(S0, K, r, sigma, T, N, n_sim, seed=42):
    np.random.seed(seed)
    dt    = T / N
    paths = np.zeros((n_sim, N + 1))
    paths[:, 0] = S0
    
    for t in range(1, N + 1):
        Z = np.random.standard_normal(n_sim)
        paths[:, t] = paths[:, t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )
    
    S_T     = paths[:, -1]
    payoff  = np.maximum(K - S_T, 0)
    discount = np.exp(-r * T)
    price   = discount * np.mean(payoff)
    std_err = discount * np.std(payoff, ddof= 1) / np.sqrt(n_sim)
    
    return price, std_err

# ---------------------------
# Calcul des prix
# ---------------------------
bs_price, d1, d2 = black_scholes_put(S0, K, r, sigma, T)
mc_price, mc_std = monte_carlo_european_put(S0, K, r, sigma, T, N, n_sim)

print("=" * 55)
print("  Question 1b : European Put - Black-Scholes vs Monte Carlo")
print("=" * 55)
print(f"\n  Paramètres : S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
print(f"\n  --- Formule Black-Scholes ---")
print(f"  d1                    : {d1:.4f}")
print(f"  d2                    : {d2:.4f}")
print(f"  N(-d1)                : {norm.cdf(-d1):.4f}")
print(f"  N(-d2)                : {norm.cdf(-d2):.4f}")
print(f"  Prix BS               : {bs_price:.4f} €")
print(f"\n  --- Monte Carlo (N={N} pas, {n_sim} simulations) ---")
print(f"  Prix MC               : {mc_price:.4f} €")
print(f"  Erreur standard       : {mc_std:.4f} €")
print(f"  IC 95%                : [{mc_price-1.96*mc_std:.4f}, {mc_price+1.96*mc_std:.4f}]")
print(f"\n  --- Comparaison ---")
print(f"  Différence absolue    : {abs(bs_price - mc_price):.4f} €")
print(f"  Différence relative   : {abs(bs_price - mc_price)/bs_price*100:.2f} %")
print(f"  BS dans IC 95% MC ?   : "
      f"{'✓ OUI' if mc_price-1.96*mc_std <= bs_price <= mc_price+1.96*mc_std else '✗ NON'}")
print("=" * 55)

# ---------------------------
# Graphique 1 : Comparaison pour différentes valeurs de S0
# ---------------------------
S0_range  = np.linspace(70, 140, 100)
bs_prices = [black_scholes_put(s, K, r, sigma, T)[0] for s in S0_range]
mc_prices = [monte_carlo_european_put(s, K, r, sigma, T, N, n_sim)[0] for s in S0_range]

plt.figure(figsize=(10, 5))
plt.plot(S0_range, bs_prices, 'b-',  linewidth=2,   label='Black-Scholes (exact)')
plt.plot(S0_range, mc_prices, 'r--', linewidth=1.5, label='Monte Carlo')
plt.axvline(x=S0, color='green', linestyle=':', linewidth=1.5, label=f'S0={S0}')
plt.axvline(x=K,  color='gray',  linestyle=':', linewidth=1.5, label=f'K={K}')
plt.xlabel("Valeur initiale S(0) (€)")
plt.ylabel("Prix du Put (€)")
plt.title("Question 1b : Prix du Put Européen\nBlack-Scholes vs Monte Carlo")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Q1b_BS_vs_MC.png", dpi=150)
plt.show()

# ---------------------------
# Graphique 2 : Comparaison pour différentes valeurs de sigma
# ---------------------------
sigma_range  = np.linspace(0.05, 0.50, 50)
bs_sig       = [black_scholes_put(S0, K, r, s, T)[0] for s in sigma_range]
mc_sig       = [monte_carlo_european_put(S0, K, r, s, T, N, n_sim)[0] for s in sigma_range]

plt.figure(figsize=(10, 5))
plt.plot(sigma_range * 100, bs_sig, 'b-',  linewidth=2,   label='Black-Scholes (exact)')
plt.plot(sigma_range * 100, mc_sig, 'r--', linewidth=1.5, label='Monte Carlo')
plt.axvline(x=sigma*100, color='green', linestyle=':', linewidth=1.5, label=f'σ={sigma*100}%')
plt.xlabel("Volatilité σ (%)")
plt.ylabel("Prix du Put (€)")
plt.title("Question 1b : Sensibilité au paramètre σ\nBlack-Scholes vs Monte Carlo")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Q1b_sigma_sensitivity.png", dpi=150)
plt.show()

# ---------------------------
# Graphique 3 : Comparaison pour différentes valeurs de r
# ---------------------------
r_range  = np.linspace(0.001, 0.10, 50)
bs_r     = [black_scholes_put(S0, K, rv, sigma, T)[0] for rv in r_range]
mc_r     = [monte_carlo_european_put(S0, K, rv, sigma, T, N, n_sim)[0] for rv in r_range]

plt.figure(figsize=(10, 5))
plt.plot(r_range * 100, bs_r, 'b-',  linewidth=2,   label='Black-Scholes (exact)')
plt.plot(r_range * 100, mc_r, 'r--', linewidth=1.5, label='Monte Carlo')
plt.axvline(x=r*100, color='green', linestyle=':', linewidth=1.5, label=f'r={r*100}%')
plt.xlabel("Taux d'intérêt r (%)")
plt.ylabel("Prix du Put (€)")
plt.title("Question 1b : Sensibilité au paramètre r\nBlack-Scholes vs Monte Carlo")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Q1b_r_sensitivity.png", dpi=150)
plt.show()