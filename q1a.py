# ============================================================
# ACTU-F4002 - Modèles financiers en temps continu
# Question 1a : Prix d'un European Put par Monte Carlo
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Paramètres globaux
# ---------------------------
r     = 0.02       # taux d'intérêt
S0    = 100        # valeur initiale du sous-jacent
sigma = 0.15       # volatilité
K     = 110        # strike
T     = 0.5        # maturité en années (6 mois)
N     = 26         # nombre de pas (6 mois en 26 pas ~ hebdomadaire)
dt    = T / N      # pas temporel
n_sim = 10000      # nombre de simulations

# ---------------------------
# Fonction de simulation Monte Carlo - European Put
# ---------------------------
def monte_carlo_european_put(S0, K, r, sigma, T, N, n_sim, seed=42):
    """
    Calcule le prix d'un European Put par Monte Carlo.
    
    Paramètres :
    ------------
    S0     : valeur initiale du sous-jacent
    K      : strike
    r      : taux d'intérêt
    sigma  : volatilité
    T      : maturité (en années)
    N      : nombre de pas temporels
    n_sim  : nombre de simulations
    seed   : graine aléatoire pour reproductibilité
    
    Retourne :
    ----------
    price  : prix estimé du put
    std_err: erreur standard de l'estimation
    paths  : trajectoires simulées (n_sim x N+1)
    """
    np.random.seed(seed)
    dt = T / N
    
    # Matrice des trajectoires : n_sim lignes, N+1 colonnes
    paths = np.zeros((n_sim, N + 1))
    paths[:, 0] = S0
    
    # Simulation des trajectoires
    for t in range(1, N + 1):
        Z = np.random.standard_normal(n_sim)
        paths[:, t] = paths[:, t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )
    
    # Calcul des payoffs à maturité
    S_T    = paths[:, -1]
    payoff = np.maximum(K - S_T, 0)
    
    # Prix actualisé
    discount = np.exp(-r * T)
    price    = discount * np.mean(payoff)
    std_err  = discount * np.std(payoff) / np.sqrt(n_sim)
    
    return price, std_err, paths

# ---------------------------
# Calcul du prix de base
# ---------------------------
price, std_err, paths = monte_carlo_european_put(S0, K, r, sigma, T, N, n_sim)

print("=" * 50)
print("  Question 1a : European Put - Monte Carlo")
print("=" * 50)
print(f"  Paramètres : S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
print(f"  Pas temporel (dt) = {T/N:.4f} ans (~1 semaine)")
print(f"  Nombre de simulations : {n_sim}")
print(f"  Prix estimé du Put    : {price:.4f} €")
print(f"  Erreur standard       : {std_err:.4f} €")
print(f"  IC 95%                : [{price - 1.96*std_err:.4f}, {price + 1.96*std_err:.4f}]")
print("=" * 50)

# ---------------------------
# Graphique 1 : Quelques trajectoires simulées
# ---------------------------
time_grid = np.linspace(0, T, N + 1)

plt.figure(figsize=(10, 5))
for i in range(200):  # afficher 200 trajectoires
    plt.plot(time_grid, paths[i], alpha=0.3, linewidth=0.5, color='steelblue')
plt.axhline(y=K, color='red', linestyle='--', linewidth=1.5, label=f'Strike K={K}')
plt.axhline(y=S0, color='green', linestyle='--', linewidth=1.5, label=f'S0={S0}')
plt.xlabel("Temps (années)")
plt.ylabel("Prix du sous-jacent S(t)")
plt.title("Question 1a : Trajectoires simulées du sous-jacent (Black-Scholes)")
plt.legend()
plt.tight_layout()
plt.savefig("Q1a_trajectoires.png", dpi=150)
plt.show()

# ---------------------------
# Graphique 2 : Distribution des payoffs
# ---------------------------
S_T    = paths[:, -1]
payoff = np.maximum(K - S_T, 0)

plt.figure(figsize=(10, 5))
plt.hist(payoff[payoff > 0], bins=60, color='steelblue', edgecolor='white', alpha=0.8)
plt.xlabel("Payoff du Put (€)")
plt.ylabel("Fréquence")
plt.title(f"Question 1a : Distribution des payoffs positifs\n"
          f"({np.mean(payoff == 0)*100:.1f}% des options expirent sans valeur)")
plt.tight_layout()
plt.savefig("Q1a_payoffs.png", dpi=150)
plt.show()

# ---------------------------
# Graphique 3 : Impact des paramètres r, sigma, S0
# ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

## Impact de r
r_values = np.linspace(0.001, 0.10, 30)
prices_r = [monte_carlo_european_put(S0, K, r_val, sigma, T, N, n_sim)[0] 
            for r_val in r_values]
axes[0].plot(r_values * 100, prices_r, 'o-', color='steelblue', markersize=4)
axes[0].axvline(x=r*100, color='red', linestyle='--', label=f'r={r*100}%')
axes[0].set_xlabel("Taux d'intérêt r (%)")
axes[0].set_ylabel("Prix du Put (€)")
axes[0].set_title("Impact du taux d'intérêt r")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

## Impact de sigma
sigma_values = np.linspace(0.05, 0.50, 30)
prices_sigma = [monte_carlo_european_put(S0, K, r, sig_val, sigma_val=None, T=T, N=N, n_sim=n_sim)[0]
                if False else 
                monte_carlo_european_put(S0, K, r, sig_val, T, N, n_sim)[0]
                for sig_val in sigma_values]
axes[1].plot(sigma_values * 100, prices_sigma, 'o-', color='darkorange', markersize=4)
axes[1].axvline(x=sigma*100, color='red', linestyle='--', label=f'σ={sigma*100}%')
axes[1].set_xlabel("Volatilité σ (%)")
axes[1].set_ylabel("Prix du Put (€)")
axes[1].set_title("Impact de la volatilité σ")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

## Impact de S0
S0_values = np.linspace(70, 140, 30)
prices_S0 = [monte_carlo_european_put(s0_val, K, r, sigma, T, N, n_sim)[0] 
             for s0_val in S0_values]
axes[2].plot(S0_values, prices_S0, 'o-', color='green', markersize=4)
axes[2].axvline(x=S0, color='red', linestyle='--', label=f'S0={S0}')
axes[2].set_xlabel("Valeur initiale S(0) (€)")
axes[2].set_ylabel("Prix du Put (€)")
axes[2].set_title("Impact de la valeur initiale S(0)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle("Question 1a : Analyse de sensibilité du prix du Put Européen", 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q1a_sensibilite.png", dpi=150)
plt.show()