# ============================================================
# ACTU-F4002 - Modèles financiers en temps continu
# Question 2a : Prix d'un Lookback Put par Monte Carlo
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Paramètres globaux
# ---------------------------
r     = 0.02
S0    = 90
sigma = 0.20
T     = 0.5
N     = 26        # pas temporel hebdomadaire
n_sim = 50000
SEED  = 42

# ---------------------------
# Fonction Monte Carlo Lookback Put
# ---------------------------
def monte_carlo_lookback_put(S0, r, sigma, T, N, n_sim, seed=42):
    """
    Calcule le prix d'un Lookback Put par Monte Carlo.
    Payoff = max(0, max(S) - S(T))

    Paramètres :
    ------------
    S0    : valeur initiale
    r     : taux d'intérêt
    sigma : volatilité
    T     : maturité (en années)
    N     : nombre de pas temporels
    n_sim : nombre de simulations
    seed  : graine aléatoire

    Retourne :
    ----------
    price   : prix estimé
    std_err : erreur standard
    payoffs : vecteur des payoffs
    paths   : trajectoires simulées
    """
    np.random.seed(seed)
    dt    = T / N
    paths = np.zeros((n_sim, N+1))
    paths[:, 0] = S0

    # Simulation des trajectoires
    for t in range(1, N+1):
        Z = np.random.standard_normal(n_sim)
        paths[:, t] = paths[:, t-1] * np.exp(
            (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        )

    # Maximum sur chaque trajectoire
    M_T = np.max(paths, axis=1)

    # Valeur finale
    S_T = paths[:, -1]

    # Payoff Lookback Put
    payoffs  = np.maximum(M_T - S_T, 0)
    discount = np.exp(-r*T)
    price    = discount * np.mean(payoffs)
    std_err  = discount * np.std(payoffs, ddof=1) / np.sqrt(n_sim)

    return price, std_err, payoffs, paths

# ---------------------------
# Calcul du prix
# ---------------------------
price, std_err, payoffs, paths = monte_carlo_lookback_put(
    S0, r, sigma, T, N, n_sim, seed=SEED
)

print("=" * 55)
print("  Question 2a : Lookback Put - Monte Carlo")
print("=" * 55)
print(f"  Paramètres : S0={S0}, r={r}, sigma={sigma}, T={T}")
print(f"  N={N} pas, {n_sim} simulations")
print(f"  Prix estimé  : {price:.4f} €")
print(f"  Std Erreur   : {std_err:.4f} €")
print(f"  IC 95%       : [{price-1.96*std_err:.4f}, "
      f"{price+1.96*std_err:.4f}]")
print("=" * 55)

# ---------------------------
# Graphique 1 : Trajectoires + maximum
# ---------------------------
time_grid = np.linspace(0, T, N+1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Trajectoires
for i in range(200):
    axes[0].plot(time_grid, paths[i], alpha=0.2,
                  linewidth=0.5, color='steelblue')

# Mettre en évidence 5 trajectoires avec leur maximum
colors = ['red', 'green', 'orange', 'purple', 'brown']
for i, c in enumerate(colors):
    axes[0].plot(time_grid, paths[i], linewidth=1.5, 
                  color=c, alpha=0.9)
    max_idx = np.argmax(paths[i])
    axes[0].plot(time_grid[max_idx], paths[i, max_idx],
                  '*', markersize=12, color=c)

axes[0].set_xlabel("Temps (années)")
axes[0].set_ylabel("Prix du sous-jacent S(t)")
axes[0].set_title("Trajectoires simulées\n(★ = maximum de chaque trajectoire colorée)")
axes[0].grid(True, alpha=0.3)

# Distribution des payoffs
axes[1].hist(payoffs[payoffs > 0], bins=60,
              color='darkorange', edgecolor='white', alpha=0.8)
axes[1].axvline(x=np.mean(payoffs), color='red',
                 linestyle='--', linewidth=2,
                 label=f'Moyenne = {np.mean(payoffs):.2f}€')
axes[1].set_xlabel("Payoff du Lookback Put (€)")
axes[1].set_ylabel("Fréquence")
axes[1].set_title(f"Distribution des payoffs positifs\n"
                   f"({np.mean(payoffs==0)*100:.1f}% expirent sans valeur)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2a : Lookback Put - Monte Carlo",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2a_lookback_MC.png", dpi=150)
plt.show()