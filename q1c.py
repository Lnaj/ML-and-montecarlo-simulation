# ============================================================
# ACTU-F4002 - Modèles financiers en temps continu
# Question 1c : Convergence de Monte Carlo
#               - en fonction du nombre de simulations
#               - en fonction du pas temporel
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------
# Paramètres globaux
# ---------------------------
r     = 0.02
S0    = 100
sigma = 0.15
K     = 110
T     = 0.5
SEED  = 42

# ---------------------------
# Fonctions
# ---------------------------
def black_scholes_put(S0, K, r, sigma, T):
    d1    = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2    = d1 - sigma*np.sqrt(T)
    price = K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
    return price

def monte_carlo_european_put(S0, K, r, sigma, T, N, n_sim, seed=42):
    np.random.seed(seed)
    dt    = T / N
    paths = np.zeros((n_sim, N+1))
    paths[:, 0] = S0

    for t in range(1, N+1):
        Z = np.random.standard_normal(n_sim)
        paths[:, t] = paths[:, t-1] * np.exp(
            (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        )

    S_T      = paths[:, -1]
    payoff   = np.maximum(K - S_T, 0)
    discount = np.exp(-r*T)
    price    = discount * np.mean(payoff)
    std_err  = discount * np.std(payoff) / np.sqrt(n_sim)

    return price, std_err

# ---------------------------
# Prix de référence Black-Scholes
# ---------------------------
bs_price = black_scholes_put(S0, K, r, sigma, T)
print(f"Prix Black-Scholes (référence) : {bs_price:.4f} €\n")

# ============================================================
# PARTIE 1 : Convergence en fonction du nombre de simulations
# ============================================================
N_fixed    = 26   # pas temporel fixé (hebdomadaire)
sim_values = [100, 500, 1000, 2000, 5000, 10000, 
              20000, 50000, 100000]

prices_sim  = []
stderr_sim  = []
errors_sim  = []

print("=" * 60)
print("  Convergence en fonction du nombre de simulations")
print("  (N pas = 26, ~hebdomadaire)")
print("=" * 60)
print(f"  {'n_sim':<12} {'Prix MC':<12} {'Std Err':<12} "
      f"{'|Erreur|':<12} {'IC 95%'}")
print("-" * 60)

for n_sim in sim_values:
    price, std_err = monte_carlo_european_put(
        S0, K, r, sigma, T, N_fixed, n_sim, seed=SEED
    )
    error = abs(price - bs_price)
    prices_sim.append(price)
    stderr_sim.append(std_err)
    errors_sim.append(error)
    
    print(f"  {n_sim:<12} {price:<12.4f} {std_err:<12.4f} "
          f"{error:<12.4f} "
          f"[{price-1.96*std_err:.4f}, {price+1.96*std_err:.4f}]")

print("=" * 60)

# ============================================================
# PARTIE 2 : Convergence en fonction du pas temporel
# ============================================================
n_sim_fixed = 50000   # nombre de simulations fixé
# Pas temporels : de 6 pas à 180 pas (1 jour)
N_values = [6, 10, 13, 26, 52, 90, 120, 180]
# Correspondance approximative :
# 6   -> ~1 mois
# 13  -> ~2 semaines
# 26  -> ~1 semaine
# 52  -> ~2.5 jours
# 180 -> ~1 jour

prices_N  = []
stderr_N  = []
errors_N  = []

print(f"\n{'=' * 60}")
print(f"  Convergence en fonction du pas temporel")
print(f"  (n_sim = {n_sim_fixed})")
print(f"{'=' * 60}")
print(f"  {'N pas':<8} {'dt (jours)':<14} {'Prix MC':<12} "
      f"{'Std Err':<12} {'|Erreur|':<10}")
print("-" * 60)

for N in N_values:
    price, std_err = monte_carlo_european_put(
        S0, K, r, sigma, T, N, n_sim_fixed, seed=SEED
    )
    error  = abs(price - bs_price)
    dt_days = (T / N) * 365
    prices_N.append(price)
    stderr_N.append(std_err)
    errors_N.append(error)
    
    print(f"  {N:<8} {dt_days:<14.1f} {price:<12.4f} "
          f"{std_err:<12.4f} {error:<10.4f}")

print("=" * 60)

# ============================================================
# GRAPHIQUES
# ============================================================

# ---------------------------
# Graphique 1 : Convergence en n_sim
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Prix MC vs BS
axes[0].semilogx(sim_values, prices_sim, 'o-', 
                  color='steelblue', linewidth=2, markersize=6,
                  label='Prix Monte Carlo')
axes[0].axhline(y=bs_price, color='red', linestyle='--', 
                 linewidth=2, label=f'Black-Scholes = {bs_price:.4f}€')

# Intervalle de confiance 95%
prices_arr = np.array(prices_sim)
stderr_arr = np.array(stderr_sim)
axes[0].fill_between(sim_values,
                      prices_arr - 1.96*stderr_arr,
                      prices_arr + 1.96*stderr_arr,
                      alpha=0.2, color='steelblue', label='IC 95%')
axes[0].set_xlabel("Nombre de simulations (échelle log)")
axes[0].set_ylabel("Prix du Put (€)")
axes[0].set_title("Convergence du prix MC\nen fonction du nombre de simulations")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Erreur absolue vs n_sim
axes[1].loglog(sim_values, errors_sim, 'o-', 
                color='darkorange', linewidth=2, markersize=6,
                label='|Erreur MC - BS|')

# Courbe théorique 1/sqrt(n)
n_theory = np.array(sim_values, dtype=float)
scale    = errors_sim[0] * np.sqrt(sim_values[0])
axes[1].loglog(sim_values, scale / np.sqrt(n_theory), 'r--', 
                linewidth=1.5, label=r'$C/\sqrt{n}$ (théorique)')
axes[1].set_xlabel("Nombre de simulations (échelle log)")
axes[1].set_ylabel("Erreur absolue (€) (échelle log)")
axes[1].set_title("Erreur absolue MC vs BS\n(convergence en $1/\\sqrt{n}$)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 1c : Convergence en fonction du nombre de simulations",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q1c_convergence_nsim.png", dpi=150)
plt.show()

# ---------------------------
# Graphique 2 : Convergence en pas temporel
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

dt_days = [(T/N)*365 for N in N_values]

# Prix MC vs BS
axes[0].plot(N_values, prices_N, 'o-', 
              color='steelblue', linewidth=2, markersize=6,
              label='Prix Monte Carlo')
axes[0].axhline(y=bs_price, color='red', linestyle='--', 
                 linewidth=2, label=f'Black-Scholes = {bs_price:.4f}€')
prices_arr2 = np.array(prices_N)
stderr_arr2 = np.array(stderr_N)
axes[0].fill_between(N_values,
                      prices_arr2 - 1.96*stderr_arr2,
                      prices_arr2 + 1.96*stderr_arr2,
                      alpha=0.2, color='steelblue', label='IC 95%')
axes[0].set_xlabel("Nombre de pas temporels N")
axes[0].set_ylabel("Prix du Put (€)")
axes[0].set_title("Convergence du prix MC\nen fonction du pas temporel")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Erreur vs N
axes[1].plot(N_values, errors_N, 'o-', 
              color='darkorange', linewidth=2, markersize=6,
              label='|Erreur MC - BS|')
axes[1].axvline(x=26, color='green', linestyle=':', 
                 linewidth=1.5, label='N=26 (hebdomadaire)')
axes[1].axvline(x=180, color='purple', linestyle=':', 
                 linewidth=1.5, label='N=180 (journalier)')
axes[1].set_xlabel("Nombre de pas temporels N")
axes[1].set_ylabel("Erreur absolue (€)")
axes[1].set_title("Erreur absolue MC vs BS\nen fonction du pas temporel")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 1c : Convergence en fonction du pas temporel",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q1c_convergence_dt.png", dpi=150)
plt.show()

# ---------------------------
# Graphique 3 : Heatmap Prix MC selon n_sim ET N
# ---------------------------
sim_grid = [1000, 5000, 10000, 50000, 100000]
N_grid   = [6, 13, 26, 52, 180]

price_matrix = np.zeros((len(sim_grid), len(N_grid)))
error_matrix = np.zeros((len(sim_grid), len(N_grid)))

for i, n_sim in enumerate(sim_grid):
    for j, N in enumerate(N_grid):
        p, _ = monte_carlo_european_put(
            S0, K, r, sigma, T, N, n_sim, seed=SEED
        )
        price_matrix[i, j] = p
        error_matrix[i, j] = abs(p - bs_price)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap prix
im1 = axes[0].imshow(price_matrix, aspect='auto', cmap='RdYlGn')
axes[0].set_xticks(range(len(N_grid)))
axes[0].set_xticklabels([f'N={n}' for n in N_grid])
axes[0].set_yticks(range(len(sim_grid)))
axes[0].set_yticklabels([f'{n:,}' for n in sim_grid])
axes[0].set_xlabel("Pas temporel (N)")
axes[0].set_ylabel("Nombre de simulations")
axes[0].set_title(f"Prix MC (BS = {bs_price:.4f}€)")
plt.colorbar(im1, ax=axes[0])
for i in range(len(sim_grid)):
    for j in range(len(N_grid)):
        axes[0].text(j, i, f'{price_matrix[i,j]:.3f}',
                    ha='center', va='center', fontsize=8, fontweight='bold')

# Heatmap erreur
im2 = axes[1].imshow(error_matrix, aspect='auto', cmap='YlOrRd')
axes[1].set_xticks(range(len(N_grid)))
axes[1].set_xticklabels([f'N={n}' for n in N_grid])
axes[1].set_yticks(range(len(sim_grid)))
axes[1].set_yticklabels([f'{n:,}' for n in sim_grid])
axes[1].set_xlabel("Pas temporel (N)")
axes[1].set_ylabel("Nombre de simulations")
axes[1].set_title("Erreur absolue |MC - BS|")
plt.colorbar(im2, ax=axes[1])
for i in range(len(sim_grid)):
    for j in range(len(N_grid)):
        axes[1].text(j, i, f'{error_matrix[i,j]:.3f}',
                    ha='center', va='center', fontsize=8, fontweight='bold')

plt.suptitle("Question 1c : Heatmap Prix et Erreur\n"
             "selon n_sim et pas temporel N",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q1c_heatmap.png", dpi=150)
plt.show()