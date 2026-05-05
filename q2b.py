# ============================================================
# ACTU-F4002 - Modèles financiers en temps continu
# Question 2b : Stabilité et convergence du Lookback Put
#               - en fonction du nombre de simulations
#               - en fonction du pas temporel
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
SEED  = 42

# ---------------------------
# Fonction Monte Carlo Lookback Put
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

# ============================================================
# PARTIE 1 : Convergence en fonction du nombre de simulations
# ============================================================
N_fixed    = 26
sim_values = [100, 500, 1000, 2000, 5000, 
              10000, 20000, 50000, 100000]

prices_sim = []
stderr_sim = []

print("=" * 65)
print("  Question 2b - Partie 1 : Convergence en n_sim")
print(f"  (N pas = {N_fixed} fixé, ~hebdomadaire)")
print("=" * 65)
print(f"  {'n_sim':<12} {'Prix MC':<12} {'Std Err':<12} {'IC 95%'}")
print("-" * 65)

for n_sim in sim_values:
    p, s = monte_carlo_lookback_put(
        S0, r, sigma, T, N_fixed, n_sim, seed=SEED
    )
    prices_sim.append(p)
    stderr_sim.append(s)
    print(f"  {n_sim:<12} {p:<12.4f} {s:<12.4f} "
          f"[{p-1.96*s:.4f}, {p+1.96*s:.4f}]")

print("=" * 65)

# ============================================================
# PARTIE 2 : Convergence en fonction du pas temporel
# ============================================================
n_sim_fixed = 50000
N_values    = [6, 13, 26, 52, 90, 180]

prices_N = []
stderr_N = []

print(f"\n{'=' * 65}")
print(f"  Question 2b - Partie 2 : Convergence en pas temporel")
print(f"  (n_sim = {n_sim_fixed} fixé)")
print(f"{'=' * 65}")
print(f"  {'N pas':<8} {'dt (jours)':<14} {'Prix MC':<12} "
      f"{'Std Err':<12} {'IC 95%'}")
print("-" * 65)

for N_val in N_values:
    p, s = monte_carlo_lookback_put(
        S0, r, sigma, T, N_val, n_sim_fixed, seed=SEED
    )
    dt_days = (T / N_val) * 365
    prices_N.append(p)
    stderr_N.append(s)
    print(f"  {N_val:<8} {dt_days:<14.1f} {p:<12.4f} "
          f"{s:<12.4f} [{p-1.96*s:.4f}, {p+1.96*s:.4f}]")

print("=" * 65)

# ============================================================
# GRAPHIQUES
# ============================================================

# ---------------------------
# Graphique 1 : Convergence en n_sim
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

prices_arr = np.array(prices_sim)
stderr_arr = np.array(stderr_sim)

# Prix MC
axes[0].semilogx(sim_values, prices_sim, 'o-',
                  color='steelblue', linewidth=2, markersize=6,
                  label='Prix MC Lookback')
axes[0].fill_between(sim_values,
                      prices_arr - 1.96*stderr_arr,
                      prices_arr + 1.96*stderr_arr,
                      alpha=0.2, color='steelblue', label='IC 95%')
axes[0].set_xlabel("Nombre de simulations (échelle log)")
axes[0].set_ylabel("Prix du Lookback Put (€)")
axes[0].set_title("Convergence du prix\nen fonction du nombre de simulations")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Erreur standard
axes[1].loglog(sim_values, stderr_sim, 'o-',
                color='darkorange', linewidth=2, markersize=6,
                label='Erreur standard')

# Courbe théorique 1/sqrt(n)
n_theory = np.array(sim_values, dtype=float)
scale    = stderr_sim[0] * np.sqrt(sim_values[0])
axes[1].loglog(sim_values, scale / np.sqrt(n_theory), 'r--',
                linewidth=1.5, label=r'$C/\sqrt{n}$ (théorique)')
axes[1].set_xlabel("Nombre de simulations (échelle log)")
axes[1].set_ylabel("Erreur standard (€) (échelle log)")
axes[1].set_title("Décroissance de l'erreur standard\n"
                   r"(convergence en $1/\sqrt{n}$)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2b : Convergence en fonction du nombre de simulations",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2b_convergence_nsim.png", dpi=150)
plt.show()

# ---------------------------
# Graphique 2 : Convergence en pas temporel
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

prices_arr2 = np.array(prices_N)
stderr_arr2 = np.array(stderr_N)

# Prix MC
axes[0].plot(N_values, prices_N, 'o-',
              color='steelblue', linewidth=2, markersize=6,
              label='Prix MC Lookback')
axes[0].fill_between(N_values,
                      prices_arr2 - 1.96*stderr_arr2,
                      prices_arr2 + 1.96*stderr_arr2,
                      alpha=0.2, color='steelblue', label='IC 95%')
axes[0].axvline(x=26,  color='green',  linestyle=':',
                 linewidth=1.5, label='N=26 (hebdomadaire)')
axes[0].axvline(x=180, color='purple', linestyle=':',
                 linewidth=1.5, label='N=180 (journalier)')
axes[0].set_xlabel("Nombre de pas temporels N")
axes[0].set_ylabel("Prix du Lookback Put (€)")
axes[0].set_title("Convergence du prix\nen fonction du pas temporel")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Erreur standard
axes[1].plot(N_values, stderr_N, 'o-',
              color='darkorange', linewidth=2, markersize=6,
              label='Erreur standard')
axes[1].axvline(x=26,  color='green',  linestyle=':',
                 linewidth=1.5, label='N=26 (hebdomadaire)')
axes[1].axvline(x=180, color='purple', linestyle=':',
                 linewidth=1.5, label='N=180 (journalier)')
axes[1].set_xlabel("Nombre de pas temporels N")
axes[1].set_ylabel("Erreur standard (€)")
axes[1].set_title("Erreur standard\nen fonction du pas temporel")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2b : Convergence en fonction du pas temporel",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2b_convergence_dt.png", dpi=150)
plt.show()

# ---------------------------
# Graphique 3 : Heatmap prix selon n_sim ET N
# ---------------------------
sim_grid = [1000, 5000, 10000, 50000, 100000]
N_grid   = [6, 13, 26, 52, 180]

price_matrix  = np.zeros((len(sim_grid), len(N_grid)))
stderr_matrix = np.zeros((len(sim_grid), len(N_grid)))

for i, n_sim in enumerate(sim_grid):
    for j, N in enumerate(N_grid):
        p, s = monte_carlo_lookback_put(
            S0, r, sigma, T, N, n_sim, seed=SEED
        )
        price_matrix[i, j]  = p
        stderr_matrix[i, j] = s

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap prix
im1 = axes[0].imshow(price_matrix, aspect='auto', cmap='RdYlGn')
axes[0].set_xticks(range(len(N_grid)))
axes[0].set_xticklabels([f'N={n}' for n in N_grid])
axes[0].set_yticks(range(len(sim_grid)))
axes[0].set_yticklabels([f'{n:,}' for n in sim_grid])
axes[0].set_xlabel("Pas temporel (N)")
axes[0].set_ylabel("Nombre de simulations")
axes[0].set_title("Prix MC du Lookback Put")
plt.colorbar(im1, ax=axes[0])
for i in range(len(sim_grid)):
    for j in range(len(N_grid)):
        axes[0].text(j, i, f'{price_matrix[i,j]:.3f}',
                    ha='center', va='center',
                    fontsize=8, fontweight='bold')

# Heatmap erreur standard
im2 = axes[1].imshow(stderr_matrix, aspect='auto', cmap='YlOrRd_r')
axes[1].set_xticks(range(len(N_grid)))
axes[1].set_xticklabels([f'N={n}' for n in N_grid])
axes[1].set_yticks(range(len(sim_grid)))
axes[1].set_yticklabels([f'{n:,}' for n in sim_grid])
axes[1].set_xlabel("Pas temporel (N)")
axes[1].set_ylabel("Nombre de simulations")
axes[1].set_title("Erreur standard")
plt.colorbar(im2, ax=axes[1])
for i in range(len(sim_grid)):
    for j in range(len(N_grid)):
        axes[1].text(j, i, f'{stderr_matrix[i,j]:.3f}',
                    ha='center', va='center',
                    fontsize=8, fontweight='bold')

plt.suptitle("Question 2b : Heatmap Prix et Erreur\n"
             "selon n_sim et pas temporel N",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2b_heatmap.png", dpi=150)
plt.show()