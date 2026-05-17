# ============================================================
# ACTU-F4002 - Modèles financiers en temps continu
# Question 2 : Lookback Put / European Put
# Script fusionné : 2a + 2b + 2c + 2d + 2e
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# PARAMÈTRES GLOBAUX
# ============================================================
r     = 0.02
S0    = 90
sigma = 0.20
T     = 0.5
K     = 110
SEED  = 42

# Paramètres principaux retenus
N_weekly = 26
N_daily  = 180
n_sim    = 50000

# Constante Broadie-Glasserman-Kou
# beta = -zeta(1/2) / sqrt(2*pi) ≈ 0.5826
BETA = 0.5826


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def simulate_gbm_paths(S0, r, sigma, T, N, n_sim, seed=42):
    """
    Simule des trajectoires GBM sous la mesure risque-neutre.
    Retour :
        paths : array shape (n_sim, N+1)
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

    return paths


def monte_carlo_lookback_put(S0, r, sigma, T, N, n_sim, seed=42, corrected=False):
    """
    Prix Monte Carlo d'un lookback put floating strike :
        payoff = M_T - S_T
    avec M_T = max_t S(t), observé sur grille discrète.

    Si corrected=True :
        applique la correction BGK :
        M_corrected = M_discrete * exp(beta * sigma * sqrt(dt))

    Retour :
        price
        std_err
        payoffs
        discounted_payoffs
        paths
    """
    paths = simulate_gbm_paths(S0, r, sigma, T, N, n_sim, seed)
    dt = T / N

    M_T = np.max(paths, axis=1)

    if corrected:
        correction_factor = np.exp(BETA * sigma * np.sqrt(dt))
        M_T = M_T * correction_factor

    S_T = paths[:, -1]
    payoffs = M_T - S_T
    discount = np.exp(-r * T)
    discounted_payoffs = discount * payoffs

    price = np.mean(discounted_payoffs)
    std_err = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sim)

    return price, std_err, payoffs, discounted_payoffs, paths


def monte_carlo_european_put(S0, K, r, sigma, T, N, n_sim, seed=42):
    """
    Prix Monte Carlo d'un put européen.
    Retour :
        price
        std_err
        payoffs
        discounted_payoffs
        paths
    """
    paths = simulate_gbm_paths(S0, r, sigma, T, N, n_sim, seed)

    S_T = paths[:, -1]
    payoffs = np.maximum(K - S_T, 0)
    discount = np.exp(-r * T)
    discounted_payoffs = discount * payoffs

    price = np.mean(discounted_payoffs)
    std_err = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sim)

    return price, std_err, payoffs, discounted_payoffs, paths


def black_scholes_put(S0, K, r, sigma, T):
    """
    Prix Black-Scholes d'un put européen.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def black_scholes_lookback_put(S0, r, sigma, T):
    """
    Prix théorique exact du lookback put floating strike
    dans le modèle de Black-Scholes.
    """
    c1 = (r + 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
    c2 = (r - 0.5 * sigma**2) * T / (sigma * np.sqrt(T))

    A = np.exp(-r * T) * norm.cdf(-c2)
    B = norm.cdf(-c1)
    C = (sigma**2 / (2 * r)) * (
        norm.cdf(c1) - np.exp(-r * T) * norm.cdf(-c2)
    )

    price = S0 * (A - B + C)
    return price, c1, c2, A, B, C


def theoretical_bias(S0, sigma, T, N):
    """
    Approximation théorique du biais BGK :
        biais ≈ beta * sigma * S0 * sqrt(T/N)
    """
    return BETA * sigma * S0 * np.sqrt(T / N)


# ============================================================
# QUESTION 2a : Prix d'un Lookback Put par Monte Carlo
# ============================================================
price_2a, std_err_2a, payoffs_2a, discounted_payoffs_2a, paths_2a = monte_carlo_lookback_put(
    S0, r, sigma, T, N_weekly, n_sim, seed=SEED, corrected=False
)

print("=" * 55)
print("  Question 2a : Lookback Put - Monte Carlo")
print("=" * 55)
print(f"  Paramètres : S0={S0}, r={r}, sigma={sigma}, T={T}")
print(f"  N={N_weekly} pas, {n_sim} simulations")
print(f"  Prix estimé  : {price_2a:.4f} €")
print(f"  Std Erreur   : {std_err_2a:.4f} €")
print(f"  IC 95%       : [{price_2a-1.96*std_err_2a:.4f}, "
      f"{price_2a+1.96*std_err_2a:.4f}]")
print("=" * 55)

# Graphique 2a
time_grid = np.linspace(0, T, N_weekly + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Trajectoires
for i in range(200):
    axes[0].plot(time_grid, paths_2a[i], alpha=0.2,
                 linewidth=0.5, color='steelblue')

# 5 trajectoires mises en évidence
colors = ['red', 'green', 'orange', 'purple', 'brown']
for i, c in enumerate(colors):
    axes[0].plot(time_grid, paths_2a[i], linewidth=1.5, color=c, alpha=0.9)
    max_idx = np.argmax(paths_2a[i])
    axes[0].plot(time_grid[max_idx], paths_2a[i, max_idx],
                 '*', markersize=12, color=c)

axes[0].set_xlabel("Temps (années)")
axes[0].set_ylabel("Prix du sous-jacent S(t)")
axes[0].set_title("Trajectoires simulées\n(★ = maximum de chaque trajectoire colorée)")
axes[0].grid(True, alpha=0.3)

# Distribution des payoffs
axes[1].hist(payoffs_2a[payoffs_2a > 0], bins=60,
             color='darkorange', edgecolor='white', alpha=0.8)
axes[1].axvline(x=np.mean(payoffs_2a), color='red',
                linestyle='--', linewidth=2,
                label=f'Moyenne = {np.mean(payoffs_2a):.2f}€')
axes[1].set_xlabel("Payoff du Lookback Put (€)")
axes[1].set_ylabel("Fréquence")
axes[1].set_title(f"Distribution des payoffs positifs\n"
                  f"({np.mean(payoffs_2a==0)*100:.1f}% expirent sans valeur)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2a : Lookback Put - Monte Carlo",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2a_lookback_MC.png", dpi=150)
plt.show()


# ============================================================
# QUESTION 2b : Stabilité et convergence
# ============================================================

# ---------------------------
# Partie 1 : convergence en n_sim
# ---------------------------
sim_values = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
prices_sim = []
stderr_sim = []

print("=" * 65)
print("  Question 2b - Partie 1 : Convergence en n_sim")
print(f"  (N pas = {N_weekly} fixé, ~hebdomadaire)")
print("=" * 65)
print(f"  {'n_sim':<12} {'Prix MC':<12} {'Std Err':<12} {'IC 95%'}")
print("-" * 65)

for n_sim_val in sim_values:
    p, s, _, _, _ = monte_carlo_lookback_put(
        S0, r, sigma, T, N_weekly, n_sim_val, seed=SEED, corrected=False
    )
    prices_sim.append(p)
    stderr_sim.append(s)
    print(f"  {n_sim_val:<12} {p:<12.4f} {s:<12.4f} "
          f"[{p-1.96*s:.4f}, {p+1.96*s:.4f}]")

print("=" * 65)

# ---------------------------
# Partie 2 : convergence en N
# ---------------------------
n_sim_fixed = 50000
N_values = [6, 13, 26, 52, 90, 180]

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
    p, s, _, _, _ = monte_carlo_lookback_put(
        S0, r, sigma, T, N_val, n_sim_fixed, seed=SEED, corrected=False
    )
    dt_days = (T / N_val) * 365
    prices_N.append(p)
    stderr_N.append(s)
    print(f"  {N_val:<8} {dt_days:<14.1f} {p:<12.4f} "
          f"{s:<12.4f} [{p-1.96*s:.4f}, {p+1.96*s:.4f}]")

print("=" * 65)

# Graphiques 2b
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

prices_arr = np.array(prices_sim)
stderr_arr = np.array(stderr_sim)

axes[0].semilogx(sim_values, prices_sim, 'o-',
                 color='steelblue', linewidth=2, markersize=6,
                 label='Prix MC Lookback')
axes[0].fill_between(sim_values,
                     prices_arr - 1.96 * stderr_arr,
                     prices_arr + 1.96 * stderr_arr,
                     alpha=0.2, color='steelblue', label='IC 95%')
axes[0].set_xlabel("Nombre de simulations (échelle log)")
axes[0].set_ylabel("Prix du Lookback Put (€)")
axes[0].set_title("Convergence du prix\nen fonction du nombre de simulations")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

n_theory = np.array(sim_values, dtype=float)
scale = stderr_sim[0] * np.sqrt(sim_values[0])
axes[1].loglog(sim_values, stderr_sim, 'o-',
               color='darkorange', linewidth=2, markersize=6,
               label='Erreur standard')
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

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

prices_arr2 = np.array(prices_N)
stderr_arr2 = np.array(stderr_N)

axes[0].plot(N_values, prices_N, 'o-',
             color='steelblue', linewidth=2, markersize=6,
             label='Prix MC Lookback')
axes[0].fill_between(N_values,
                     prices_arr2 - 1.96 * stderr_arr2,
                     prices_arr2 + 1.96 * stderr_arr2,
                     alpha=0.2, color='steelblue', label='IC 95%')
axes[0].axvline(x=26, color='green', linestyle=':',
                linewidth=1.5, label='N=26 (hebdomadaire)')
axes[0].axvline(x=180, color='purple', linestyle=':',
                linewidth=1.5, label='N=180 (journalier)')
axes[0].set_xlabel("Nombre de pas temporels N")
axes[0].set_ylabel("Prix du Lookback Put (€)")
axes[0].set_title("Convergence du prix\nen fonction du pas temporel")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(N_values, stderr_N, 'o-',
             color='darkorange', linewidth=2, markersize=6,
             label='Erreur standard')
axes[1].axvline(x=26, color='green', linestyle=':',
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

# Heatmap
sim_grid = [1000, 5000, 10000, 50000, 100000]
N_grid = [6, 13, 26, 52, 180]

price_matrix = np.zeros((len(sim_grid), len(N_grid)))
stderr_matrix = np.zeros((len(sim_grid), len(N_grid)))

for i, n_sim_val in enumerate(sim_grid):
    for j, N_val in enumerate(N_grid):
        p, s, _, _, _ = monte_carlo_lookback_put(
            S0, r, sigma, T, N_val, n_sim_val, seed=SEED, corrected=False
        )
        price_matrix[i, j] = p
        stderr_matrix[i, j] = s

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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
        axes[0].text(j, i, f'{price_matrix[i, j]:.3f}',
                     ha='center', va='center',
                     fontsize=8, fontweight='bold')

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
        axes[1].text(j, i, f'{stderr_matrix[i, j]:.3f}',
                     ha='center', va='center',
                     fontsize=8, fontweight='bold')

plt.suptitle("Question 2b : Heatmap Prix et Erreur\nselon n_sim et pas temporel N",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2b_heatmap.png", dpi=150)
plt.show()


# ============================================================
# QUESTION 2c : Intervalle de confiance à 95%
# ============================================================

sim_values_2c = [1000, 5000, 10000, 20000, 50000, 100000]

print("=" * 75)
print("  Question 2c : Choix du nombre de simulations")
print(f"  (N = {N_weekly} pas fixé)")
print("=" * 75)
print(f"  {'n_sim':<12} {'Prix MC':<12} {'Std Err':<12} "
      f"{'Largeur IC':<12} {'IC 95%'}")
print("-" * 75)

widths = []

for n_sim_val in sim_values_2c:
    p, s, _, _, _ = monte_carlo_lookback_put(
        S0, r, sigma, T, N_weekly, n_sim_val, seed=SEED, corrected=False
    )
    width = 2 * 1.96 * s
    widths.append(width)

    print(f"  {n_sim_val:<12} {p:<12.4f} {s:<12.4f} "
          f"{width:<12.4f} [{p - 1.96*s:.4f}, {p + 1.96*s:.4f}]")

print("=" * 75)

n_sim_final = 50000
N_final = 26

price_final, std_final, payoffs_final, discounted_payoffs_final, _ = monte_carlo_lookback_put(
    S0, r, sigma, T, N_final, n_sim_final, seed=SEED, corrected=False
)

IC_low = price_final - 1.96 * std_final
IC_high = price_final + 1.96 * std_final
width = IC_high - IC_low

print(f"\n{'=' * 60}")
print(f"  Question 2c : Intervalle de confiance final")
print(f"{'=' * 60}")
print(f"  Paramètres retenus :")
print(f"  -> n_sim = {n_sim_final}")
print(f"  -> N     = {N_final} pas (~hebdomadaire)")
print(f"\n  Prix estimé      : {price_final:.4f} €")
print(f"  Erreur standard  : {std_final:.4f} €")
print(f"  Borne inférieure : {IC_low:.4f} €")
print(f"  Borne supérieure : {IC_high:.4f} €")
print(f"  Largeur IC 95%   : {width:.4f} €")
print(f"{'=' * 60}")

# Stabilité sur plusieurs runs
n_runs = 30
prices_runs = []
IC_lows = []
IC_highs = []

print(f"\n{'=' * 75}")
print(f"  Stabilité de l'IC sur {n_runs} runs indépendants")
print(f"  (n_sim={n_sim_final}, N={N_final})")
print(f"{'=' * 75}")
print(f"  {'Run':<8} {'Prix MC':<12} {'IC bas':<12} {'IC haut':<12}")
print("-" * 75)

for run in range(n_runs):
    p, s, _, _, _ = monte_carlo_lookback_put(
        S0, r, sigma, T, N_final, n_sim_final, seed=run, corrected=False
    )
    low = p - 1.96 * s
    high = p + 1.96 * s

    prices_runs.append(p)
    IC_lows.append(low)
    IC_highs.append(high)

    print(f"  {run + 1:<8} {p:<12.4f} {low:<12.4f} {high:.4f}")

print(f"\n  Moyenne des prix  : {np.mean(prices_runs):.4f} €")
print(f"  Std des prix      : {np.std(prices_runs, ddof=1):.4f} €")
print(f"  Min prix          : {np.min(prices_runs):.4f} €")
print(f"  Max prix          : {np.max(prices_runs):.4f} €")
print("=" * 75)

# Graphiques 2c
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].semilogx(sim_values_2c, widths, 'o-',
                 color='steelblue', linewidth=2, markersize=8)
axes[0].axhline(y=widths[-1], color='red', linestyle='--',
                linewidth=1.5,
                label=f'Largeur finale = {widths[-1]:.4f} €')
axes[0].axvline(x=n_sim_final, color='green', linestyle=':',
                linewidth=1.5,
                label=f'Choix : n_sim = {n_sim_final}')
axes[0].set_xlabel("Nombre de simulations (échelle log)")
axes[0].set_ylabel("Largeur de l'IC 95% (€)")
axes[0].set_title("Largeur de l'IC 95%\nen fonction du nombre de simulations")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(discounted_payoffs_final, bins=60,
             color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axvline(x=price_final, color='red',
                linestyle='--', linewidth=2,
                label=f'Moyenne = {price_final:.4f} €')
axes[1].axvline(x=IC_low, color='orange',
                linestyle=':', linewidth=2,
                label=f'IC bas = {IC_low:.4f} €')
axes[1].axvline(x=IC_high, color='orange',
                linestyle=':', linewidth=2,
                label=f'IC haut = {IC_high:.4f} €')
axes[1].set_xlabel("Payoff actualisé du Lookback Put (€)")
axes[1].set_ylabel("Fréquence")
axes[1].set_title("Distribution des payoffs actualisés\navec IC 95%")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2c : Intervalle de confiance à 95%",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2c_IC.png", dpi=150)
plt.show()

plt.figure(figsize=(12, 5))
runs = np.arange(1, n_runs + 1)

plt.plot(runs, prices_runs, 'o-',
         color='steelblue', linewidth=1.5,
         markersize=5, label='Prix MC par run')
plt.fill_between(runs, IC_lows, IC_highs,
                 alpha=0.2, color='steelblue',
                 label='IC 95% par run')
plt.axhline(y=np.mean(prices_runs), color='red',
            linestyle='--', linewidth=2,
            label=f'Moyenne = {np.mean(prices_runs):.4f} €')
plt.axhline(y=np.mean(prices_runs) + np.std(prices_runs, ddof=1),
            color='orange', linestyle=':', linewidth=1.5,
            label=f'±1 std = {np.std(prices_runs, ddof=1):.4f} €')
plt.axhline(y=np.mean(prices_runs) - np.std(prices_runs, ddof=1),
            color='orange', linestyle=':', linewidth=1.5)
plt.xlabel("Numéro du run")
plt.ylabel("Prix estimé (€)")
plt.title(f"Question 2c : Stabilité du prix MC sur {n_runs} runs indépendants\n"
          f"(n_sim={n_sim_final}, N={N_final})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Q2c_stabilite.png", dpi=150)
plt.show()


# ============================================================
# QUESTION 2d : Prix théorique + biais de discrétisation + BGK
# ============================================================
lp_bs, c1, c2, A, B, C = black_scholes_lookback_put(S0, r, sigma, T)

lp_mc_bias, lp_std_bias, _, _, _ = monte_carlo_lookback_put(
    S0, r, sigma, T, N_weekly, n_sim, seed=SEED, corrected=False
)
IC_low_bias = lp_mc_bias - 1.96 * lp_std_bias
IC_high_bias = lp_mc_bias + 1.96 * lp_std_bias

lp_mc_corr, lp_std_corr, _, _, _ = monte_carlo_lookback_put(
    S0, r, sigma, T, N_weekly, n_sim, seed=SEED, corrected=True
)
IC_low_corr = lp_mc_corr - 1.96 * lp_std_corr
IC_high_corr = lp_mc_corr + 1.96 * lp_std_corr

bias_real = lp_bs - lp_mc_bias
bias_theo = theoretical_bias(S0, sigma, T, N_weekly)

print("=" * 75)
print("  Question 2d : Prix théorique du Lookback Put")
print("  + biais de discrétisation + correction BGK (1997)")
print("=" * 75)

print(f"\n  Paramètres : S0={S0}, r={r}, sigma={sigma}, T={T}, N={N_weekly}, n_sim={n_sim}")

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

print(f"\n  --- Analyse du biais pour N={N_weekly} ---")
print(f"  Biais réel (BS - MC biaisé)      : {bias_real:.4f} €")
print(f"  Biais théorique BGK approx.      : {bias_theo:.4f} €")
print(f"  Réduction grâce à la correction  : {abs(lp_bs-lp_mc_bias)-abs(lp_bs-lp_mc_corr):.4f} €")

p26_bias, s26_bias, _, _, _ = monte_carlo_lookback_put(S0, r, sigma, T, 26, n_sim, seed=SEED, corrected=False)
p26_corr, s26_corr, _, _, _ = monte_carlo_lookback_put(S0, r, sigma, T, 26, n_sim, seed=SEED, corrected=True)
p180_bias, s180_bias, _, _, _ = monte_carlo_lookback_put(S0, r, sigma, T, 180, n_sim, seed=SEED, corrected=False)
p180_corr, s180_corr, _, _, _ = monte_carlo_lookback_put(S0, r, sigma, T, 180, n_sim, seed=SEED, corrected=True)

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

IC_low_180_corr = p180_corr - 1.96 * s180_corr
IC_high_180_corr = p180_corr + 1.96 * s180_corr

print(f"\n  --- Monte Carlo AVEC correction BGK (N=180) ---")
print(f"  Prix MC corrigé         : {p180_corr:.4f} €")
print(f"  Erreur standard         : {s180_corr:.4f} €")
print(f"  IC 95%                  : [{IC_low_180_corr:.4f}, {IC_high_180_corr:.4f}]")
print(f"  Écart absolu à BS       : {abs(lp_bs - p180_corr):.4f} €")
print(f"  Écart relatif à BS      : {abs(lp_bs - p180_corr)/lp_bs*100:.2f} %")
print(f"  BS dans IC 95% ?        : {'✓ OUI' if IC_low_180_corr <= lp_bs <= IC_high_180_corr else '✗ NON'}")

# Graphiques 2d
N_values_2d = [6, 13, 26, 52, 90, 180, 365]
biases_real = []
biases_theo = []
prices_bias = []
prices_corr = []

for N_val in N_values_2d:
    p_bias, _, _, _, _ = monte_carlo_lookback_put(
        S0, r, sigma, T, N_val, n_sim, seed=SEED, corrected=False
    )
    p_corr, _, _, _, _ = monte_carlo_lookback_put(
        S0, r, sigma, T, N_val, n_sim, seed=SEED, corrected=True
    )

    biases_real.append(lp_bs - p_bias)
    biases_theo.append(theoretical_bias(S0, sigma, T, N_val))
    prices_bias.append(p_bias)
    prices_corr.append(p_corr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(N_values_2d, biases_real, 'o-',
             color='red', linewidth=2, markersize=8,
             label='Biais réel (BS - MC biaisé)')
axes[0].plot(N_values_2d, biases_theo, 's--',
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

N_arr = np.array(N_values_2d, dtype=float)
scale = biases_real[0] * np.sqrt(N_values_2d[0])

axes[1].loglog(N_values_2d, biases_real, 'o-',
               color='red', linewidth=2, markersize=8,
               label='Biais réel')
axes[1].loglog(N_values_2d, scale / np.sqrt(N_arr), 'k--',
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

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(N_values_2d, prices_bias, 'o-',
             color='red', linewidth=2, markersize=8,
             label='MC sans correction')
axes[0].plot(N_values_2d, prices_corr, 's-',
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

axes[1].plot(N_values_2d, errors_bias, 'o-',
             color='red', linewidth=2, markersize=8,
             label='|Erreur| sans correction')
axes[1].plot(N_values_2d, errors_corr, 's-',
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

S0_range = np.linspace(70, 140, 100)
sig_range = np.linspace(0.05, 0.50, 60)

bs_s0 = [black_scholes_lookback_put(s, r, sigma, T)[0] for s in S0_range]
mc_s0_bias = [monte_carlo_lookback_put(s, r, sigma, T, 26, n_sim, seed=SEED, corrected=False)[0]
              for s in S0_range]
mc_s0_corr = [monte_carlo_lookback_put(s, r, sigma, T, 26, n_sim, seed=SEED, corrected=True)[0]
              for s in S0_range]

bs_sig = [black_scholes_lookback_put(S0, r, s, T)[0] for s in sig_range]
mc_sig_bias = [monte_carlo_lookback_put(S0, r, s, T, 26, n_sim, seed=SEED, corrected=False)[0]
               for s in sig_range]
mc_sig_corr = [monte_carlo_lookback_put(S0, r, s, T, 26, n_sim, seed=SEED, corrected=True)[0]
               for s in sig_range]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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


# ============================================================
# QUESTION 2e : European Put vs Lookback Put
# ============================================================
N_euro = 26
N_lb = 180

ep_bs = black_scholes_put(S0, K, r, sigma, T)
lp_bs = black_scholes_lookback_put(S0, r, sigma, T)[0]

ep_mc, ep_std, ep_payoffs_raw, ep_payoffs, ep_paths = monte_carlo_european_put(
    S0, K, r, sigma, T, N_euro, n_sim, seed=SEED
)

lp_mc, lp_std, lp_payoffs_raw, lp_payoffs, lp_paths = monte_carlo_lookback_put(
    S0, r, sigma, T, N_lb, n_sim, seed=SEED, corrected=True
)

ep_ic_low = ep_mc - 1.96 * ep_std
ep_ic_high = ep_mc + 1.96 * ep_std

lp_ic_low = lp_mc - 1.96 * lp_std
lp_ic_high = lp_mc + 1.96 * lp_std

print("=" * 80)
print("  Question 2e : European Put vs Lookback Put")
print("  (Lookback valorisé par BS et MC corrigé BGK)")
print("=" * 80)
print(f"\n  {'Critère':<38} {'European Put':<18} {'Lookback Put'}")
print("-" * 80)
print(f"  {'Prix Black-Scholes (€)':<38} {ep_bs:<18.4f} {lp_bs:.4f}")
print(f"  {'Prix Monte Carlo (€)':<38} {ep_mc:<18.4f} {lp_mc:.4f}")
print(f"  {'Erreur standard MC (€)':<38} {ep_std:<18.4f} {lp_std:.4f}")
print(f"  {'IC 95% bas (€)':<38} {ep_ic_low:<18.4f} {lp_ic_low:.4f}")
print(f"  {'IC 95% haut (€)':<38} {ep_ic_high:<18.4f} {lp_ic_high:.4f}")
print(f"  {'BS dans IC 95% MC ?':<38} "
      f"{'✓ OUI' if ep_ic_low <= ep_bs <= ep_ic_high else '✗ NON':<18} "
      f"{'✓ OUI' if lp_ic_low <= lp_bs <= lp_ic_high else '✗ NON'}")
print(f"  {'% payoffs nuls':<38} "
      f"{np.mean(ep_payoffs == 0) * 100:<18.1f} "
      f"{np.mean(lp_payoffs == 0) * 100:.1f}")
print(f"  {'Payoff actualisé moyen (€)':<38} "
      f"{np.mean(ep_payoffs):<18.4f} {np.mean(lp_payoffs):.4f}")
print(f"  {'Payoff actualisé max (€)':<38} "
      f"{np.max(ep_payoffs):<18.4f} {np.max(lp_payoffs):.4f}")
print(f"  {'Std des payoffs actualisés (€)':<38} "
      f"{np.std(ep_payoffs, ddof=1):<18.4f} {np.std(lp_payoffs, ddof=1):.4f}")
print("-" * 80)
print(f"  {'Prime Lookback / European':<38} x{lp_bs / ep_bs:.2f}")
print(f"  {'Différence de prix BS (€)':<38} {lp_bs - ep_bs:.4f}")
print("=" * 80)

# Graphiques 2e
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

S0_range = np.linspace(70, 140, 100)
ep_S0 = [black_scholes_put(s, K, r, sigma, T) for s in S0_range]
lp_S0 = [black_scholes_lookback_put(s, r, sigma, T)[0] for s in S0_range]

axes[0].plot(S0_range, ep_S0, 'b-', linewidth=2,
             label=f'European Put (K={K})')
axes[0].plot(S0_range, lp_S0, 'r-', linewidth=2,
             label='Lookback Put')
axes[0].fill_between(S0_range, ep_S0, lp_S0,
                     alpha=0.15, color='green',
                     label='Écart de prix')
axes[0].axvline(x=S0, color='gray', linestyle=':',
                linewidth=1.5, label=f'S0={S0}')
axes[0].axvline(x=K, color='black', linestyle=':',
                linewidth=1.5, label=f'K={K}')
axes[0].set_xlabel("Valeur initiale S(0) (€)")
axes[0].set_ylabel("Prix de l'option (€)")
axes[0].set_title("Prix en fonction de S(0)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

sig_range = np.linspace(0.05, 0.50, 100)
ep_sig = [black_scholes_put(S0, K, r, s, T) for s in sig_range]
lp_sig = [black_scholes_lookback_put(S0, r, s, T)[0] for s in sig_range]

axes[1].plot(sig_range * 100, ep_sig, 'b-', linewidth=2,
             label=f'European Put (K={K})')
axes[1].plot(sig_range * 100, lp_sig, 'r-', linewidth=2,
             label='Lookback Put')
axes[1].fill_between(sig_range * 100, ep_sig, lp_sig,
                     alpha=0.15, color='green',
                     label='Écart de prix')
axes[1].axvline(x=sigma * 100, color='gray', linestyle=':',
                linewidth=1.5, label=f'σ={sigma*100:.0f}%')
axes[1].set_xlabel("Volatilité σ (%)")
axes[1].set_ylabel("Prix de l'option (€)")
axes[1].set_title("Prix en fonction de σ")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2e : European Put vs Lookback Put - Prix théoriques",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2e_prix_comparaison.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(ep_payoffs, bins=60, alpha=0.6,
             color='steelblue', edgecolor='white',
             label=f'European Put\n(moy={np.mean(ep_payoffs):.2f}€)')
axes[0].hist(lp_payoffs, bins=60, alpha=0.6,
             color='darkorange', edgecolor='white',
             label=f'Lookback Put BGK\n(moy={np.mean(lp_payoffs):.2f}€)')
axes[0].axvline(x=np.mean(ep_payoffs), color='steelblue',
                linestyle='--', linewidth=2)
axes[0].axvline(x=np.mean(lp_payoffs), color='darkorange',
                linestyle='--', linewidth=2)
axes[0].set_xlabel("Payoff actualisé (€)")
axes[0].set_ylabel("Fréquence")
axes[0].set_title("Distribution des payoffs actualisés")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

mc_ep_S0 = [monte_carlo_european_put(s, K, r, sigma, T, N_euro, 10000, seed=SEED)[0]
            for s in S0_range]
mc_lp_S0 = [monte_carlo_lookback_put(s, r, sigma, T, N_lb, 10000, seed=SEED, corrected=True)[0]
            for s in S0_range]

axes[1].plot(S0_range, mc_ep_S0, 'b-', linewidth=2,
             label='European Put (MC)')
axes[1].plot(S0_range, mc_lp_S0, 'r-', linewidth=2,
             label='Lookback Put BGK (MC)')
axes[1].fill_between(S0_range, mc_ep_S0, mc_lp_S0,
                     alpha=0.15, color='green',
                     label='Différence de prix')
axes[1].axvline(x=S0, color='gray', linestyle=':',
                linewidth=1.5, label=f'S0={S0}')
axes[1].set_xlabel("Valeur initiale S(0) (€)")
axes[1].set_ylabel("Prix estimé MC (€)")
axes[1].set_title("Prix MC en fonction de S(0)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2e : Comparaison des payoffs et prix MC",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2e_payoffs_comparaison.png", dpi=150)
plt.show()

time_grid_euro = np.linspace(0, T, N_euro + 1)
time_grid_lb = np.linspace(0, T, N_lb + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['blue', 'red', 'green', 'orange', 'purple']

for idx, c in enumerate(colors):
    S_traj_ep = ep_paths[idx]
    ep_pay = max(K - S_traj_ep[-1], 0)

    axes[0].plot(time_grid_euro, S_traj_ep, color=c, linewidth=1.5,
                 label=f'Traj {idx+1} | EP={ep_pay:.1f}€')

    S_traj_lp = lp_paths[idx]
    M_traj_lp = np.maximum.accumulate(S_traj_lp)

    dt_lb = T / N_lb
    correction_factor = np.exp(BETA * sigma * np.sqrt(dt_lb))
    lp_pay = max(M_traj_lp[-1] * correction_factor - S_traj_lp[-1], 0)

    axes[1].plot(time_grid_lb, S_traj_lp, color=c, linewidth=1.5,
                 label=f'Traj {idx+1} | LP={lp_pay:.1f}€')
    axes[1].plot(time_grid_lb, M_traj_lp, color=c, linewidth=1,
                 linestyle='--', alpha=0.6)

axes[0].axhline(y=K, color='black', linestyle='--',
                linewidth=1.5, label=f'Strike K={K}')
axes[0].set_xlabel("Temps (années)")
axes[0].set_ylabel("S(t)")
axes[0].set_title("European Put\n(payoff = max(K-S(T), 0))")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Temps (années)")
axes[1].set_ylabel("S(t) et M(t)")
axes[1].set_title("Lookback Put corrigé BGK\n(trait plein=S(t), pointillé=M(t))")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2e : Illustration des payoffs sur 5 trajectoires",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2e_trajectoires.png", dpi=150)
plt.show()