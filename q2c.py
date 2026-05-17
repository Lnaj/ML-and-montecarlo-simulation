# ============================================================
# ACTU-F4002 - Modèles financiers en temps continu
# Question 2c : Intervalle de confiance à 95%
#               pour le prix du Lookback Put
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
    """
    Simule le prix du Lookback Put à strike flottant :
        payoff = M_T - S_T
    avec M_T = max_{0<=t<=T} S(t) observé sur une grille discrète.

    Retourne :
        - price              : prix MC actualisé
        - std_err            : erreur standard sur le prix actualisé
        - payoffs            : payoffs bruts
        - discounted_payoffs : payoffs actualisés
    """
    np.random.seed(seed)

    dt    = T / N
    paths = np.zeros((n_sim, N + 1))
    paths[:, 0] = S0

    for t in range(1, N + 1):
        Z = np.random.standard_normal(n_sim)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    M_T = np.max(paths, axis=1)
    S_T = paths[:, -1]

    # Pour un lookback put floating strike, M_T >= S_T toujours
    payoffs = M_T - S_T

    discount = np.exp(-r * T)
    discounted_payoffs = discount * payoffs

    price   = np.mean(discounted_payoffs)
    std_err = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sim)

    return price, std_err, payoffs, discounted_payoffs


# ============================================================
# PARTIE 1 : Choix du nombre de simulations optimal
# ============================================================
# On teste différentes valeurs de n_sim pour justifier notre choix
N_fixed    = 26
sim_values = [1000, 5000, 10000, 20000, 50000, 100000]

print("=" * 75)
print("  Question 2c : Choix du nombre de simulations")
print(f"  (N = {N_fixed} pas fixé)")
print("=" * 75)
print(f"  {'n_sim':<12} {'Prix MC':<12} {'Std Err':<12} "
      f"{'Largeur IC':<12} {'IC 95%'}")
print("-" * 75)

widths = []

for n_sim in sim_values:
    p, s, _, _ = monte_carlo_lookback_put(
        S0, r, sigma, T, N_fixed, n_sim, seed=SEED
    )
    width = 2 * 1.96 * s
    widths.append(width)

    print(f"  {n_sim:<12} {p:<12.4f} {s:<12.4f} "
          f"{width:<12.4f} [{p - 1.96*s:.4f}, {p + 1.96*s:.4f}]")

print("=" * 75)


# ============================================================
# PARTIE 2 : IC final avec n_sim et N choisis
# ============================================================
# Choix justifié : n_sim=50000, N=26
# -> bon compromis précision / temps de calcul
n_sim_final = 50000
N_final     = 26

price_final, std_final, payoffs_final, discounted_payoffs_final = monte_carlo_lookback_put(
    S0, r, sigma, T, N_final, n_sim_final, seed=SEED
)

IC_low  = price_final - 1.96 * std_final
IC_high = price_final + 1.96 * std_final
width   = IC_high - IC_low

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


# ============================================================
# PARTIE 3 : Stabilité de l'IC sur plusieurs runs
# ============================================================
# On répète l'estimation 30 fois avec des seeds différentes
# pour vérifier la stabilité de l'IC
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
    p, s, _, _ = monte_carlo_lookback_put(
        S0, r, sigma, T, N_final, n_sim_final, seed=run
    )
    low  = p - 1.96 * s
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


# ============================================================
# GRAPHIQUES
# ============================================================

# ---------------------------
# Graphique 1 : Largeur IC en fonction de n_sim
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Panneau gauche : largeur de l'IC 95%
axes[0].semilogx(sim_values, widths, 'o-',
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

# --- Panneau droit : histogramme des payoffs ACTUALISÉS
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


# ---------------------------
# Graphique 2 : Stabilité sur 30 runs
# ---------------------------
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