# ============================================================
# ACTU-F4002 - Modèles financiers en temps continu
# Question 2e : Comparaison European Put vs Lookback Put
#               + Recommandation selon profil investisseur
#               (version cohérente avec correction BGK)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------
# Paramètres globaux
# ---------------------------
r      = 0.02
S0     = 90
sigma  = 0.20
K      = 110
T      = 0.5
N_euro = 26
N_lb   = 180          # On retient une grille plus fine pour le Lookback
n_sim  = 50000
SEED   = 42

# Constante de Broadie-Glasserman-Kou
BETA = 0.5826

# ---------------------------
# Fonctions théoriques
# ---------------------------
def black_scholes_put(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return price

def black_scholes_lookback_put(S0, r, sigma, T):
    c1 = (r + 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
    c2 = (r - 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
    A  = np.exp(-r * T) * norm.cdf(-c2)
    B  = norm.cdf(-c1)
    C  = (sigma**2 / (2 * r)) * (norm.cdf(c1) - np.exp(-r * T) * norm.cdf(-c2))
    price = S0 * (A - B + C)
    return price

# ---------------------------
# Fonctions Monte Carlo
# ---------------------------
def monte_carlo_european_put(S0, K, r, sigma, T, N, n_sim, seed=42):
    np.random.seed(seed)

    dt = T / N
    paths = np.zeros((n_sim, N + 1))
    paths[:, 0] = S0

    for t in range(1, N + 1):
        Z = np.random.standard_normal(n_sim)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    S_T = paths[:, -1]
    payoffs = np.maximum(K - S_T, 0)
    discount = np.exp(-r * T)

    discounted_payoffs = discount * payoffs
    price = np.mean(discounted_payoffs)
    std_err = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sim)

    return price, std_err, discounted_payoffs, paths


def monte_carlo_lookback_put_corrected(S0, r, sigma, T, N, n_sim, seed=42):
    """
    Lookback Put MC avec correction BGK.
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

    return price, std_err, discounted_payoffs, paths


# ============================================================
# CALCUL DES PRIX
# ============================================================
ep_bs = black_scholes_put(S0, K, r, sigma, T)
lp_bs = black_scholes_lookback_put(S0, r, sigma, T)

ep_mc, ep_std, ep_payoffs, ep_paths = monte_carlo_european_put(
    S0, K, r, sigma, T, N_euro, n_sim, seed=SEED
)

lp_mc, lp_std, lp_payoffs, lp_paths = monte_carlo_lookback_put_corrected(
    S0, r, sigma, T, N_lb, n_sim, seed=SEED
)

ep_ic_low  = ep_mc - 1.96 * ep_std
ep_ic_high = ep_mc + 1.96 * ep_std

lp_ic_low  = lp_mc - 1.96 * lp_std
lp_ic_high = lp_mc + 1.96 * lp_std

# ============================================================
# TABLEAU COMPARATIF COMPLET
# ============================================================
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

# ============================================================
# GRAPHIQUES
# ============================================================

# ---------------------------
# Graphique 1 : Prix théoriques en fonction de S0 et sigma
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

S0_range = np.linspace(70, 140, 100)
ep_S0 = [black_scholes_put(s, K, r, sigma, T) for s in S0_range]
lp_S0 = [black_scholes_lookback_put(s, r, sigma, T) for s in S0_range]

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
lp_sig = [black_scholes_lookback_put(S0, r, s, T) for s in sig_range]

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

# ---------------------------
# Graphique 2 : Distribution des payoffs et prix MC
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogrammes superposés des payoffs actualisés
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

# Prix MC en fonction de S0
mc_ep_S0 = [monte_carlo_european_put(s, K, r, sigma, T, N_euro, 10000, seed=SEED)[0]
            for s in S0_range]
mc_lp_S0 = [monte_carlo_lookback_put_corrected(s, r, sigma, T, N_lb, 10000, seed=SEED)[0]
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

# ---------------------------
# Graphique 3 : Scénarios de trajectoires
# ---------------------------
time_grid_euro = np.linspace(0, T, N_euro + 1)
time_grid_lb   = np.linspace(0, T, N_lb + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['blue', 'red', 'green', 'orange', 'purple']

for idx, c in enumerate(colors):
    # Traj European Put
    S_traj_ep = ep_paths[idx]
    ep_pay = max(K - S_traj_ep[-1], 0)

    axes[0].plot(time_grid_euro, S_traj_ep, color=c, linewidth=1.5,
                 label=f'Traj {idx+1} | EP={ep_pay:.1f}€')

    # Traj Lookback Put
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