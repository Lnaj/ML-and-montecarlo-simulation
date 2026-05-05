# ============================================================
# ACTU-F4002 - Modèles financiers en temps continu
# Question 2e : Comparaison European Put vs Lookback Put
#               + Recommandation selon profil investisseur
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
SEED  = 42

# ---------------------------
# Fonctions
# ---------------------------
def black_scholes_put(S0, K, r, sigma, T):
    d1    = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2    = d1 - sigma*np.sqrt(T)
    price = K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
    return price

def black_scholes_lookback_put(S0, r, sigma, T):
    c1    = (r + 0.5*sigma**2)*T / (sigma*np.sqrt(T))
    c2    = (r - 0.5*sigma**2)*T / (sigma*np.sqrt(T))
    A     = np.exp(-r*T) * norm.cdf(-c2)
    B     = norm.cdf(-c1)
    C     = (sigma**2/(2*r)) * (norm.cdf(c1) - np.exp(-r*T)*norm.cdf(-c2))
    price = S0 * (A - B + C)
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
    payoffs  = np.maximum(K - S_T, 0)
    discount = np.exp(-r*T)
    price    = discount * np.mean(payoffs)
    std_err  = discount * np.std(payoffs) / np.sqrt(n_sim)
    return price, std_err, payoffs, paths

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
    return price, std_err, payoffs, paths

# ============================================================
# CALCUL DES PRIX
# ============================================================
ep_bs = black_scholes_put(S0, K, r, sigma, T)
lp_bs = black_scholes_lookback_put(S0, r, sigma, T)

ep_mc, ep_std, ep_payoffs, ep_paths = monte_carlo_european_put(
    S0, K, r, sigma, T, N, n_sim, seed=SEED
)
lp_mc, lp_std, lp_payoffs, lp_paths = monte_carlo_lookback_put(
    S0, r, sigma, T, N, n_sim, seed=SEED
)

# ============================================================
# TABLEAU COMPARATIF COMPLET
# ============================================================
print("=" * 65)
print("  Question 2e : European Put vs Lookback Put")
print("=" * 65)
print(f"\n  {'Critère':<35} {'European Put':<15} {'Lookback Put'}")
print("-" * 65)
print(f"  {'Prix Black-Scholes (€)':<35} {ep_bs:<15.4f} {lp_bs:.4f}")
print(f"  {'Prix Monte Carlo (€)':<35} {ep_mc:<15.4f} {lp_mc:.4f}")
print(f"  {'Erreur standard MC (€)':<35} {ep_std:<15.4f} {lp_std:.4f}")
print(f"  {'IC 95% bas (€)':<35} "
      f"{ep_mc-1.96*ep_std:<15.4f} {lp_mc-1.96*lp_std:.4f}")
print(f"  {'IC 95% haut (€)':<35} "
      f"{ep_mc+1.96*ep_std:<15.4f} {lp_mc+1.96*lp_std:.4f}")
print(f"  {'% payoffs nuls':<35} "
      f"{np.mean(ep_payoffs==0)*100:<15.1f} "
      f"{np.mean(lp_payoffs==0)*100:.1f}")
print(f"  {'Payoff moyen (€)':<35} "
      f"{np.mean(ep_payoffs):<15.4f} {np.mean(lp_payoffs):.4f}")
print(f"  {'Payoff max (€)':<35} "
      f"{np.max(ep_payoffs):<15.4f} {np.max(lp_payoffs):.4f}")
print(f"  {'Std des payoffs (€)':<35} "
      f"{np.std(ep_payoffs):<15.4f} {np.std(lp_payoffs):.4f}")
print("-" * 65)
print(f"  {'Prime Lookback / European':<35} x{lp_bs/ep_bs:.2f}")
print(f"  {'Différence de prix (€)':<35} {lp_bs-ep_bs:.4f}")
print("=" * 65)

# ============================================================
# GRAPHIQUES
# ============================================================

# ---------------------------
# Graphique 1 : Prix en fonction de S0 et sigma
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

S0_range = np.linspace(70, 140, 100)
ep_S0    = [black_scholes_put(s, K, r, sigma, T) for s in S0_range]
lp_S0    = [black_scholes_lookback_put(s, r, sigma, T) for s in S0_range]

axes[0].plot(S0_range, ep_S0, 'b-',  linewidth=2,
              label=f'European Put (K={K})')
axes[0].plot(S0_range, lp_S0, 'r-',  linewidth=2,
              label='Lookback Put')
axes[0].fill_between(S0_range, ep_S0, lp_S0,
                      alpha=0.15, color='green',
                      label='Prime Lookback')
axes[0].axvline(x=S0, color='gray', linestyle=':',
                 linewidth=1.5, label=f'S0={S0}')
axes[0].axvline(x=K,  color='black', linestyle=':',
                 linewidth=1.5, label=f'K={K}')
axes[0].set_xlabel("Valeur initiale S(0) (€)")
axes[0].set_ylabel("Prix de l'option (€)")
axes[0].set_title("Prix en fonction de S(0)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

sig_range = np.linspace(0.05, 0.50, 100)
ep_sig    = [black_scholes_put(S0, K, r, s, T) for s in sig_range]
lp_sig    = [black_scholes_lookback_put(S0, r, s, T) for s in sig_range]

axes[1].plot(sig_range*100, ep_sig, 'b-',  linewidth=2,
              label=f'European Put (K={K})')
axes[1].plot(sig_range*100, lp_sig, 'r-',  linewidth=2,
              label='Lookback Put')
axes[1].fill_between(sig_range*100, ep_sig, lp_sig,
                      alpha=0.15, color='green',
                      label='Prime Lookback')
axes[1].axvline(x=sigma*100, color='gray', linestyle=':',
                 linewidth=1.5, label=f'σ={sigma*100}%')
axes[1].set_xlabel("Volatilité σ (%)")
axes[1].set_ylabel("Prix de l'option (€)")
axes[1].set_title("Prix en fonction de σ")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2e : European Put vs Lookback Put - Prix BS",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2e_prix_comparaison.png", dpi=150)
plt.show()

# ---------------------------
# Graphique 2 : Distribution des payoffs comparée
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogrammes superposés
axes[0].hist(ep_payoffs, bins=60, alpha=0.6,
              color='steelblue', edgecolor='white',
              label=f'European Put\n(moy={np.mean(ep_payoffs):.2f}€)')
axes[0].hist(lp_payoffs, bins=60, alpha=0.6,
              color='darkorange', edgecolor='white',
              label=f'Lookback Put\n(moy={np.mean(lp_payoffs):.2f}€)')
axes[0].axvline(x=np.mean(ep_payoffs), color='steelblue',
                 linestyle='--', linewidth=2)
axes[0].axvline(x=np.mean(lp_payoffs), color='darkorange',
                 linestyle='--', linewidth=2)
axes[0].set_xlabel("Payoff (€)")
axes[0].set_ylabel("Fréquence")
axes[0].set_title("Distribution des payoffs\nEuropean Put vs Lookback Put")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Payoff moyen en fonction de S0
mc_ep_S0 = [monte_carlo_european_put(s, K, r, sigma, T, N, 10000)[0]
            for s in S0_range]
mc_lp_S0 = [monte_carlo_lookback_put(s, r, sigma, T, N, 10000)[0]
            for s in S0_range]

axes[1].plot(S0_range, mc_ep_S0, 'b-',  linewidth=2,
              label='European Put (MC)')
axes[1].plot(S0_range, mc_lp_S0, 'r-',  linewidth=2,
              label='Lookback Put (MC)')
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
time_grid = np.linspace(0, T, N+1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['blue', 'red', 'green', 'orange', 'purple']

for idx, c in enumerate(colors):
    S_traj  = ep_paths[idx]
    M_traj  = np.maximum.accumulate(S_traj)
    ep_pay  = max(K - S_traj[-1], 0)
    lp_pay  = max(M_traj[-1] - S_traj[-1], 0)

    axes[0].plot(time_grid, S_traj, color=c, linewidth=1.5,
                  label=f'Traj {idx+1} | EP={ep_pay:.1f}€')
    axes[1].plot(time_grid, S_traj, color=c, linewidth=1.5,
                  label=f'Traj {idx+1} | LP={lp_pay:.1f}€')
    axes[1].plot(time_grid, M_traj, color=c, linewidth=1,
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
axes[1].set_title("Lookback Put\n(trait plein=S(t), pointillé=M(t))\n"
                   "(payoff = max(M(T)-S(T), 0))")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle("Question 2e : Illustration des payoffs sur 5 trajectoires",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Q2e_trajectoires.png", dpi=150)
plt.show()