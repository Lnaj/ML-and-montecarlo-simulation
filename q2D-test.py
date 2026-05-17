# ============================================================
# ACTU-F4002 - Modèles financiers en temps continu
# Question 2d : Prix théorique Lookback Put - Formule BS
#               + biais de discrétisation
#               + correction de Broadie-Glasserman-Kou (1997)
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

# Constante de Broadie-Glasserman-Kou
# beta = -zeta(1/2) / sqrt(2*pi) ≈ 0.5826
BETA = 0.5826

# ---------------------------
# Fonctions
# ---------------------------
def black_scholes_lookback_put(S0, r, sigma, T):
    """
    Prix théorique exact du Lookback Put à strike flottant
    dans le modèle de Black-Scholes.
    """
    c1 = (r + 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
    c2 = (r - 0.5 * sigma**2) * T / (sigma * np.sqrt(T))

    A = np.exp(-r * T) * norm.cdf(-c2)
    B = norm.cdf(-c1)
    C = (sigma**2 / (2 * r)) * (norm.cdf(c1) - np.exp(-r * T) * norm.cdf(-c2))

    price = S0 * (A - B + C)
    return price, c1, c2, A, B, C


def monte_carlo_lookback_put_biased(S0, r, sigma, T, N, n_sim, seed=42):
    """
    Monte Carlo Lookback Put SANS correction :
    maximum discret observé sur la grille seulement.
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

    M_T = np.max(paths, axis=1)
    S_T = paths[:, -1]

    # M_T >= S_T, donc le max(...,0) est mathématiquement inutile
    payoffs = M_T - S_T
    discount = np.exp(-r * T)

    discounted_payoffs = discount * payoffs
    price = np.mean(discounted_payoffs)
    std_err = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sim)

    return price, std_err


def monte_carlo_lookback_put_corrected(S0, r, sigma, T, N, n_sim, seed=42):
    """
    Monte Carlo Lookback Put AVEC correction BGK (1997).
    Le maximum discret est corrigé vers le haut :
        M_corrected = M_discrete * exp(beta * sigma * sqrt(dt))
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

    return price, std_err


def theoretical_bias(S0, sigma, T, N):
    """
    Approximation théorique du biais BGK :
        biais ≈ beta * sigma * S0 * sqrt(T/N)
    """
    return BETA * sigma * S0 * np.sqrt(T / N)


# ============================================================
# CALCULS PRINCIPAUX
# ============================================================
lp_bs, c1, c2, A, B, C = black_scholes_lookback_put(S0, r, sigma, T)

# MC sans correction
lp_mc_bias, lp_std_bias = monte_carlo_lookback_put_biased(
    S0, r, sigma, T, N, n_sim, seed=SEED
)
IC_low_bias  = lp_mc_bias - 1.96 * lp_std_bias
IC_high_bias = lp_mc_bias + 1.96 * lp_std_bias

# MC avec correction BGK
lp_mc_corr, lp_std_corr = monte_carlo_lookback_put_corrected(
    S0, r, sigma, T, N, n_sim, seed=SEED
)
IC_low_corr  = lp_mc_corr - 1.96 * lp_std_corr
IC_high_corr = lp_mc_corr + 1.96 * lp_std_corr

# Biais théorique / réel
bias_real = lp_bs - lp_mc_bias
bias_theo = theoretical_bias(S0, sigma, T, N)


# ============================================================
# COMPARAISON DÉTAILLÉE POUR N = 180
# ============================================================
N_180 = 180

# Monte Carlo sans correction
mc_180_bias, std_180_bias = monte_carlo_lookback_put_biased(
    S0, r, sigma, T, N_180, n_sim, seed=SEED
)
IC_low_180_bias  = mc_180_bias - 1.96 * std_180_bias
IC_high_180_bias = mc_180_bias + 1.96 * std_180_bias

# Monte Carlo avec correction BGK
mc_180_corr, std_180_corr = monte_carlo_lookback_put_corrected(
    S0, r, sigma, T, N_180, n_sim, seed=SEED
)
IC_low_180_corr  = mc_180_corr - 1.96 * std_180_corr
IC_high_180_corr = mc_180_corr + 1.96 * std_180_corr

print("\n" + "=" * 70)
print("  Comparaison détaillée pour N = 180")
print("=" * 70)

print("\n  --- Monte Carlo SANS correction (N=180) ---")
print(f"  Prix BS exact           : {lp_bs:.4f} €")
print(f"  Prix MC sans correction : {mc_180_bias:.4f} €")
print(f"  Erreur standard         : {std_180_bias:.4f} €")
print(f"  IC 95%                  : [{IC_low_180_bias:.4f}, {IC_high_180_bias:.4f}]")
print(f"  Écart absolu à BS       : {abs(lp_bs - mc_180_bias):.4f} €")
print(f"  Écart relatif à BS      : {abs(lp_bs - mc_180_bias)/lp_bs*100:.2f} %")
print(f"  BS dans l'IC 95% ?      : {'✓ OUI' if IC_low_180_bias <= lp_bs <= IC_high_180_bias else '✗ NON'}")

print("\n  --- Monte Carlo AVEC correction BGK (N=180) ---")
print(f"  Prix BS exact           : {lp_bs:.4f} €")
print(f"  Prix MC corrigé BGK     : {mc_180_corr:.4f} €")
print(f"  Erreur standard         : {std_180_corr:.4f} €")
print(f"  IC 95%                  : [{IC_low_180_corr:.4f}, {IC_high_180_corr:.4f}]")
print(f"  Écart absolu à BS       : {abs(lp_bs - mc_180_corr):.4f} €")
print(f"  Écart relatif à BS      : {abs(lp_bs - mc_180_corr)/lp_bs*100:.2f} %")
print(f"  BS dans l'IC 95% ?      : {'✓ OUI' if IC_low_180_corr <= lp_bs <= IC_high_180_corr else '✗ NON'}")

print("=" * 70)