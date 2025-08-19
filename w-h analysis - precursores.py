import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Constantes
K = 0.9
lambda_nm = 0.15406  # Cu Kα em nm
deg2rad = np.pi/180.0

# Dados obtidos pelo peakfit Origin
example_data = [
    ("COP001","(001)", 19.38031, 0.29591),
    ("COP001","(100)", 33.23660, 0.27743),
    ("COP001","(101)", 38.68230, 0.32237),
    ("COP001","(102)", 52.15536, 0.43600),
    ("COP001","(110)", 59.17427, 0.44078),
    ("COP001","(111)", 62.80474, 0.45869)
]
example_df = pd.DataFrame(example_data, columns=["Sample","hkl","two_theta_deg","fwhm_deg"])

# Função para corrigir o FWHM instrumental
def correct_instrumental(beta_meas_deg, beta_instr_deg=None):
    beta_m_rad = np.asarray(beta_meas_deg) * deg2rad
    if beta_instr_deg is None:
        beta_i_rad = 0.0
    else:
        if np.isscalar(beta_instr_deg):
            beta_i_rad = float(beta_instr_deg) * deg2rad
        else:
            beta_i_rad = np.asarray(beta_instr_deg) * deg2rad
    beta_eff_sq = np.maximum(beta_m_rad**2 - np.array(beta_i_rad)**2, 0.0)
    beta_eff = np.sqrt(beta_eff_sq)
    beta_eff = np.where(beta_eff <= 0.0, 1e-8, beta_eff)
    return beta_eff

# Função Williamson–Hall
def williamson_hall(df_in, sample_col="Sample", twotheta_col="two_theta_deg",
                    fwhm_col="fwhm_deg", beta_instr_col="beta_instr_deg"):
    df = df_in.copy()
    has_instr = beta_instr_col in df.columns

    df["theta_rad"] = (df[twotheta_col]/2.0) * deg2rad
    if has_instr:
        df["beta_eff_rad"] = correct_instrumental(df[fwhm_col].values, df[beta_instr_col].values)
    else:
        df["beta_eff_rad"] = correct_instrumental(df[fwhm_col].values, None)

    df["X = 4·sin(theta)"] = 4.0 * np.sin(df["theta_rad"])
    df["Y = β·cos(theta)"] = df["beta_eff_rad"] * np.cos(df["theta_rad"])
    df["D_Scherrer_nm"] = (K * lambda_nm) / (df["beta_eff_rad"] * np.cos(df["theta_rad"]))

    rows = []
    fit_points = []
    for sample, g in df.groupby(sample_col):
        X = g["X = 4·sin(theta)"].values
        Y = g["Y = β·cos(theta)"].values
        fit_points.append((sample, X, Y))

        b, a = np.polyfit(X, Y, 1)  # slope = microstrain, intercept = tamanho
        n = len(X)

        if n >= 3:
            A = np.vstack([np.ones_like(X), X]).T
            Y_fit = a + b*X
            rss = np.sum((Y - Y_fit)**2)
            dof = max(n - 2, 1)
            sigma2 = rss / dof
            cov = sigma2 * np.linalg.inv(A.T @ A)
            se_a = float(np.sqrt(cov[0,0]))
            se_b = float(np.sqrt(cov[1,1]))
        else:
            se_a = np.nan
            se_b = np.nan

        intercept = a
        epsilon = b
        D_WH_nm = (K * lambda_nm) / intercept if intercept > 0 else np.nan
        if n >= 3 and intercept > 0 and not np.isnan(se_a):
            se_D = (K * lambda_nm / (intercept**2)) * se_a
        else:
            se_D = np.nan

        D_s_vals = g["D_Scherrer_nm"].values
        rows.append({
            "Sample": sample,
            "n_picos": n,
            "intercepto (a)": float(intercept),
            "slope (b) = ε": float(epsilon),
            "SE(a)": se_a,
            "SE(b)": se_b,
            "D_WH (nm)": float(D_WH_nm),
            "SE(D_WH) (nm)": float(se_D),
            "Scherrer_mean (nm)": float(np.mean(D_s_vals)),
            "Scherrer_min (nm)": float(np.min(D_s_vals)),
            "Scherrer_max (nm)": float(np.max(D_s_vals)),
        })

    res = pd.DataFrame(rows)
    return df, res, fit_points

# Rodar o cálculo
df_points, df_results, fit_points = williamson_hall(example_df)

# Arredondar e exibir resultados
for c in ["intercepto (a)","slope (b) = ε","SE(a)","SE(b)","D_WH (nm)","SE(D_WH) (nm)",
          "Scherrer_mean (nm)","Scherrer_min (nm)","Scherrer_max (nm)"]:
    df_results[c] = df_results[c].astype(float).round(6)

print(df_results)

# Gráficos por amostra
for sample, X, Y in fit_points:
    b, a = np.polyfit(X, Y, 1)
    X_line = np.linspace(min(X), max(X), 100)
    Y_line = a + b*X_line

    plt.figure()
    plt.scatter(X, Y, label="Dados (Y vs X)")
    plt.plot(X_line, Y_line, label="Ajuste linear (W–H)", color="red")
    D_val = (K * lambda_nm) / a if a > 0 else np.nan
    eps_val = b
    plt.title(f"{sample} — D_WH ≈ {D_val:.2f} nm, ε ≈ {eps_val:.3e}")
    plt.xlabel("X = 4·sin(θ)")
    plt.ylabel("Y = β·cos(θ)")
    plt.legend()
    plt.show()

# -----------------------------
# Exportar para .dat
# -----------------------------
df_results.to_csv("resultados precursor_WH.dat", sep="\t", index=False)

print("Arquivo 'resultados_WH.dat' salvo com sucesso!")
print(df_results)
