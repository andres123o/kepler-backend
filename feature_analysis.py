"""
Feature Selection Pipeline — Campaña: Primer Depósito
Target: usuarios_primer_cashin (log-transformado)

Etapas:
  0. Carga y diagnóstico básico
  1. Near-zero variance
  2. VIF — multicolinealidad entre features
  3. PCA — estructura latente
  4. Mutual Information vs target
  5. MRMR — máxima relevancia, mínima redundancia
  6. Granger Causality — causalidad temporal
  7. Estabilidad en walk-forward CV
  8. Tabla resumen de decisión

Ejecutar desde kepler-backend/:
    python experimentacion/primer_depositos/feature_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no necesita pantalla
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import grangercausalitytests
from tabulate import tabulate

# ────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ────────────────────────────────────────────────────────────────

DATA_PATH   = Path("experimentacion/master_consolidado_final_v2.csv")
RESULTS_DIR = Path("experimentacion/primer_depositos/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "usuarios_primer_cashin"

FUNNEL_FEATURES = [
    # Volumen del pipeline (cantidad en distintos puntos)
    "step_09_full_account",       # usuarios que completaron el proceso digital
    "full_users_aprobados",       # stock disponible para depositar (t+1)
    # Calidad de conversión (tasas del embudo)
    "tasa_basic_a_risk",          # calidad KYC del cohort entrante (Granger p=0.038)
    "tasa_risk_a_fulldata",       # avance post-riesgo
    "tasa_fulldata_a_video",      # cuello de botella principal (Granger p≈0)
    # Velocidad y perfil
    "mediana_dias_registro_a_full",  # velocidad de onboarding (MRMR rank 2 tras controlar volumen)
    "pct_perfil_conservador",     # mix de riesgo — conservadores depositan más (Granger p=0.004)
    "pct_perfil_arriesgado",      # mix de riesgo — señal complementaria
    # Eliminados:
    # "usuarios_registro_base"     → señal 3-4 semanas antes, no t+1
    # "tasa_video_a_review"        → NZV (cv=0.0088, casi todos avanzan)
    # "tasa_review_a_aprobado"     → Granger p=0.620, estabilidad 0%
    # "tasa_registro_a_aprobado"   → VIF=38.64, derivada de otras tasas, Granger p=0.118
    # "tasa_rechazo_implicita_kyc" → Granger p=0.608, estabilidad 0%
]

MACRO_FEATURES = [
    # Spread captura la señal de Banrep + TES juntos — más información que la tasa sola
    "spread_tes_banrep",
    "TRM",
    "sp500_cambio_semanal_pct",   # implicaciones económicas detrás del dato, no el dato crudo
    "brent_cambio_semanal_pct",   # ídem
    "colcap_cambio_semanal_pct",  # ídem
    "trends_cdt",                 # intención inversora (Granger p=0.004)
    "trends_acciones",            # apetito de riesgo complementario
    "pct_dias_quincena",          # % días de la semana que caen en ventana post-quincena (re-ingeniería de is_ventana_quincena)
    # Eliminados:
    # "Tasa_Intervencion_Mensual" → reemplazada por spread_tes_banrep (más completa)
    # "is_ventana_quincena"       → reemplazada por pct_dias_quincena (continua, no binaria)
]

ALL_FEATURES = FUNNEL_FEATURES + MACRO_FEATURES

GRANGER_MAX_LAG    = 3
CV_N_FOLDS         = 5
MRMR_K             = 12   # top K features en MRMR
STABILITY_THRESH   = 0.70  # feature debe aparecer en >= 70% de folds
NZV_CV_THRESHOLD   = 0.01  # coef. variación < 1% = near-zero variance

# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────

def section(title):
    w = 72
    print(f"\n{'=' * w}\n  {title}\n{'=' * w}")

def subsection(title):
    print(f"\n-- {title} {'-' * max(2, 64 - len(title))}")

def save_csv(df, name):
    p = RESULTS_DIR / f"{name}.csv"
    df.to_csv(p)
    print(f"  → {p}")

def save_plot(fig, name):
    p = RESULTS_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

def grupo(feat):
    return "funnel" if feat in FUNNEL_FEATURES else "macro"

# ────────────────────────────────────────────────────────────────
# 0. CARGA Y DIAGNÓSTICO BÁSICO
# ────────────────────────────────────────────────────────────────

section("ETAPA 0 — Carga y diagnóstico básico")

df_raw = pd.read_csv(DATA_PATH, sep=";")
print(f"  Shape original: {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas")

# ── Re-ingeniería: is_ventana_quincena (binaria) → pct_dias_quincena (continua)
# Días del mes que pertenecen a la ventana post-quincena colombiana
# Fuente: cicloUsuario.md — picos en días {28,29,30} fin de mes y {1,2,3} inicio + {15,16,17} mitad
_QUINCENA_DAYS = {1, 2, 3, 15, 16, 17, 28, 29, 30}

def _calc_pct_quincena(semana_str):
    try:
        monday = pd.to_datetime(semana_str, dayfirst=True)
        count = sum(1 for i in range(7) if (monday + pd.Timedelta(days=i)).day in _QUINCENA_DAYS)
        return round(count / 7, 4)
    except Exception:
        return np.nan

df_raw["pct_dias_quincena"] = df_raw["semana"].apply(_calc_pct_quincena)
print(f"  pct_dias_quincena — media: {df_raw['pct_dias_quincena'].mean():.3f}  "
      f"min: {df_raw['pct_dias_quincena'].min():.3f}  "
      f"max: {df_raw['pct_dias_quincena'].max():.3f}  "
      f"valores únicos: {df_raw['pct_dias_quincena'].nunique()}")

available = [f for f in ALL_FEATURES if f in df_raw.columns]
missing   = [f for f in ALL_FEATURES if f not in df_raw.columns]

if missing:
    print(f"  Advertencia — Features no encontrados en el CSV: {missing}")

df = df_raw[available + [TARGET]].copy()
print(f"  Features cargados: {len(available)}  |  target: {TARGET}")

# Valores faltantes
subsection("Valores faltantes")
nulls = df.isnull().sum()
nulls = nulls[nulls > 0]
if nulls.empty:
    print("  Sin valores faltantes.")
else:
    null_pct = (nulls / len(df) * 100).round(1)
    print(tabulate(
        pd.DataFrame({"nulos": nulls, "%": null_pct}),
        headers="keys", tablefmt="simple"
    ))

# Imputar con mediana para no bloquear el análisis
df = df.fillna(df.median(numeric_only=True))

# Transformación log del target
y_raw = df[TARGET].values
y_log = np.log1p(y_raw)

print(f"\n  Target — min: {y_raw.min():.0f}  max: {y_raw.max():.0f}  "
      f"media: {y_raw.mean():.0f}  std: {y_raw.std():.0f}")

# ────────────────────────────────────────────────────────────────
# 1. NEAR-ZERO VARIANCE
# ────────────────────────────────────────────────────────────────

section("ETAPA 1 — Near-zero variance")
print(f"  Criterio: coeficiente de variación normalizado (std / |media|) < {NZV_CV_THRESHOLD}")

nzv_rows = []
for f in available:
    s = df[f]
    mean_abs = s.abs().mean()
    std_v    = s.std()
    cv_norm  = std_v / (mean_abs + 1e-10)
    nzv_rows.append({
        "feature":    f,
        "grupo":      grupo(f),
        "media":      round(s.mean(), 4),
        "std":        round(std_v, 4),
        "cv_norm":    round(cv_norm, 4),
        "n_unicos":   s.nunique(),
        "nzv_flag":   cv_norm < NZV_CV_THRESHOLD,
    })

nzv_df = pd.DataFrame(nzv_rows).set_index("feature")
flagged = nzv_df[nzv_df["nzv_flag"]]

print(tabulate(
    nzv_df[["grupo", "media", "std", "cv_norm", "n_unicos", "nzv_flag"]]
          .sort_values("cv_norm"),
    headers="keys", tablefmt="simple", floatfmt=".4f"
))

if flagged.empty:
    print("\n  Ningún feature eliminado por near-zero variance.")
else:
    print(f"\n  ⚠  Eliminados por NZV: {flagged.index.tolist()}")

features_v1 = [f for f in available if not nzv_df.loc[f, "nzv_flag"]]
print(f"\n  Features que continúan al análisis: {len(features_v1)} de {len(available)}")

save_csv(nzv_df, "01_nzv")

# ────────────────────────────────────────────────────────────────
# 2. VIF — MULTICOLINEALIDAD
# ────────────────────────────────────────────────────────────────

section("ETAPA 2 — VIF (Variance Inflation Factor)")
print("  Threshold: VIF > 10 = alta colinealidad | VIF 5-10 = revisar")

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(df[features_v1]),
    columns=features_v1
)

vif_rows = []
for i, f in enumerate(features_v1):
    try:
        v = variance_inflation_factor(X_scaled.values, i)
    except Exception:
        v = float("nan")
    vif_rows.append({"feature": f, "grupo": grupo(f), "VIF": round(v, 2)})

vif_df = pd.DataFrame(vif_rows).set_index("feature").sort_values("VIF", ascending=False)
vif_df["estado"] = vif_df["VIF"].apply(
    lambda v: "ALTO  >" if v > 10 else ("MEDIO >" if v > 5 else "OK")
)

print(tabulate(vif_df, headers="keys", tablefmt="simple", floatfmt=".2f"))
print(f"\n  Nota: no se elimina aquí — MRMR elige cuál conservar cuando hay colinealidad.")

save_csv(vif_df, "02_vif")

# ────────────────────────────────────────────────────────────────
# 3. PCA — ESTRUCTURA LATENTE
# ────────────────────────────────────────────────────────────────

section("ETAPA 3 — PCA (estructura latente — diagnóstico, no transformación)")

X_pca_input = scaler.fit_transform(df[features_v1])
pca = PCA()
pca.fit(X_pca_input)

exp_var  = pca.explained_variance_ratio_
cum_var  = np.cumsum(exp_var)
n_85     = int(np.searchsorted(cum_var, 0.85)) + 1
n_95     = int(np.searchsorted(cum_var, 0.95)) + 1

print(f"  Features analizados: {len(features_v1)}")
print(f"  Componentes para 85% varianza: {n_85}")
print(f"  Componentes para 95% varianza: {n_95}")

if n_85 <= 3:
    print(f"  → SEÑAL FUERTE: {len(features_v1)} features son básicamente {n_85} señales latentes")
elif n_85 <= len(features_v1) // 2:
    print(f"  → Alta redundancia — se puede comprimir a ~{n_85} features sin perder mucho")
else:
    print(f"  → Buena independencia entre features")

# Loadings por componente (top 5 cada uno)
subsection("Loadings por componente (top 5 features con mayor peso)")
n_pc_show = min(6, len(features_v1))
loadings = pd.DataFrame(
    pca.components_[:n_pc_show].T,
    index=features_v1,
    columns=[f"PC{i+1}" for i in range(n_pc_show)]
)
for pc_i in range(n_pc_show):
    pc_col = f"PC{pc_i+1}"
    top5   = loadings[pc_col].abs().nlargest(5)
    pct_v  = exp_var[pc_i] * 100
    print(f"\n  {pc_col}  ({pct_v:.1f}% varianza):")
    for feat, _ in top5.items():
        val  = loadings.loc[feat, pc_col]
        sign = "+" if val > 0 else "-"
        print(f"    {sign}{abs(val):.3f}  {feat}  [{grupo(feat)}]")

# — Scree plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
xrange = range(1, len(exp_var) + 1)
axes[0].bar(xrange, exp_var * 100, color="steelblue", alpha=0.75)
axes[0].set_xlabel("Componente")
axes[0].set_ylabel("% Varianza explicada")
axes[0].set_title("Scree Plot")
axes[0].set_xlim(0.3, min(len(exp_var) + 0.7, 16))

axes[1].plot(xrange, cum_var * 100, "o-", color="steelblue", linewidth=2)
axes[1].axhline(85, color="orange", linestyle="--", label="85%")
axes[1].axhline(95, color="red",    linestyle="--", label="95%")
axes[1].axvline(n_85, color="orange", linestyle=":", alpha=0.6)
axes[1].axvline(n_95, color="red",    linestyle=":", alpha=0.6)
axes[1].set_xlabel("N° componentes")
axes[1].set_ylabel("% Varianza acumulada")
axes[1].set_title("Varianza Acumulada")
axes[1].legend()
axes[1].set_xlim(0.3, min(len(exp_var) + 0.7, 16))
plt.suptitle("PCA — Estructura latente de features", fontsize=13, fontweight="bold")
plt.tight_layout()
save_plot(fig, "03_pca_scree")

# — Heatmap loadings
fig2, ax2 = plt.subplots(figsize=(11, max(6, len(features_v1) * 0.38)))
sns.heatmap(
    loadings.round(3), annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, linewidths=0.4, ax=ax2,
    cbar_kws={"shrink": 0.7}
)
ax2.set_title("PCA Loadings — Contribución de cada feature a cada componente")
plt.tight_layout()
save_plot(fig2, "03_pca_loadings")

save_csv(loadings, "03_pca_loadings")

# ────────────────────────────────────────────────────────────────
# 4. MUTUAL INFORMATION
# ────────────────────────────────────────────────────────────────

section("ETAPA 4 — Mutual Information vs target (log-transformado)")

mi_scores = mutual_info_regression(
    df[features_v1].values,
    y_log,
    n_neighbors=5,
    random_state=42
)

mi_df = pd.DataFrame({
    "feature":  features_v1,
    "grupo":    [grupo(f) for f in features_v1],
    "MI_score": mi_scores.round(4),
}).set_index("feature").sort_values("MI_score", ascending=False)
mi_df["rank_MI"] = range(1, len(mi_df) + 1)

print(tabulate(mi_df, headers="keys", tablefmt="simple", floatfmt=".4f"))

# — Plot MI
fig, ax = plt.subplots(figsize=(10, max(5, len(features_v1) * 0.38)))
colors = ["#2196F3" if g == "funnel" else "#FF9800" for g in mi_df["grupo"]]
ax.barh(mi_df.index[::-1], mi_df["MI_score"][::-1], color=colors[::-1], alpha=0.85)
ax.set_xlabel("Mutual Information Score")
ax.set_title(f"Mutual Information — features vs {TARGET} (log)")
ax.axvline(mi_df["MI_score"].median(), color="gray", linestyle="--",
           alpha=0.7, label="mediana")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="#2196F3", label="Funnel"),
    Patch(facecolor="#FF9800", label="Macro"),
    plt.Line2D([0], [0], color="gray", linestyle="--", label="mediana"),
])
plt.tight_layout()
save_plot(fig, "04_mutual_information")

save_csv(mi_df, "04_mutual_information")

# ────────────────────────────────────────────────────────────────
# 5. MRMR
# ────────────────────────────────────────────────────────────────

section("ETAPA 5 — MRMR (Maximum Relevance, Minimum Redundancy)")

mrmr_df = None
try:
    from mrmr import mrmr_regression as mrmr_fn

    k = min(MRMR_K, len(features_v1))
    selected_mrmr = mrmr_fn(
        X=df[features_v1].reset_index(drop=True),
        y=pd.Series(y_log),
        K=k
    )

    mrmr_df = pd.DataFrame({
        "mrmr_rank": range(1, len(selected_mrmr) + 1),
        "feature":   selected_mrmr,
        "grupo":     [grupo(f) for f in selected_mrmr],
    }).set_index("feature")
    mrmr_df = mrmr_df.join(mi_df[["MI_score", "rank_MI"]])

    print(f"  Top {k} features (orden = relevancia − redundancia acumulada):\n")
    print(tabulate(mrmr_df, headers="keys", tablefmt="simple"))

    fuera = [f for f in features_v1 if f not in selected_mrmr]
    print(f"\n  Fuera del top {k}: {fuera}")

    save_csv(mrmr_df, "05_mrmr")

except ImportError:
    print("  ✗ mrmr-selection no encontrado — instalá con:")
    print("      pip install mrmr-selection")
    print("  Etapa saltada.")

# ────────────────────────────────────────────────────────────────
# 6. GRANGER CAUSALITY
# ────────────────────────────────────────────────────────────────

section("ETAPA 6 — Granger Causality (lags 1–3 semanas)")
print(f"  H₀: X no Granger-causa {TARGET}")
print(f"  p < 0.05 → X tiene poder predictivo temporal sobre {TARGET}\n")

gc_rows = []
for f in features_v1:
    feat_s   = df[f].ffill().bfill()
    target_s = df[TARGET].ffill().bfill()
    data_gc  = pd.concat([target_s, feat_s], axis=1).dropna().values

    row = {"feature": f, "grupo": grupo(f)}
    try:
        res = grangercausalitytests(data_gc, maxlag=GRANGER_MAX_LAG, verbose=False)
        for lag in range(1, GRANGER_MAX_LAG + 1):
            row[f"p_lag{lag}"] = round(res[lag][0]["ssr_ftest"][1], 4)
        row["min_p"] = round(min(row[f"p_lag{l}"] for l in range(1, GRANGER_MAX_LAG + 1)), 4)
        row["sig"]   = row["min_p"] < 0.05
    except Exception as e:
        for lag in range(1, GRANGER_MAX_LAG + 1):
            row[f"p_lag{lag}"] = None
        row["min_p"] = None
        row["sig"]   = False

    gc_rows.append(row)

gc_df = pd.DataFrame(gc_rows).set_index("feature").sort_values("min_p")

print(tabulate(
    gc_df[["grupo", "p_lag1", "p_lag2", "p_lag3", "min_p", "sig"]],
    headers="keys", tablefmt="simple", floatfmt=".4f"
))

sig_list = gc_df[gc_df["sig"]].index.tolist()
print(f"\n  Granger significativo (p < 0.05): {sig_list}")

save_csv(gc_df, "06_granger")

# ────────────────────────────────────────────────────────────────
# 7. ESTABILIDAD EN WALK-FORWARD CV
# ────────────────────────────────────────────────────────────────

section("ETAPA 7 — Estabilidad en walk-forward CV")

n          = len(df)
fold_size  = n // CV_N_FOLDS
min_train  = fold_size * 2   # mínimo 2 bloques de historia
n_folds_run = CV_N_FOLDS - 2  # número de folds reales que corren

print(f"  {n} semanas totales  |  {CV_N_FOLDS} bloques  |  "
      f"mínimo de entrenamiento: {min_train} semanas\n")

stability = {f: [] for f in features_v1}
mediana_mi_threshold = mi_df["MI_score"].median()

for fold_i in range(n_folds_run):
    train_end = min_train + fold_i * fold_size
    X_fold = df[features_v1].iloc[:train_end].values
    y_fold = np.log1p(df[TARGET].iloc[:train_end].values)

    mi_fold = mutual_info_regression(X_fold, y_fold, random_state=42)
    mi_fold_s = pd.Series(mi_fold, index=features_v1)
    top_half  = set(mi_fold_s.nlargest(len(features_v1) // 2).index)

    for f in features_v1:
        stability[f].append(1 if f in top_half else 0)

    top5_str = ", ".join(mi_fold_s.nlargest(5).index.tolist())
    print(f"  Fold {fold_i+1}  ({train_end} sem.)  top-5 MI: {top5_str}")

stab_df = pd.DataFrame([
    {
        "feature":   f,
        "grupo":     grupo(f),
        "n_folds":   len(v),
        "seleccionado": sum(v),
        "pct":       round(sum(v) / len(v) * 100, 1) if v else 0,
    }
    for f, v in stability.items()
]).set_index("feature").sort_values("pct", ascending=False)

stab_df["estable"] = stab_df["pct"] >= STABILITY_THRESH * 100

print(f"\n  Threshold estabilidad: ≥ {int(STABILITY_THRESH * 100)}%\n")
print(tabulate(stab_df, headers="keys", tablefmt="simple"))

save_csv(stab_df, "07_stability")

# ────────────────────────────────────────────────────────────────
# 8. TABLA RESUMEN DE DECISIÓN
# ────────────────────────────────────────────────────────────────

section("ETAPA 8 — Tabla resumen de decisión")

summary = pd.DataFrame({"grupo": [grupo(f) for f in features_v1]}, index=features_v1)

# VIF
summary["VIF"]    = vif_df["VIF"]
summary["vif_ok"] = vif_df["VIF"] <= 10

# MI
summary["MI"]     = mi_df["MI_score"]
summary["mi_ok"]  = mi_df["MI_score"] >= mi_df["MI_score"].median()

# MRMR
if mrmr_df is not None:
    mrmr_set = set(mrmr_df.index)
    summary["mrmr_rank"] = [mrmr_df.loc[f, "mrmr_rank"] if f in mrmr_set else None
                             for f in features_v1]
    summary["mrmr_ok"]   = [f in mrmr_set for f in features_v1]
else:
    summary["mrmr_ok"] = None

# Granger
summary["granger_p"]  = gc_df["min_p"]
summary["granger_ok"] = gc_df["sig"]

# Estabilidad
summary["estab_pct"] = stab_df["pct"]
summary["estab_ok"]  = stab_df["estable"]

# Score — cuántos criterios pasa
criteria = ["vif_ok", "mi_ok", "granger_ok", "estab_ok"]
if mrmr_df is not None:
    criteria.append("mrmr_ok")

summary["n_criterios"] = summary[criteria].sum(axis=1)
n_total = len(criteria)

summary["decision"] = summary["n_criterios"].apply(
    lambda x: "INCLUIR" if x >= n_total - 1
              else ("REVISAR" if x >= n_total - 2 else "EXCLUIR")
)

summary = summary.sort_values(["decision", "n_criterios"], ascending=[True, False])

# Mostrar
cols_show = ["grupo", "VIF", "MI", "granger_p", "estab_pct", "n_criterios", "decision"]
if mrmr_df is not None:
    cols_show.insert(4, "mrmr_rank")

print(f"  Criterios: {criteria}  (total: {n_total})")
print(f"  INCLUIR si pasa ≥ {n_total - 1} | REVISAR si pasa ≥ {n_total - 2}\n")
print(tabulate(summary[cols_show], headers="keys", tablefmt="simple", floatfmt=".3f"))

inc  = summary[summary["decision"] == "INCLUIR"].index.tolist()
rev  = summary[summary["decision"] == "REVISAR"].index.tolist()
exc  = summary[summary["decision"] == "EXCLUIR"].index.tolist()

print(f"\n  ✅ INCLUIR  ({len(inc)}): {inc}")
print(f"  🟡 REVISAR  ({len(rev)}): {rev}")
print(f"  ❌ EXCLUIR  ({len(exc)}): {exc}")

save_csv(summary, "08_decision_table")

# ────────────────────────────────────────────────────────────────
# FIN
# ────────────────────────────────────────────────────────────────

section("ANÁLISIS COMPLETO")
print(f"  Campaña  : Primer Depósito")
print(f"  Target   : {TARGET}  (log-transformado)")
print(f"  Dataset  : {len(df)} semanas")
print(f"  Resultados: {RESULTS_DIR}/")
print(f"\n  Archivos generados:")
for p in sorted(RESULTS_DIR.glob("*")):
    print(f"    {p.name}")
print()
