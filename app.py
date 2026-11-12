"""
Sistema de modelamiento de compostaje (versión Python / Streamlit)

Réplica de la aplicación MATLAB del informe de laboratorio
"Caracterización de residuos sólidos y evaluación del proceso de compostaje".

Ejecutar con:  streamlit run app.py
"""

from datetime import date, datetime, time
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================================
# CONSTANTES DEL EXPERIMENTO (tomadas del informe)
# ==========================================================

FECHA_INICIO = date(2026, 10, 8)
ARCHIVO_DATOS = Path(__file__).parent / "datos_compostaje.csv"

MASA_BOLSAS = {"Bolsa 1": 3000.0, "Bolsa 2": 8500.0}
MASA_COMPOSTAJE = 1250.0  # g de biodegradables incorporados al balde

RESIDUOS = pd.DataFrame(
    {
        "Material": ["Biodegradables", "Plástico", "Cartón", "Icopor", "Aluminio"],
        "Masa (g)": [5000.0, 1500.0, 153.5, 53.7, 7.4],
    }
)

DATOS_INICIALES = pd.DataFrame(
    {
        "Fecha": ["2026-10-15", "2026-10-21", "2026-10-23",
                  "2026-10-27", "2026-10-29", "2026-11-03"],
        "Hora": ["12:30", "18:30", "18:30", "19:00", "20:00", "20:00"],
        "Temperatura": [34.0, 18.0, 22.0, 18.0, 23.0, 20.0],
        "Observaciones": ["Registro experimental inicial"] * 6,
    }
)

# Paleta pensada para compostaje: verde hoja, tierra, ámbar, gris pizarra
VERDE = "#2E6B45"
TIERRA = "#7A5230"
AMBAR = "#D9902F"
PIZARRA = "#4A5A68"
COLORES_RESIDUOS = [VERDE, "#3B7FB8", AMBAR, "#9A6FB0", "#B5483A"]

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "font.size": 10,
    }
)

# ==========================================================
# DATOS: carga, normalización y guardado
# ==========================================================


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el día a partir de la fecha y ordena cronológicamente."""
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.strftime("%Y-%m-%d")
    df["Temperatura"] = df["Temperatura"].astype(float)
    df["Observaciones"] = df["Observaciones"].fillna("").astype(str)
    inicio = pd.Timestamp(FECHA_INICIO)
    df["Dia"] = (pd.to_datetime(df["Fecha"]) - inicio).dt.days.astype(int)
    df = df.sort_values(["Fecha", "Hora"]).reset_index(drop=True)
    return df[["Fecha", "Hora", "Dia", "Temperatura", "Observaciones"]]


def cargar_datos() -> pd.DataFrame:
    if ARCHIVO_DATOS.exists():
        df = pd.read_csv(ARCHIVO_DATOS, dtype={"Fecha": str, "Hora": str})
    else:
        df = DATOS_INICIALES.copy()
    df = normalizar(df)
    df.to_csv(ARCHIVO_DATOS, index=False)
    return df


def guardar_datos(df: pd.DataFrame) -> None:
    st.session_state.datos = normalizar(df)
    st.session_state.datos.to_csv(ARCHIVO_DATOS, index=False)


# ==========================================================
# CÁLCULOS ESTADÍSTICOS
# ==========================================================


def composicion() -> pd.DataFrame:
    """Masa y participación porcentual de cada fracción (modelo 5.4.1)."""
    df = RESIDUOS.copy()
    total = df["Masa (g)"].sum()
    df["Participación (%)"] = df["Masa (g)"] / total * 100
    return df


def cuartiles(valores) -> tuple[float, float, float]:
    """Cuartiles por mitades de los datos ordenados (criterio del informe)."""
    s = np.sort(np.asarray(valores, dtype=float))
    n = len(s)
    mitad = n // 2
    inferior = s[:mitad]
    superior = s[mitad + n % 2:]
    return float(np.median(inferior)), float(np.median(s)), float(np.median(superior))


def estadisticos(df: pd.DataFrame) -> pd.DataFrame:
    T = df["Temperatura"].to_numpy()
    q1, med, q3 = cuartiles(T)
    filas = [
        ("Número de mediciones", f"{len(T)}"),
        ("Temperatura media", f"{T.mean():.2f} °C"),
        ("Mediana", f"{med:.2f} °C"),
        ("Temperatura mínima", f"{T.min():.2f} °C"),
        ("Temperatura máxima", f"{T.max():.2f} °C"),
        ("Rango", f"{T.max() - T.min():.2f} °C"),
        ("Desviación estándar muestral",
         f"{T.std(ddof=1):.2f} °C" if len(T) > 1 else "—"),
        ("Primer cuartil", f"{q1:.2f} °C"),
        ("Tercer cuartil", f"{q3:.2f} °C"),
        ("Rango intercuartílico", f"{q3 - q1:.2f} °C"),
    ]
    return pd.DataFrame(filas, columns=["Indicador", "Resultado"])


def ajustar_modelo(df: pd.DataFrame):
    """Regresión lineal por mínimos cuadrados (equivalente a polyfit)."""
    if len(df) < 3 or df["Dia"].nunique() < 2:
        return None
    t = df["Dia"].to_numpy(dtype=float)
    T = df["Temperatura"].to_numpy(dtype=float)
    m, b = np.polyfit(t, T, 1)
    estimada = m * t + b
    residuos = T - estimada
    ss_res = float(np.sum(residuos**2))
    ss_tot = float(np.sum((T - T.mean()) ** 2))
    return {
        "pendiente": float(m),
        "intercepto": float(b),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mae": float(np.mean(np.abs(residuos))),
        "rmse": float(np.sqrt(np.mean(residuos**2))),
        "estimada": estimada,
        "residuos": residuos,
    }


def tabla_ajuste(df: pd.DataFrame, modelo: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Día": df["Dia"],
            "Real (°C)": df["Temperatura"],
            "Estimada (°C)": modelo["estimada"].round(4),
            "Residuo (°C)": modelo["residuos"].round(4),
        }
    )


# ==========================================================
# GRÁFICAS
# ==========================================================


def grafica_barras_porcentaje():
    d = composicion()
    fig, ax = plt.subplots(figsize=(7, 4))
    barras = ax.bar(d["Material"], d["Participación (%)"], color=COLORES_RESIDUOS)
    for barra, v in zip(barras, d["Participación (%)"]):
        ax.text(barra.get_x() + barra.get_width() / 2, v + 1, f"{v:.2f}%",
                ha="center", fontsize=9)
    ax.set_title("Composición porcentual de los residuos")
    ax.set_ylabel("Participación (%)")
    ax.set_ylim(0, 85)
    ax.grid(axis="x", visible=False)
    return fig


def grafica_torta():
    d = composicion()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(d["Masa (g)"], colors=COLORES_RESIDUOS, startangle=90, counterclock=False,
           wedgeprops={"edgecolor": "white", "linewidth": 1})
    etiquetas = [f"{m}: {p:.2f}%" for m, p in zip(d["Material"], d["Participación (%)"])]
    ax.legend(etiquetas, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.set_title("Distribución de residuos caracterizados")
    return fig


def grafica_pareto():
    d = composicion().sort_values("Masa (g)", ascending=False).reset_index(drop=True)
    acumulado = d["Masa (g)"].cumsum() / d["Masa (g)"].sum() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(d["Material"], d["Masa (g)"], color=VERDE)
    ax.set_ylabel("Masa (g)", color=VERDE)
    ax.grid(axis="x", visible=False)
    ax2 = ax.twinx()
    ax2.plot(d["Material"], acumulado, marker="o", color=AMBAR, linewidth=2)
    ax2.set_ylabel("Porcentaje acumulado (%)", color=AMBAR)
    ax2.set_ylim(0, 110)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.set_title("Diagrama de Pareto de los residuos")
    return fig


def grafica_balance():
    caracterizada = RESIDUOS["Masa (g)"].sum()
    total = sum(MASA_BOLSAS.values())
    diferencia = total - caracterizada
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.bar(["Masa inicial recolectada"], [caracterizada], color=VERDE,
           label="Masa de fracciones caracterizadas")
    ax.bar(["Masa inicial recolectada"], [diferencia], bottom=[caracterizada],
           color=AMBAR, label="Diferencia por conciliar")
    ax.text(0, caracterizada / 2, f"{caracterizada:,.1f} g", ha="center",
            color="white", fontweight="bold")
    ax.text(0, caracterizada + diferencia / 2, f"{diferencia:,.1f} g", ha="center",
            fontweight="bold")
    ax.set_ylabel("Masa (g)")
    ax.set_title("Balance aritmético de masas")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.grid(axis="x", visible=False)
    return fig


def grafica_aprovechamiento():
    biodeg = float(RESIDUOS.loc[RESIDUOS["Material"] == "Biodegradables", "Masa (g)"].iloc[0])
    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.bar(["Biodegradable\ncaracterizada", "Incorporada\nal compostaje"],
                    [biodeg, MASA_COMPOSTAJE], color=[VERDE, TIERRA])
    for b in barras:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 60,
                f"{b.get_height():,.1f} g", ha="center")
    ax.set_ylabel("Masa (g)")
    ax.set_title("Aprovechamiento de la fracción biodegradable")
    ax.grid(axis="x", visible=False)
    return fig


def grafica_temperatura(df: pd.DataFrame, modelo=None):
    d = df.sort_values("Dia")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(d["Dia"], d["Temperatura"], marker="o", markersize=8, linewidth=1.8,
            color=VERDE, label="Temperatura registrada")
    for x, y in zip(d["Dia"], d["Temperatura"]):
        ax.annotate(f"{y:.0f} °C", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)
    if modelo is not None:
        x = np.linspace(d["Dia"].min(), d["Dia"].max(), 100)
        ax.plot(x, modelo["pendiente"] * x + modelo["intercepto"], "--",
                color=AMBAR, linewidth=2, label="Tendencia lineal")
    ax.set_title("Comportamiento térmico del compostaje")
    ax.set_xlabel("Días transcurridos desde el inicio (8/10/2026)")
    ax.set_ylabel("Temperatura (°C)")
    ax.legend(frameon=False)
    return fig


def grafica_regresion(df: pd.DataFrame, modelo: dict):
    d = df.sort_values("Dia")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(d["Dia"], d["Temperatura"], s=70, color=VERDE,
               label="Temperaturas experimentales", zorder=3)
    x = np.linspace(d["Dia"].min(), d["Dia"].max(), 100)
    ax.plot(x, modelo["pendiente"] * x + modelo["intercepto"], color=AMBAR,
            linewidth=2.2, label="Tendencia estadística")
    ax.set_title("Regresión lineal de la temperatura del compostaje")
    ax.set_xlabel("Días transcurridos")
    ax.set_ylabel("Temperatura (°C)")
    ax.text(0.97, 0.95,
            f"T(t) = {modelo['intercepto']:.4f} {modelo['pendiente']:+.4f}t\n"
            f"R² = {modelo['r2']:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc"})
    ax.legend(frameon=False, loc="lower left")
    return fig


def grafica_caja(df: pd.DataFrame):
    T = df["Temperatura"].to_numpy(dtype=float)
    q1, med, q3 = cuartiles(T)
    iqr = q3 - q1
    dentro = T[(T >= q1 - 1.5 * iqr) & (T <= q3 + 1.5 * iqr)]
    atipicos = T[(T < q1 - 1.5 * iqr) | (T > q3 + 1.5 * iqr)]
    stats = [{"med": med, "q1": q1, "q3": q3, "whislo": dentro.min(),
              "whishi": dentro.max(), "fliers": atipicos}]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.bxp(stats, widths=0.5, patch_artist=True, showfliers=True,
           boxprops={"facecolor": "#BFD8C6", "edgecolor": PIZARRA},
           medianprops={"color": "black", "linewidth": 2},
           flierprops={"marker": "o", "markerfacecolor": "#3B7FB8",
                       "markeredgecolor": "#3B7FB8", "markersize": 8})
    ax.set_xticks([1], ["Temperatura"])
    ax.set_ylabel("Temperatura (°C)")
    ax.set_title("Distribución de temperaturas")
    ax.grid(axis="x", visible=False)
    return fig


def grafica_residuos(df: pd.DataFrame, modelo: dict):
    d = df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.stem(d["Dia"], modelo["residuos"], basefmt=" ")
    ax.set_title("Análisis de residuos de la regresión lineal")
    ax.set_xlabel("Días transcurridos")
    ax.set_ylabel("Residuo (°C)")
    return fig


def mostrar(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# ==========================================================
# EXPORTACIÓN
# ==========================================================


def exportar_excel(df: pd.DataFrame, modelo) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="Mediciones", index=False)
        composicion().to_excel(xl, sheet_name="Residuos", index=False)
        estadisticos(df).to_excel(xl, sheet_name="Estadisticos", index=False)
        if modelo is not None:
            tabla_ajuste(df, modelo).to_excel(xl, sheet_name="Ajuste lineal", index=False)
    return buf.getvalue()


def exportar_pdf(df: pd.DataFrame, modelo) -> bytes:
    buf = BytesIO()
    caracterizada = RESIDUOS["Masa (g)"].sum()
    total = sum(MASA_BOLSAS.values())
    with PdfPages(buf) as pdf:
        # Página 1: resumen
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "Informe de seguimiento del compostaje", fontsize=18,
                 fontweight="bold")
        fig.text(0.08, 0.91, f"Inicio del experimento: {FECHA_INICIO:%d/%m/%Y} | "
                 f"Generado: {datetime.now():%d/%m/%Y %H:%M}", fontsize=9)
        lineas = [
            f"Masa inicial recolectada: {total:,.1f} g",
            f"Masa caracterizada: {caracterizada:,.1f} g",
            f"Diferencia por conciliar: {total - caracterizada:,.1f} g",
            f"Biodegradables incorporados al compostaje: {MASA_COMPOSTAJE:,.1f} g",
            "",
            *[f"{r.Indicador}: {r.Resultado}" for r in estadisticos(df).itertuples()],
        ]
        if modelo is not None:
            lineas += [
                "",
                f"T(t) = {modelo['intercepto']:.4f} {modelo['pendiente']:+.4f} t",
                f"R² = {modelo['r2']:.4f} | MAE = {modelo['mae']:.2f} °C | "
                f"RMSE = {modelo['rmse']:.2f} °C",
            ]
        lineas += [
            "",
            "Nota: el modelo lineal es exploratorio. Con pocas mediciones y sin",
            "control de condiciones ambientales no permite predecir el proceso.",
        ]
        fig.text(0.08, 0.86, "\n".join(lineas), fontsize=10, va="top", linespacing=1.6)
        pdf.savefig(fig)
        plt.close(fig)

        # Páginas de gráficas
        figuras = [grafica_temperatura(df, modelo)]
        if modelo is not None:
            figuras += [grafica_regresion(df, modelo), grafica_residuos(df, modelo)]
        figuras += [grafica_caja(df), grafica_pareto(), grafica_balance()]
        for f in figuras:
            pdf.savefig(f, bbox_inches="tight")
            plt.close(f)

        # Tabla de mediciones
        fig, ax = plt.subplots(figsize=(8.27, 5))
        ax.axis("off")
        tabla = ax.table(cellText=df.values.tolist(), colLabels=list(df.columns),
                         loc="upper center", cellLoc="left")
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(8)
        tabla.scale(1, 1.4)
        ax.set_title("Historial de mediciones")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return buf.getvalue()


# ==========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(page_title="Seguimiento de compostaje", page_icon="🌱", layout="wide")

st.markdown(
    f"""
    <style>
    [data-testid="stMetric"] {{
        background: #F3F6F2;
        border-left: 4px solid {VERDE};
        padding: 0.6rem 0.9rem;
        border-radius: 4px;
    }}
    [data-testid="stMetricValue"] {{ color: {VERDE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌱 Seguimiento y modelamiento del compostaje")
st.caption("Caracterización de residuos y análisis térmico · Universidad Cooperativa de Colombia")

if "datos" not in st.session_state:
    st.session_state.datos = cargar_datos()

df = st.session_state.datos
modelo = ajustar_modelo(df)

# ==========================================================
# MENÚ LATERAL
# ==========================================================

st.sidebar.title("Menú")
seccion = st.sidebar.radio(
    "Sección",
    [
        "Dashboard",
        "Caracterización de residuos",
        "Comportamiento térmico",
        "Datos y estadísticos",
        "Nueva medición",
        "Historial",
        "Exportar informes",
    ],
)
st.sidebar.divider()
st.sidebar.caption(f"Inicio del experimento: {FECHA_INICIO:%d/%m/%Y}")
st.sidebar.caption(f"Mediciones registradas: {len(df)}")

# ==========================================================
# DASHBOARD
# ==========================================================

if seccion == "Dashboard":
    caracterizada = RESIDUOS["Masa (g)"].sum()
    pct_bio = RESIDUOS["Masa (g)"].iloc[0] / caracterizada * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Masa caracterizada", f"{caracterizada:,.1f} g")
    c2.metric("Fracción biodegradable", f"{pct_bio:.2f} %")
    c3.metric("Masa orgánica incorporada", f"{MASA_COMPOSTAJE:,.0f} g")
    c4.metric("Temperatura promedio", f"{df['Temperatura'].mean():.2f} °C")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mediciones", len(df))
    c2.metric("Temperatura media", f"{df['Temperatura'].mean():.1f} °C")
    c3.metric("Temperatura mínima", f"{df['Temperatura'].min():.1f} °C")
    c4.metric("Temperatura máxima", f"{df['Temperatura'].max():.1f} °C")

    st.divider()
    izq, der = st.columns([3, 2])
    with izq:
        mostrar(grafica_temperatura(df, modelo))
    with der:
        mostrar(grafica_torta())

    ultima = df.iloc[-1]
    st.caption(
        f"Última medición: {datetime.strptime(ultima['Fecha'], '%Y-%m-%d'):%d/%m/%Y} "
        f"{ultima['Hora']} · {ultima['Temperatura']:.1f} °C"
    )

# ==========================================================
# CARACTERIZACIÓN DE RESIDUOS
# ==========================================================

elif seccion == "Caracterización de residuos":
    st.header("Caracterización de residuos sólidos")

    caracterizada = RESIDUOS["Masa (g)"].sum()
    total = sum(MASA_BOLSAS.values())

    c1, c2, c3 = st.columns(3)
    c1.metric("Masa inicial recolectada", f"{total:,.0f} g")
    c2.metric("Masa caracterizada", f"{caracterizada:,.1f} g")
    c3.metric("Diferencia por conciliar", f"{total - caracterizada:,.1f} g")

    st.info(
        "Los porcentajes describen solo las fracciones pesadas (6.714,6 g), no los "
        "11.500 g originales: no se registraron las masas descartadas durante el cuarteo."
    )

    izq, der = st.columns(2)
    with izq:
        mostrar(grafica_barras_porcentaje())
    with der:
        mostrar(grafica_torta())

    izq, der = st.columns(2)
    with izq:
        mostrar(grafica_pareto())
    with der:
        mostrar(grafica_balance())

    st.subheader("Aprovechamiento de la fracción biodegradable")
    izq, der = st.columns([2, 3])
    with izq:
        mostrar(grafica_aprovechamiento())
    with der:
        st.write(
            f"De los 5.000 g de biodegradables se incorporaron **{MASA_COMPOSTAJE:,.0f} g** "
            f"al compostaje, es decir el **{MASA_COMPOSTAJE / 5000 * 100:.0f} %** "
            "de la fracción caracterizada."
        )
        st.write(
            "No se registraron las masas de tierra, aserrín ni el volumen de agua, "
            "por lo que no se puede calcular la relación C/N ni la reducción de masa."
        )

# ==========================================================
# COMPORTAMIENTO TÉRMICO
# ==========================================================

elif seccion == "Comportamiento térmico":
    st.header("Modelo estadístico del comportamiento térmico")

    mostrar(grafica_temperatura(df, modelo))

    if modelo is None:
        st.warning("Se necesitan al menos 3 mediciones en días distintos para ajustar la regresión.")
    else:
        st.subheader("Regresión lineal")
        st.latex(rf"T(t) = {modelo['intercepto']:.4f} {modelo['pendiente']:+.4f}\,t")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pendiente", f"{modelo['pendiente']:.4f} °C/día")
        c2.metric("R²", f"{modelo['r2']:.4f}")
        c3.metric("MAE", f"{modelo['mae']:.2f} °C")
        c4.metric("RMSE", f"{modelo['rmse']:.2f} °C")

        st.info(
            f"La relación lineal con el tiempo explica cerca del {modelo['r2'] * 100:.1f} % "
            "de la variabilidad. Es un análisis exploratorio, no una relación causal."
        )

        izq, der = st.columns(2)
        with izq:
            mostrar(grafica_regresion(df, modelo))
        with der:
            mostrar(grafica_residuos(df, modelo))

        st.subheader("Distribución de las temperaturas")
        izq, der = st.columns([2, 3])
        with izq:
            mostrar(grafica_caja(df))
        with der:
            q1, med, q3 = cuartiles(df["Temperatura"])
            iqr = q3 - q1
            atipicos = df.loc[
                (df["Temperatura"] > q3 + 1.5 * iqr) | (df["Temperatura"] < q1 - 1.5 * iqr),
                ["Fecha", "Hora", "Temperatura"],
            ]
            if atipicos.empty:
                st.write("Según el criterio de Tukey no hay observaciones atípicas.")
            else:
                st.write("Posibles observaciones atípicas (criterio de Tukey). "
                         "Se conservan en el análisis por ser mediciones experimentales:")
                st.dataframe(atipicos, hide_index=True)

        st.subheader("Estimación exploratoria")
        t_obs_min, t_obs_max = int(df["Dia"].min()), int(df["Dia"].max())
        dia = st.number_input("Día del proceso a consultar", min_value=0, max_value=60,
                              value=t_obs_max)
        estimada = modelo["pendiente"] * dia + modelo["intercepto"]
        st.metric("Temperatura estimada", f"{estimada:.2f} °C")
        if dia < t_obs_min or dia > t_obs_max:
            st.warning(
                f"El día {dia} está fuera del rango observado (días {t_obs_min} a {t_obs_max}). "
                "Es una extrapolación y no es confiable."
            )
        if estimada < 0:
            st.error("La estimación es negativa: el modelo lineal no tiene sentido físico fuera de los datos.")
        st.caption("Con pocas observaciones y condiciones no controladas, la ecuación no sirve para predecir el compostaje.")

# ==========================================================
# DATOS Y ESTADÍSTICOS
# ==========================================================

elif seccion == "Datos y estadísticos":
    st.header("Datos y estadísticos")

    izq, der = st.columns(2)
    with izq:
        st.subheader("Residuos caracterizados")
        tabla = composicion()
        st.dataframe(
            tabla.style.format({"Masa (g)": "{:,.1f}", "Participación (%)": "{:.2f}"}),
            hide_index=True,
        )
        st.caption(f"Total: {tabla['Masa (g)'].sum():,.1f} g")
    with der:
        st.subheader("Indicadores del seguimiento térmico")
        st.dataframe(estadisticos(df), hide_index=True)

    st.subheader("Valores reales, estimados y residuos")
    if modelo is None:
        st.warning("Se necesitan al menos 3 mediciones en días distintos.")
    else:
        st.dataframe(tabla_ajuste(df, modelo), hide_index=True)

# ==========================================================
# NUEVA MEDICIÓN
# ==========================================================

elif seccion == "Nueva medición":
    st.header("Registrar nueva medición")

    with st.form("form_medicion"):
        fecha = st.date_input("Fecha de medición", value=date.today(),
                              min_value=FECHA_INICIO, max_value=date.today(),
                              format="DD/MM/YYYY")
        hora = st.time_input("Hora", value=datetime.now().time().replace(second=0, microsecond=0),
                             step=60)
        temperatura = st.number_input("Temperatura (°C)", min_value=-10.0, max_value=100.0,
                                      value=20.0, step=0.5)
        observaciones = st.text_area(
            "Observaciones",
            placeholder="Condiciones ambientales o intervenciones realizadas (volteo, humedecimiento, adición de aserrín...)",
        )
        guardar = st.form_submit_button("Guardar medición", type="primary")

    dia_calculado = (fecha - FECHA_INICIO).days
    st.caption(f"Día del proceso: {dia_calculado} (se calcula desde el {FECHA_INICIO:%d/%m/%Y}).")
    st.caption("Registra también si antes de medir volteaste, humedeciste o añadiste material.")

    if guardar:
        hora_txt = hora.strftime("%H:%M")
        ya_existe = ((df["Fecha"] == str(fecha)) & (df["Hora"] == hora_txt)).any()
        if ya_existe:
            st.error("Ya existe una medición para esa fecha y hora.")
        else:
            nuevo = pd.DataFrame([{
                "Fecha": str(fecha),
                "Hora": hora_txt,
                "Temperatura": float(temperatura),
                "Observaciones": observaciones.strip(),
            }])
            guardar_datos(pd.concat([df, nuevo], ignore_index=True))
            st.success("Medición registrada.")
            st.rerun()

# ==========================================================
# HISTORIAL
# ==========================================================

elif seccion == "Historial":
    st.header("Historial de mediciones")

    st.dataframe(df, hide_index=True)

    if len(df) > 0:
        st.subheader("Eliminar un registro")
        etiquetas = [
            f"{datetime.strptime(r.Fecha, '%Y-%m-%d'):%d/%m/%Y} {r.Hora} · {r.Temperatura:.1f} °C"
            for r in df.itertuples()
        ]
        elegido = st.selectbox("Registro", range(len(df)), format_func=lambda i: etiquetas[i])
        confirmar = st.checkbox("Confirmo que quiero eliminar este registro")
        if st.button("Eliminar registro", disabled=not confirmar):
            guardar_datos(df.drop(index=elegido))
            st.success("Registro eliminado.")
            st.rerun()

# ==========================================================
# EXPORTAR
# ==========================================================

elif seccion == "Exportar informes":
    st.header("Exportación de resultados")
    st.write("Descarga las mediciones, los estadísticos y las gráficas para el informe.")

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Descargar CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="datos_compostaje.csv",
        mime="text/csv",
    )
    c2.download_button(
        "Descargar Excel",
        data=exportar_excel(df, modelo),
        file_name="compostaje.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    c3.download_button(
        "Descargar informe PDF",
        data=exportar_pdf(df, modelo),
        file_name="informe_compostaje.pdf",
        mime="application/pdf",
    )
    st.caption("El PDF incluye indicadores, gráficas, tabla de mediciones y una nota metodológica.")