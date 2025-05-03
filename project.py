import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load date
@st.cache_data
def load_data(path="ilidata.csv"):
    df = pd.read_csv(path)
    df["weeks"] = np.arange(len(df))
    return df

df = load_data()

# Select State
st.sidebar.header("Configuration")
states = sorted(df["state"].unique())
selected_state = st.sidebar.selectbox(
    "Select a state to analyze:", 
    states, 
    key="state_select"
)
state_data = df[df["state"] == selected_state]

# Time series plot
st.header(f"Time Series of Influenza Like Illness (ILI) % for {selected_state}")
st.markdown(
    r"""
    This plot shows the weekly percentage of influenza-like illness (ILI)
    visits for the selected state. The horizontal axis (“Week Index”) counts 
    weeks starting at 0 (the week the data collection began); the vertical axis 
    (“ILI %”) indicates what percentage of total patient visits were for ILI 
    in that week.
    """
)

fig_ts, ax_ts = plt.subplots()
ax_ts.plot(
    state_data["weeks"],
    state_data["ili"],
    color="salmon",
    linewidth=2,
    label="ILI % over time"
)
ax_ts.set_xlabel("Week Index")
ax_ts.set_ylabel("ILI (%)")
ax_ts.set_title(f"ILI Time Series")
ax_ts.legend()
st.pyplot(fig_ts)


# Histogram
st.header("Histogram of ILI % with Exponential Density Fit")
st.markdown(
    r"""
    Here we examine the distribution of weekly ILI percentages for the chosen state.
    The histogram shows the density of ILI % values (area = 1).
    We overlay the estimated exponetial density Exponential($\lambda$) density (dashed line), where λ is estimated as: $$\hat\lambda = \frac{1}{\bar y}$$, and $\bar y$ is the sample mean of ILI %.
    """
)

ili_vals = state_data["ili"].dropna().values
lambda_hat = 1.0 / np.mean(ili_vals)

# Plot
fig_hist, ax_hist = plt.subplots()
ax_hist.hist(
    ili_vals,
    bins=30,
    density=True,
    color="salmon",    
    edgecolor="#1F77B4", 
    alpha=0.75,
    label="Empirical ILI % density"
)

x = np.linspace(0, ili_vals.max(), 200)
exp_pdf = lambda_hat * np.exp(-lambda_hat * x)
ax_hist.plot(
    x,
    exp_pdf,
    color="blue",     
    linewidth=2,
    label=f"Exp PDF"
)

ax_hist.set_xlabel("ILI (%)")
ax_hist.set_ylabel("Density")
ax_hist.set_title(f"ILI % Distribution & Exponential Fit")
ax_hist.legend()
st.pyplot(fig_hist)
