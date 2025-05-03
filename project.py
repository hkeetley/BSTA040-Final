import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data(path="ilidata.csv"):
    df = pd.read_csv(path)
    # create a zero‐based week index
    df["weeks"] = np.arange(len(df))
    return df

df = load_data()

# Select state 
states = sorted(df["state"].unique())
selected_state = st.selectbox(
    "Select a state:", 
    states, 
    key="state_select"
)

# Filter data
state_data = df[df["state"] == selected_state]

# Time series plot
st.markdown("### ILI % Over Time")
st.line_chart(
    state_data.set_index("weeks")["ili"],
    height=300
)

# Histogra,
st.markdown("### Histogram of ILI % with Exponential Density Overlay")

# Extract values and estimate 
ili_vals = state_data["ili"].dropna().values
lambda_hat = 1.0 / np.mean(ili_vals)

# Plot
fig, ax = plt.subplots()
ax.hist(
    ili_vals,
    bins=30,
    density=True,
    alpha=0.6,
    color="salmon",
    edgecolor="black",
    label="ILI Percent"
)
x = np.linspace(0, ili_vals.max(), 200)
exp_pdf = lambda_hat * np.exp(-lambda_hat * x)
ax.plot(
    x,
    exp_pdf,
    linestyle="--",
    linewidth=2,
    label=f"Exp PDF"
)
ax.set_xlabel("ILI %")
ax.set_ylabel("Density")
ax.legend()

st.pyplot(fig)
