import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Coin Flipping Simulation", layout="wide")


# Simulate unfair coin flipping
def simulate_coin_flips(prob_heads, num_flips, num_simulations):
    simulations = []
    for _ in range(num_simulations):
        flips = np.random.choice([0, 1], size=num_flips, p=[1 - prob_heads, prob_heads])
        cumulative_heads = np.cumsum(flips) / np.arange(1, num_flips + 1)
        simulations.append(cumulative_heads)
    return np.array(simulations)


st.title("Coin Flipping Simulation")

# Create input controls in a multi-column layout
col1, col2, col3 = st.columns(3)

with col1:
    prob_heads = st.number_input(
        "Probability of Heads (0 to 1)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

with col2:
    num_flips = st.number_input("Number of Flips", min_value=1, value=100)

with col3:
    num_simulations = st.number_input("Number of Simulations", min_value=1, value=10)

# Simulate button
simulate_button = st.button("Simulate", type="primary")

# Initialize session state for simulation results
if "simulations" not in st.session_state:
    st.session_state.simulations = None

# Run simulation when button is clicked
if simulate_button:
    st.session_state.simulations = simulate_coin_flips(
        prob_heads, num_flips, num_simulations
    )

# Display results if simulations exist
if st.session_state.simulations is not None:
    simulations = st.session_state.simulations

    # Prepare the figure
    fig = go.Figure()

    # Plot each simulation
    for i in range(simulations.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, num_flips + 1),
                y=simulations[i],
                mode="lines",
                name=f"Simulation {i+1}",
                line=dict(width=1),
            )
        )

    # Average cumulative proportion of heads
    average_cumulative_heads = np.mean(simulations, axis=0)

    # Perform linear regression on the average cumulative heads
    time_steps = np.arange(1, num_flips + 1).reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(time_steps, average_cumulative_heads)
    reg_line = reg.predict(time_steps)

    # Plot the average cumulative proportion of heads
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, num_flips + 1),
            y=average_cumulative_heads,
            mode="lines",
            name="Average Cumulative Heads",
            line=dict(width=4, color="red", dash="dash"),
        )
    )

    # Plot the regression line
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, num_flips + 1),
            y=reg_line,
            mode="lines",
            name="Regression Line",
            line=dict(width=4, color="green"),
        )
    )

    fig.update_layout(
        title="Cumulative Proportion of Heads Over Flips with Regression",
        xaxis_title="Number of Flips",
        yaxis_title="Cumulative Proportion of Heads",
        template="plotly_white",
        height=600,
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Calculate metrics
    final_proportions = simulations[:, -1]
    avg_heads = np.mean(final_proportions)
    avg_tails = 1 - avg_heads
    variance_heads = np.var(final_proportions)

    # Regression metrics
    slope = reg.coef_[0]
    intercept = reg.intercept_

    # Display metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Average Final Proportion of Heads", f"{avg_heads:.4f}")
        st.metric("Average Final Proportion of Tails", f"{avg_tails:.4f}")
        st.metric("Variance of Final Proportion of Heads", f"{variance_heads:.4f}")

    with col2:
        st.metric("Linear Regression Slope", f"{slope:.4f}")
        st.metric("Linear Regression Intercept", f"{intercept:.4f}")

else:
    # Show placeholder until simulation is run
    st.info("Click the Simulate button to run the coin flip simulation.")
