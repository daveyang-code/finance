import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import beta

st.set_page_config(page_title="Coin Toss Beta Distribution", layout="wide")


# Helper function to simulate coin flips
def simulate_coin_flips(prob_heads, num_flips):
    return np.random.choice([0, 1], size=num_flips, p=[1 - prob_heads, prob_heads])


st.title("Coin Toss Beta Distribution")

col1, col2 = st.columns(2)

with col1:
    prob_heads = st.number_input(
        "Probability of Heads (0 to 1)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

with col2:
    num_flips = st.number_input("Number of Flips", min_value=1, value=10)

simulate_button = st.button("Simulate Flips", type="primary")

# Initialize session state for simulation results
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = []

# Run simulation when button is clicked
if simulate_button:
    # Clear previous results
    st.session_state.simulation_results = []

    # Simulate the coin flips
    all_flips = simulate_coin_flips(prob_heads, num_flips)

    # Process each flip
    for i in range(num_flips):
        # Get the flips up to the current flip
        flips_so_far = all_flips[: i + 1]

        num_heads = flips_so_far.sum()
        num_tails = len(flips_so_far) - num_heads

        # Beta distribution parameters
        alpha = 1 + num_heads  # Prior alpha is 1, updated with observed heads
        beta_param = 1 + num_tails  # Prior beta is 1, updated with observed tails

        # Calculate metrics
        posterior_mean = alpha / (alpha + beta_param)
        posterior_variance = (alpha * beta_param) / (
            (alpha + beta_param) ** 2 * (alpha + beta_param + 1)
        )

        # Store results
        st.session_state.simulation_results.append(
            {
                "result": "Heads" if all_flips[i] > 0 else "Tails",
                "num_heads": num_heads,
                "num_tails": num_tails,
                "alpha": alpha,
                "beta": beta_param,
                "posterior_mean": posterior_mean,
                "posterior_variance": posterior_variance,
            }
        )

# Display results if simulations exist
if st.session_state.simulation_results:
    # Slider to visualize the progression of the Beta distribution
    max_flip = len(st.session_state.simulation_results)
    selected_flip = st.slider(
        "Select Flip to Visualize Beta Distribution",
        min_value=1,
        max_value=max_flip,
        value=max_flip,
    )

    fig = go.Figure()

    # Generate x values for the Beta distribution
    x = np.linspace(0, 1, 500)

    # Get the selected result
    selected_result = st.session_state.simulation_results[selected_flip - 1]
    alpha = selected_result["alpha"]
    beta_param = selected_result["beta"]
    y = beta.pdf(x, alpha, beta_param)

    # Add the selected Beta distribution to the plot
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"Flip {selected_flip} (Heads: {selected_result['num_heads']}, Tails: {selected_result['num_tails']})",
            line=dict(width=2, color="green"),
        )
    )

    fig.update_layout(
        title=f"Beta Distribution After {selected_flip} Flips",
        xaxis_title="Probability of Heads",
        yaxis_title="Density",
        template="plotly_white",
        height=600,
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Create and display the metrics table
    st.subheader("Statistics for Each Flip")

    # Convert results to DataFrame for table display
    df = pd.DataFrame(st.session_state.simulation_results)
    df = df.rename(
        columns={
            "result": "Result",
            "num_heads": "Heads",
            "num_tails": "Tails",
            "alpha": "Alpha",
            "beta": "Beta",
            "posterior_mean": "Posterior Mean",
            "posterior_variance": "Posterior Variance",
        }
    )

    # Format the floating point columns
    df["Posterior Mean"] = df["Posterior Mean"].map("{:.4f}".format)
    df["Posterior Variance"] = df["Posterior Variance"].map("{:.4f}".format)

    df.index = df.index + 1

    # Display the table
    st.dataframe(df)

else:
    # Show placeholder until simulation is run
    st.info(
        "Click the Simulate Flips button to run the simulation and view the Beta distribution progression."
    )
