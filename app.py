import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import LinearRegression

st.set_page_config(page_title="Linear Regression App", layout="centered")

st.title("📊 Linear Regression Model")
st.caption("Upload a CSV or manually enter data to fit a linear model")

model = LinearRegression()

# --- INPUT MODE ---
input_mode = st.radio("Choose input method:", ["📁 Upload CSV", "📝 Manual Entry"], horizontal=True)

# --- DATA HANDLING ---
x = y = None

if input_mode == "📁 Upload CSV":
    uploaded_file = st.file_uploader("Drag and drop or browse CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("Preview:", df.head())

        if len(df.columns) >= 2:
            x_col = st.selectbox("Select X column", df.columns)
            y_col = st.selectbox("Select Y column", df.columns)
            x = df[x_col].values
            y = df[y_col].values
        else:
            st.warning("Your CSV must have at least 2 columns.")

elif input_mode == "📝 Manual Entry":
    num_points = st.number_input("Number of data points", min_value=2, max_value=20, step=1, value=5)
    st.markdown("### Enter values:")
    x = []
    y = []
    cols = st.columns(2)
    for i in range(num_points):
        with cols[0]:
            x_val = st.number_input(f"x[{i+1}]", key=f"x_{i}")
        with cols[1]:
            y_val = st.number_input(f"y[{i+1}]", key=f"y_{i}")
        x.append(x_val)
        y.append(y_val)

# --- MODEL & VISUALIZATION ---
if x and y and len(x) > 1:
    model.fit(x, y)
    y_pred = model.predict(x)

    st.subheader("📌 Model Parameters")
    st.write(f"**Weight (slope)**: `{model.weight:.4f}`")
    st.write(f"**Bias (intercept)**: `{model.bias:.4f}`")

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Actual Data", color="#1f77b4")
    ax.plot(x, y_pred, color="red", label="Regression Line")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Linear Regression Fit")
    ax.legend()
    st.pyplot(fig)

    # Prediction
    st.subheader("🔮 Predict")
    new_x = st.number_input("Enter new X value to predict Y")
    if new_x is not None:
        predicted_y = model.predict([new_x])[0]
        st.success(f"Predicted Y value: `{predicted_y:.2f}`")
