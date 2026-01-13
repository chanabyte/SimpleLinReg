import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def gen_house_data(n_samples=100):
    np.random.seed(50)
    size = np.maximum(np.random.normal(1800, 500, n_samples), 0)
    price = size * 100 + np.random.normal(0, 10000, n_samples)
    return pd.DataFrame({'size': size, 'price': price})

def train_model():
    df = gen_house_data()
    X = df[["size"]]
    Y = df[["price"]]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def main():
    st.title("House Price Prediction App")
    st.write("This app predicts house prices based on their size using a simple linear regression model.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Adjust House Size")
        size = st.slider(
            "House size (sq ft)", 
            min_value=500, 
            max_value=3000, 
            value=1500, 
            step=10, 
        # orientation="vertical"  # Makes the slider vertical
        )


    model = train_model()
    pred_price = model.predict([[size]])

    df = gen_house_data()
    fig = px.scatter(df, x="size", y="price", title="House Size vs Price")
    fig.add_scatter(
        x=[size],
        y=[pred_price[0].item()],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='Prediction'
    )
    fig.update_layout(
        xaxis_title="House Size (sq ft)",
        yaxis_title="Price ($)",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white"
    )

    
    with col2:
        st.subheader("Predicted Price")
        st.success(f"Estimated Price: ${pred_price[0].item():,.2f}")
    
    st.subheader("Scatter Plot")
    st.plotly_chart(fig)



    st.markdown("---")
    st.markdown("**Created by [Swarna Sre :)](https://github.com/chanabyte)**")
    st.markdown("Source code available on [GitHub](https://github.com/chanabyte/SimpleLinReg/blob/main/HousePredModel.py).")

if __name__ == "__main__":
    main()
