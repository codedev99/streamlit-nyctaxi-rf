import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

DATA_PATH = "./NYCData/2018_Yellow_Taxi_Trip_Data.csv"

st.markdown("""
<style>
.main {
    background-color: #F5F5F5;
}
</style>
""",
unsafe_allow_html=True)

@st.cache(persist=True)
def load_data(nrows):
    data = pd.read_csv(DATA_PATH, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    # data["date/time"] = pd.to_datetime(data["date/time"])

    return data

with header:
    st.title("My Awesome Data Science Project!!")
    st.text("This project trains Random Forests on the New York City Taxi Dataset.")

with dataset:
    st.header("NYC Taxi Dataset")
    s = "This dataset belongs to New York City Yellow Taxi database. Consists of datapoints describing time and location of taxi pickups."
    st.text(s)

    data = load_data(100000)
    st.write(data.head())

    st.subheader("Pick-up location id distribution on NYC Taxi Dataset")
    pulocation_dist = pd.DataFrame(data["pulocationid"].value_counts())
    st.bar_chart(pulocation_dist.head(50))

with features:
    st.header("Features which are available to use:")

    st.write(data.columns)

with model_training:
    st.header("Time to train the model!")
    st.text("Choose the hyperparameters to train the ML model.")

    sel_col, disp_col = st.columns(2)

    sel_col.subheader("Selection column")
    disp_col.subheader("Display column")

    max_depth = sel_col.slider("Select max depth of the model", min_value=10, max_value=100, value=20, step=10)
    n_estimators = sel_col.selectbox("Select number of trees in your model", options=[100,200,300,"No Limit"], index=0)
    input_feature = sel_col.text_input("Enter which feature should be used by the model", "pulocationid")

    X = data[[input_feature]].values
    Y = data[["trip_distance"]].values

    Y = Y.reshape(-1, 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    
    if n_estimators == "No Limit":
        estimator = RandomForestRegressor(max_depth=max_depth)
    else:
        estimator = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    
    estimator.fit(X_train, Y_train)
    Y_pred = estimator.predict(X_test)

    mse = mean_squared_error(Y_test, Y_pred)
    disp_col.markdown("###Mean squared error of the model is:")
    disp_col.write(mse)

    mae = mean_absolute_error(Y_test, Y_pred)
    disp_col.markdown("###Mean absolute error of the model is:")
    disp_col.write(mae)

    r2 = r2_score(Y_test, Y_pred)
    disp_col.markdown("###R squared score of the model is:")
    disp_col.write(r2)