"""Small Streamlit viewer for the supplied restaurant demand history."""
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Restaurant Demand Forecasting", layout="wide")
st.title("Restaurant Demand Forecasting")
st.caption("Historical daily customer counts; no model performance claims are shown here.")

path = Path("data/raw/clientes_restaurantes.csv")
if not path.exists():
    st.error("Demand data was not found at data/raw/clientes_restaurantes.csv.")
    st.stop()

data = pd.read_csv(path, parse_dates=["data"], date_format="%m/%d/%Y")
data = data.set_index("data").sort_index()
restaurants = ["Chimi & Churri", "Assa Frão"]
st.line_chart(data[restaurants])
st.subheader("Data coverage and missing observations")
st.write({"first_date": str(data.index.min().date()), "last_date": str(data.index.max().date()),
          "rows": len(data), "missing_by_restaurant": data[restaurants].isna().sum().to_dict()})
st.dataframe(data.tail(14), use_container_width=True)
