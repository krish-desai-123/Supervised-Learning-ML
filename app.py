import streamlit as st

st.title("My First Streamlit App")

name = st.text_input("Enter Your Name")

if name:
    st.success(f"Hello {name}!")