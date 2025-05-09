import streamlit as st

st.set_page_config(
    page_title="Real-Time Option Pricing",
    page_icon="📈",
    menu_items={"Get Help": None, "Report a Bug": None, "About": None}
    )

st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none}
        section[data-testid="stSidebar"] > div:first-child {display: none}
        .main > div:first-child {display: none}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none}
        section[data-testid="stSidebar"] > div:first-child {display: none}
        .main > div:first-child {display: none}
        button[kind="headerNoPadding"] {display: none}
        .st-emotion-cache-1dp5vir {display: none}
        [data-testid="stSidebarNav"] {display: none !important}
        .st-emotion-cache-16pwjcz {display: none}
    </style>
""", unsafe_allow_html=True)

# Create radio buttons for model selection
selected_model = st.radio(
    "Select an Options Pricing Model",
    ["Black-Scholes Model"]
)

# Add a button to navigate to the selected model
if st.button("Show Model", use_container_width=True):
    if selected_model == "Black-Scholes Model":
        st.switch_page("pages/1_black-scholes-model.py")