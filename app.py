import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="Newsy - Headlines Interrogation",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 Newsy")
    st.subheader("Headlines Interrogation and Classification")
    
    st.write("Welcome to Newsy! This is where we'll build our headlines analysis tool.")
    
    # Display environment status
    with st.expander("Environment Status"):
        st.code(f"""
        Environment: {os.getenv('ENVIRONMENT', 'development')}
        Log Level: {os.getenv('LOG_LEVEL', 'INFO')}
        """)

if __name__ == "__main__":
    main()
