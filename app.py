import streamlit as st
from pathlib import Path
from lib.config_editor_page import config_editor_page
from lib.dataset_editor_page import dataset_builder_page
from lib.automatic_evaluation_page import automatic_evaluation_page
from lib.manual_evaluation_page import manual_evaluation_page


# Must be the first Streamlit command
st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="🤖",
    layout="wide",
    # Hide default sidebar menu
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Hide streamlit default menu and footer
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* Hide default sidebar nav */
        .css-1d391kg {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# Load and inject custom CSS
def load_css():
    css_file = Path("static/style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.error("CSS file not found!")

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Configuration Editor"

def change_page(page_name):
    st.session_state.current_page = page_name

def main():
    """Main app with fancy 3D navigation."""
    # Load CSS
    load_css()
    
    # Sidebar title
    st.sidebar.markdown('<div class="page-title">Navigation</div>', unsafe_allow_html=True)
    
    # Navigation container
    st.sidebar.markdown('<div class="nav-container">', unsafe_allow_html=True)
    
    # Navigation buttons
    pages = {
        "Configuration Editor": "⚙️",
        "Datasets Page": "📝",
        "Manual Evaluation": "✍️",
        "Automatic Evaluation": "🚀"
    }
    
    # Create navigation buttons
    for page, icon in pages.items():
        button_style = "selected" if st.session_state.current_page == page else ""
        if st.sidebar.button(
            f"{icon} {page}",
            key=f"nav_{page.lower().replace(' ', '_')}",
            help=f"Go to {page}",
            on_click=change_page,
            args=(page,),
            use_container_width=True
        ):
            pass  # The on_click handler will handle the page change
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Display current page based on session state
    if st.session_state.current_page == "Configuration Editor":
        config_editor_page()
    elif st.session_state.current_page == "Datasets Page":
        dataset_builder_page()
    elif st.session_state.current_page == "Automatic Evaluation":
        automatic_evaluation_page()
    else:
        manual_evaluation_page()

    # Footer
    st.sidebar.markdown(
        '<div class="footer">LLM Evaluation Tool • Version 1.0</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()