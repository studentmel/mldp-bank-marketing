"""
ENTRY POINT FOR APPLICATION CODE:

1. Code for the application called BankConvert AI: Term Deposit Subscription Prediction Model 

2. Rationale for Name: 
Name is derived from the concept of helping banks convert potential customers into subscribers for term deposits. 
"""

#---------------------------------------------------------------------------------------------------------

# SECTION 1: ALL IMPORTS 
# Firstly, need to import the libraries needed for the application to function 

# Need import joblib to load the saved model files (.pkl) that trained in Jupyter notebook
import joblib 

# Need import streamlit since it is web framework to create web interface
import streamlit as st

# import numpy so to provide math operations for numerical computations
import numpy as np

# need to import pandas is needed for data manipulation (for creating DataFrames for prediction input)
import pandas as pd 

# Need to also import plotly to creat the interactive gauge chart for visualising prediction probability
import plotly.graph_objects as go # mainly for the visuals 

#---------------------------------------------------------------------------------------------------------

# SECTION 2: PAGE CONFIG 
# so as to set browser tab title appearing in the browser and then title and icon and layout 

st.set_page_config( 
    page_title="BankConvert AI", # Text that will be shown in the browser tab
    page_icon="🤖", # icon in the browser tab
    layout="wide",
    initial_sidebar_state="expanded" # to open sidebar default 
)

#---------------------------------------------------------------------------------------------------------

# SECTION 3: THEME CONFIG 
# so can set theme mode in application 
# MAIN TOGGLE IS FOR DARK/LIGHT MODE 

# the session_state will then persist across user interactions e.g. button clicks

if "theme" not in st.session_state: # This is to check if theme has been set before in session state
    st.session_state.theme = "dark" # Default dark mode

def get_theme():
    """To returns current theme ('dark'/'light') from session state."""
    return st.session_state.theme

#---------------------------------------------------------------------------------------------------------

# SECTION 4: CSS STYLING for theme toggle

def apply_theme():
    theme = get_theme() # Firstly need to get the current theme (dark/light)
    
    # For dark mode 
    if theme == "dark":
        bg_primary = "#0B0F1E" # for main bg colour 
        bg_secondary = "#111631" # Sidebar bg colour
        bg_card = "rgba(255, 255, 255, 0.04)" # card bg 
        bg_card_hover = "rgba(255, 255, 255, 0.07)" # Card bg during hover
        text_primary = "#F0F0F5" # text
        text_secondary = "rgba(255, 255, 255, 0.55)" # test
        text_muted = "rgba(255, 255, 255, 0.35)" # text 
        border = "rgba(255, 255, 255, 0.08)" 
        border_hover = "rgba(94, 252, 232, 0.25)" 
        sb_card_bg = "rgba(0, 0, 0, 0.25)"
        input_bg = "rgba(255, 255, 255, 0.04)" 
        table_td_color = "rgba(255, 255, 255, 0.55)" 
        shadow_glow = "0 0 30px rgba(94, 252, 232, 0.08)" # Glow effect on hover (teal shadow) so that have MORE PROFFESSIONAL UI !!
        flow_bg = "rgba(94, 252, 232, 0.08)" 
        flow_border = "rgba(94, 252, 232, 0.15)" 
        compare_before_bg = "rgba(248, 113, 113, 0.1)" 
        compare_before_border = "rgba(248, 113, 113, 0.2)"
        compare_after_bg = "rgba(52, 211, 153, 0.1)"
        compare_after_border = "rgba(52, 211, 153, 0.2)" 
        profile_bg = "rgba(108, 99, 255, 0.06)"
        profile_border = "rgba(108, 99, 255, 0.2)" 
        action_bg = "rgba(94, 252, 232, 0.06)"
        action_border = "rgba(94, 252, 232, 0.2)"

    # For light mode used when switch light mode
    else:
        bg_primary = "#F8F9FC" 
        bg_secondary = "#FFFFFF" 
        bg_card = "rgba(0, 0, 0, 0.03)"
        bg_card_hover = "rgba(0, 0, 0, 0.06)"
        text_primary = "#1A1A2E"
        text_secondary = "rgba(0, 0, 0, 0.6)"
        text_muted = "rgba(0, 0, 0, 0.4)"
        border = "rgba(0, 0, 0, 0.1)"
        border_hover = "rgba(94, 200, 200, 0.4)"
        sb_card_bg = "rgba(0, 0, 0, 0.04)"
        input_bg = "rgba(0, 0, 0, 0.03)"
        table_td_color = "rgba(0, 0, 0, 0.6)"
        shadow_glow = "0 0 30px rgba(94, 200, 200, 0.1)"
        flow_bg = "rgba(94, 200, 200, 0.08)"
        flow_border = "rgba(94, 200, 200, 0.2)"
        compare_before_bg = "rgba(248, 113, 113, 0.08)"
        compare_before_border = "rgba(248, 113, 113, 0.3)"
        compare_after_bg = "rgba(52, 211, 153, 0.08)"
        compare_after_border = "rgba(52, 211, 153, 0.3)"
        profile_bg = "rgba(108, 99, 255, 0.05)"
        profile_border = "rgba(108, 99, 255, 0.2)"
        action_bg = "rgba(94, 200, 200, 0.05)"
        action_border = "rgba(94, 200, 200, 0.25)"

    # now need to inject css to page so that ui can appear nicely
    st.markdown(f"""
<style>
    /* Firstly using the DM Sans font from Google Fonts for professional*/
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap');

    /* ROOT VARIABLES
       Defined above alr get from the on top*/
    :root {{
        --bg-primary: {bg_primary};
        --bg-secondary: {bg_secondary};
        --bg-card: {bg_card};
        --bg-card-hover: {bg_card_hover};
        --accent-1: #5EFCE8; 
        --accent-2: #6C63FF; 
        --accent-gradient: linear-gradient(135deg, #5EFCE8 0%, #6C63FF 100%); 
        --text-primary: {text_primary};
        --text-secondary: {text_secondary};
        --text-muted: {text_muted};
        --border: {border};
        --border-hover: {border_hover};
        --success: #34D399; 
        --danger: #F87171; 
        --warning: #FBBF24; 
        --radius-sm: 8px; 
        --radius-md: 14px; 
        --radius-lg: 20px; 
        --shadow-glow: {shadow_glow};
    }}
    /*GLOBAL*/
    .stApp {{
        background: var(--bg-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
    }}    
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp li, .stApp td, .stApp th {{
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
    }}

    /*SIDEBAR*/
    section[data-testid="stSidebar"] {{
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }}
    /*For the Sidebar Heading*/
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: var(--accent-1) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }}
    /*Sidebar divider */
    section[data-testid="stSidebar"] hr {{
        border-color: var(--border) !important;
        margin: 1.2rem 0 !important;
    }}
    /*body text*/
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] li {{
        color: var(--text-secondary) !important;
        font-size: 0.88rem !important;
        line-height: 1.6 !important;
    }}
    /*HEADINGS*/
    h1, h2, h3 {{
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
    }}
    h4 {{
        color: var(--accent-1) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }}
    /*HERO HEADER*/
    .hero {{
        background: linear-gradient(135deg, rgba(94,252,232,0.08) 0%, rgba(108,99,255,0.08) 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 2.8rem 2rem;
        text-align: center;
        margin-bottom: 1.8rem;
    }}
    /*Hero title*/
    .hero h1 {{
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.4rem;
        letter-spacing: -0.04em !important;
    }}
    /*Hero subtitle*/
    .hero .subtitle {{
        color: var(--text-secondary) !important;
        font-size: 1.05rem;
        font-weight: 400;
    }}
    /* Hero badge*/
    .hero .badge {{
        display: inline-block;
        margin-top: 1rem;
        padding: 6px 18px;
        background: rgba(94, 252, 232, 0.1);
        border: 1px solid rgba(94, 252, 232, 0.2);
        border-radius: 100px;
        font-size: 0.78rem;
        color: var(--accent-1) !important;
        font-weight: 600;
        letter-spacing: 0.04em;
    }}
    /*CARDS with hover effect glow*/
    .card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        transition: all 0.25s ease;
    }}
    .card:hover {{
        background: var(--bg-card-hover);
        border-color: var(--border-hover);
        box-shadow: var(--shadow-glow);
    }}
    /*Card header*/
    .card h4 {{
        margin-bottom: 0.8rem !important;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }}
    .card p {{
        color: var(--text-secondary) !important;
        font-size: 0.9rem;
        line-height: 1.65;
    }}
    .card strong {{
        color: var(--text-primary) !important;
    }}
    /*METRIC tab*/
    .metric-row {{
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1.4rem;
        flex-wrap: wrap;
    }}
    .metric-pill {{
        flex: 1;
        min-width: 130px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.2rem 1rem;
        text-align: center;
        transition: all 0.25s ease;
    }}
    .metric-pill:hover {{
        border-color: var(--border-hover);
        box-shadow: var(--shadow-glow);
    }}
    .metric-pill .value {{
        font-size: 1.6rem;
        font-weight: 700;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }}
    .metric-pill .label {{
        font-size: 0.7rem;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
        font-weight: 500;
    }}
    /*RESULT CARDS*/
    .result-yes {{
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.12) 0%, rgba(52, 211, 153, 0.04) 100%);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: var(--radius-md);
        padding: 2rem;
        text-align: center;
    }}
    .result-yes h2 {{
        color: var(--success) !important;
        font-size: 1.5rem !important;
        margin-bottom: 0.4rem;
    }}
    .result-yes .conf {{
        color: var(--text-primary) !important;
        font-size: 1.1rem;
        font-weight: 600;
    }}
    .result-yes p {{
        color: var(--text-secondary) !important;
        font-size: 0.88rem;
        margin-top: 0.6rem;
    }}
    .result-no {{
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.12) 0%, rgba(248, 113, 113, 0.04) 100%);
        border: 1px solid rgba(248, 113, 113, 0.3);
        border-radius: var(--radius-md);
        padding: 2rem;
        text-align: center;
    }}
    .result-no h2 {{
        color: var(--danger) !important;
        font-size: 1.5rem !important;
        margin-bottom: 0.4rem;
    }}
    .result-no .conf {{
        color: var(--text-primary) !important;
        font-size: 1.1rem;
        font-weight: 600;
    }}
    .result-no p {{
        color: var(--text-secondary) !important;
        font-size: 0.88rem;
        margin-top: 0.6rem;
    }}
    /*SIDEBAR STAT CARD*/
    .sb-card {{
        background: {sb_card_bg};
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .sb-card .sb-label {{
        font-size: 0.75rem;
        color: var(--text-muted) !important;
        font-weight: 500;
    }}
    .sb-card .sb-value {{
        font-size: 0.9rem;
        color: var(--accent-1) !important;
        font-weight: 700;
    }}
    /*TABS*/    
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 5px;
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: var(--text-muted) !important;
        border-radius: var(--radius-sm);
        font-weight: 500;
        font-size: 0.88rem;
        padding: 8px 20px;
    }}    
    .stTabs [aria-selected="true"] {{
        background: rgba(94, 252, 232, 0.12) !important;
        color: var(--accent-1) !important;
        font-weight: 600;
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        background: transparent !important;
        padding-top: 1.5rem !important;
    }}
    /*BUTTONS*/
    .stButton > button {{
        background: var(--accent-gradient) !important;
        color: #0B0F1E !important;
        border: none !important;
        padding: 0.8rem 2.4rem !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        border-radius: 100px !important;
        letter-spacing: 0.02em !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(94, 252, 232, 0.2) !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(94, 252, 232, 0.3) !important;
    }}    
    /*INPUTS*/
    .stSlider > div > div > div > div {{
        background: var(--accent-gradient) !important;
    }}    
    .stSlider label, .stSelectbox label {{
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }}
    div[data-baseweb="select"] > div {{
        background: {input_bg} !important;
        border-color: var(--border) !important;
        border-radius: var(--radius-sm) !important;
    }}
    /*DIVIDERS for horizontal line*/
    hr {{
        border-color: var(--border) !important;
    }}

    /*TABLE*/
    .stMarkdown table {{
        border-collapse: separate;
        border-spacing: 0;
        border-radius: var(--radius-sm);
        overflow: hidden;
    }}
    .stMarkdown th {{
        background: rgba(94, 252, 232, 0.08) !important;
        color: var(--accent-1) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        font-weight: 600 !important;
        border-color: var(--border) !important;
        padding: 0.7rem 1rem !important;
    }}
    .stMarkdown td {{
        color: {table_td_color} !important;
        border-color: var(--border) !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.88rem !important;
    }}

    /*ALERTS*/
    .stSuccess, .stInfo, .stWarning, .stError {{
        background: var(--bg-card) !important;
        border-radius: var(--radius-sm) !important;
    }}

    /*SECTION HEADER*/
    .section-header {{
        margin-bottom: 1.2rem;
    }}
    .section-header h3 {{
        font-size: 1.3rem !important;
        margin-bottom: 0.3rem !important;
    }}
    .section-header p {{
        color: var(--text-muted) !important;
        font-size: 0.88rem;
        margin-top: 0;
    }}
    /*SECTION DIVIDER LABEL*/
    .section-label {{
        font-size: 0.7rem;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }}

    /*JOURNEY STEPS*/
    .journey {{
        display: flex;
        gap: 0.8rem;
        margin-bottom: 0.8rem;
        align-items: flex-start;
    }}
    .journey .num {{
        min-width: 32px;
        height: 32px;
        background: var(--accent-gradient);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        color: #0B0F1E !important;
        flex-shrink: 0;
    }}
    .journey .content {{
        flex: 1;
    }}
    .journey .content .title {{
        font-weight: 600;
        color: var(--text-primary) !important;
        font-size: 0.9rem;
    }}
    .journey .content .desc {{
        color: var(--text-secondary) !important;
        font-size: 0.82rem;
        line-height: 1.5;
    }}
    /*PIPELINE FLOW*/
    .pipeline {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        flex-wrap: wrap;
        padding: 1.5rem 1rem;
        background: {flow_bg};
        border: 1px solid {flow_border};
        border-radius: var(--radius-md);
        margin-bottom: 1rem;
    }}
    .pipeline .step {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.6rem 1rem;
        text-align: center;
        min-width: 100px;
        transition: all 0.2s ease;
    }}
    .pipeline .step:hover {{
        border-color: var(--border-hover);
        box-shadow: var(--shadow-glow);
    }}
    .pipeline .step .step-icon {{
        font-size: 1.3rem;
        margin-bottom: 0.2rem;
    }}
    .pipeline .step .step-label {{
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-primary) !important;
        letter-spacing: 0.02em;
    }}
    .pipeline .step .step-desc {{
        font-size: 0.6rem;
        color: var(--text-muted) !important;
    }}
    .pipeline .arrow {{
        color: var(--text-muted) !important;
        font-size: 1.2rem;
        font-weight: 300;
    }}


    /*BEFORE/AFTER COMPARISON*/
    .compare-container {{
        display: flex;
        gap: 1.5rem;
        align-items: stretch;
        margin-bottom: 1rem;
    }}
    .compare-side {{
        flex: 1;
        padding: 1.5rem;
        border-radius: var(--radius-md);
        text-align: center;
    }}
    .compare-before {{
        background: {compare_before_bg};
        border: 1px solid {compare_before_border};
    }}
    .compare-after {{
        background: {compare_after_bg};
        border: 1px solid {compare_after_border};
    }}
    .compare-side .compare-icon {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }}
    .compare-side .compare-title {{
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }}
    .compare-before .compare-title {{
        color: var(--danger) !important;
    }}
    .compare-after .compare-title {{
        color: var(--success) !important;
    }}
    .compare-side .compare-stat {{
        font-size: 0.82rem;
        color: var(--text-secondary) !important;
        line-height: 1.7;
    }}
    .compare-side .compare-stat strong {{
        color: var(--text-primary) !important;
    }}
    .compare-vs {{
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-muted) !important;
        min-width: 40px;
    }}

    /*CUSTOMER PROFILE SUMMARY*/
    .profile-summary {{
        background: {profile_bg};
        border: 1px solid {profile_border};
        border-radius: var(--radius-md);
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .profile-summary .profile-title {{
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--accent-2) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {profile_border};
    }}
    .profile-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.5rem 1.5rem;
    }}
    .profile-item {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.3rem 0;
    }}
    .profile-item .p-label {{
        font-size: 0.78rem;
        color: var(--text-muted) !important;
        font-weight: 500;
    }}
    .profile-item .p-value {{
        font-size: 0.82rem;
        color: var(--text-primary) !important;
        font-weight: 600;
    }}

    /*RECOMMENDED ACTION*/
    .action-card {{
        background: {action_bg};
        border: 1px solid {action_border};
        border-radius: var(--radius-md);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }}
    /*Title*/
    .action-card .action-title {{
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--accent-1) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {action_border};
    }}
    .action-card .action-item {{
        display: flex;
        gap: 0.6rem;
        align-items: flex-start;
        margin-bottom: 0.5rem;
    }}
    .action-card .action-bullet {{
        color: var(--accent-1) !important;
        font-weight: 700;
        font-size: 0.9rem;
        flex-shrink: 0;
        margin-top: 1px;
    }}
    .action-card .action-text {{
        font-size: 0.85rem;
        color: var(--text-secondary) !important;
        line-height: 1.5;
    }}
    .action-card .action-text strong {{
        color: var(--text-primary) !important;
    }}
</style>
""", unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------

# SECTION 5: LOADING MODEL FUNCTIONS

# so that can load the trained model

@st.cache_resource # Cache function so that only runs once, IMPORTATN
def load_models():
    """Firslty, have to load saved models using joblib"""
    try:
        model = joblib.load("best_model.pkl") # trained Random Forest model
        scaler = joblib.load("scaler.pkl") # StandardScaler
        feature_columns = joblib.load("feature_columns.pkl") # feature column names
        return model, scaler, feature_columns
    except FileNotFoundError:
        # If file is missing, then for debug
        st.error("Model files not found. Please run Jupyter notebook first to generate .pkl files")
        st.info("Required files: best_model.pkl, scaler.pkl, feature_columns.pkl")
        return None, None, None


@st.cache_resource # Cache this function too
def load_thresholds():
    """
    This is to load median thresholds for economic_condition feature engineering"""
    try:
        #loading from saved thresholds.pkl file
        thresholds = joblib.load("thresholds.pkl")
        return thresholds['emp_median'], thresholds['nr_median']
    except FileNotFoundError:
        try:
            # Config a fallback such that if thresholds.pkl not found, recalculate from raw dataset
            df = pd.read_csv('bank-additional-full.csv', sep=';') #original dataset
            df = df.drop_duplicates() #Removing duplicate rows
            df = df.drop(['duration', 'campaign'], axis=1) #Removing data leakage columns
            df.loc[df['y'] == 'yes', 'y'] = 1 #Encoding target: yes to 1
            df.loc[df['y'] == 'no', 'y'] = 0 #Encoding target: no to 0
            df['y'] = df['y'].astype(int) #target to integer
            X = df.drop('y', axis=1) #Separate features
            y = df['y'] #target
            from sklearn.model_selection import train_test_split
            # redo train-test split
            X_train, _, _, _ = train_test_split(X, y, test_size=0.3, random_state=2025, stratify=y)
            return X_train['emp.var.rate'].median(), X_train['nr.employed'].median()
        except FileNotFoundError:
            # last last falllback is hardcoded default values
            return 1.1, 5191.0

