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
            return 1.1, 5191.0 # never reaches here since have threshold 
#---------------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------------
# SECTION 6: GAUGE CHART

# for plotly gauge chart for prediction probability

def create_gauge_chart(probability):
    """Need this for the visuals so that have plotly gauge chart for probability visual"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", # so that can show gauge and number

        value=probability * 100, # this is so that converting the probability to %
        domain={'x': [0, 1], 'y': [0, 1]},
        number={
            'suffix': '%', # after n.o. %
            'font': {'size': 48, 'color': '#5EFCE8', 'family': 'DM Sans'}
        },
        gauge={
            'axis': {
                'range': [0, 100], #range 0 % to 100%
                'tickwidth': 1,
                'tickcolor': 'rgba(255,255,255,0.15)', #tick marks
                'tickfont': {'color': 'rgba(255,255,255,0.4)', 'size': 11}
            },
            'bar': {'color': '#6C63FF', 'thickness': 0.3}, 
            'bgcolor': 'rgba(255,255,255,0.03)',
            'borderwidth': 0,
            'steps': [
                #three colour for the gauge so that can also give the user a better idea 
                # and also at glance understanding 
                {'range': [0, 30], 'color': 'rgba(248, 113, 113, 0.15)'}, # For red zone (0-30%)
                {'range': [30, 50], 'color': 'rgba(251, 191, 36, 0.15)'}, # Yellow zone (30-50%)
                {'range': [50, 100], 'color': 'rgba(52, 211, 153, 0.15)'} # Green zone (50-100%)
            ],
            'threshold': {
                'line': {'color': '#5EFCE8', 'width': 3},
                'thickness': 0.8,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=25, b=15),
        font=dict(family='DM Sans', size=13, color='rgba(255,255,255,0.6)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig
#---------------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------------
# SECTION 6: FEATURE ENGINEERING 
# need do the preprocessing again same as jupyter 


def create_feature_engineering(input_data, emp_median, nr_median):
    df = input_data.copy() # This need to make copy so that need to avoid modifying og data

#---------------------------------------------------------------------------------------------------------
    # Feature 1: Age Group
    # same as jupyter 

    age = df['age'].iloc[0] 
    if age <= 30:
        age_group = 'Young'
    elif age <= 45:
        age_group = 'Middle'
    elif age <= 60:
        age_group = 'Senior'
    else:
        age_group = 'Elderly'
    df['age_group'] = age_group 
#---------------------------------------------------------------------------------------------------------
    # Feature 2: Contacted Before

    df['contacted_before'] = (df['pdays'] != 999).astype(int)
#---------------------------------------------------------------------------------------------------------
    # Feature 3: Previous Success
    df['prev_success'] = (df['poutcome'] == 'success').astype(int)
#---------------------------------------------------------------------------------------------------------
    # Feature 4: Economic Condition
    emp_var_rate = df['emp.var.rate'].iloc[0]
    nr_employed = df['nr.employed'].iloc[0]
    if emp_var_rate > emp_median and nr_employed > nr_median:
        economic_condition = 'Good'
    elif emp_var_rate <= emp_median and nr_employed <= nr_median:
        economic_condition = 'Bad'
    else:
        economic_condition = 'Neutral'
    df['economic_condition'] = economic_condition
#---------------------------------------------------------------------------------------------------------
    # Feature 5: Contact Recency
    pdays = df['pdays'].iloc[0]
    if pdays == 999:
        contact_recency = 'Never'
    elif pdays <= 7:
        contact_recency = 'Recent'
    elif pdays <= 30:
        contact_recency = 'Medium'
    else:
        contact_recency = 'Long'
    df['contact_recency'] = contact_recency

    return df #5 new feature columns
#---------------------------------------------------------------------------------------------------------

def preprocess_input(input_data, feature_columns, scaler, emp_median, nr_median):
    
    # Step 1: Firstly, need to apply 5 engineered features
    df = create_feature_engineering(input_data, emp_median, nr_median)

    # Step 2: One-Hot Encoding
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing',
                       'loan', 'contact', 'month', 'day_of_week', 'poutcome',
                       'age_group', 'economic_condition', 'contact_recency']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Step 3: same col
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[feature_columns]

    # Step 5: Scaling numerical since diff scale
    numerical_cols = list(scaler.feature_names_in_)
    df_scaled = df_encoded.copy()
    df_scaled[numerical_cols] = scaler.transform(df_encoded[numerical_cols])

    return df_scaled.values
#---------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------- 
# SECTION 7: RECOMMENDED ACTIONS GENERATOR
# so that can show the rm some talking points based on the customer profile & prediction

def generate_recommendations(prediction, probability, age, job, poutcome, emp_var_rate,
                              pdays, previous, default, housing, loan, contact, education):
    """ This is so that have specific RM talking point and like the rm can dont blank out if nothing to say can help them 
    to realte too """
    actions = [] # empty array to collect

    if prediction == 1:
        # meaning that there is HIGH POTENTIAL of the customer
        
        if poutcome == 'success':
            # This feature meaning that customer had previously subscribed can be used as ref
            actions.append("Previous campaign was <strong>successful</strong> - You may want to reference their past term deposit experience and highlight improved rates from then")

        if age > 60:
            # This mean that elderly customers prioritise safety risks
            actions.append("Customer is <strong>retired/elderly</strong> - You may want to emphasise guaranteed returns, and deposit insurance protection from SDIC")
        elif age <= 30:
            # This mean that young customers may want to be discipline and save and can convey message 
            actions.append("Customer is <strong>young</strong> - You may want to position and phrase term deposit subscription as a disciplined savings tool to build financial foundation as well as grow the money")

        if emp_var_rate < 0:
            # this mean weak economy and people will seek safe investments like term deposits
            actions.append("Economy is <strong>weakening</strong> - You may want to highlight term deposits as a safe investment to customers especially during market uncertainty")

        if default == 'no' and loan == 'no' and housing == 'no':
            # No debt means likely has money available to invest
            actions.append("Customer has <strong>no existing debt</strong> - likely has disposable income available for investment")

        if contact == 'cellular':
            # Mobile contact so that can follow up
            actions.append("Contact via <strong>cellular</strong> - customer is reachable on mobile, you may want to consider sending follow-up SMS to check in with them")

        if not actions:
            # Fallback last 
            actions.append("Customer profile shows <strong>strong subscription signals</strong> - You may want to prioritise them for immediate follow-ups")
    
    else:
        # meaning there is LOW POTENTIAL
        #         
        if poutcome == 'failure':
            # customer indicate previous campaign failed and be more careful in marketing 
            actions.append("Previous campaign <strong>failed</strong> - You may want to avoid hard-sell approach and focus on explaining changed circumstances to them")

        if pdays == 999 and previous == 0:
            # meaning customer never contacted before and should build rapport
            actions.append("Customer was <strong>never contacted before</strong> - You may want to introduce yourself first to build rapport before pitching")

        if default == 'yes':
            # meaning credit default have
            actions.append("Customer has <strong>credit default</strong> - They may face financial difficulties so you may want to approach sensibly")

        if emp_var_rate > 0:
            # meaning strong economy so customer may prefer risk investments
            actions.append("Economy is <strong>strong</strong> - customer may prefer higher-risk investments, you may want to mention flexibility of shorter term deposits")

        if education in ['basic.4y', 'basic.6y', 'illiterate']:
            # meaning customer has lower education level, can use simpler language to explain products
            actions.append("Consider using <strong>simpler language</strong> to explain term deposit benefits and avoid financial jargon")

        if not actions:
            # Fallback 
            actions.append("Customer shows <strong>low subscription likelihood</strong> - You may want to deprioritise and allocate time to higher-potential prospects")

    return actions # Return reco
#---------------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------------
# SECTION 8: MAIN APP

def main():
    # Firslty, need to load all model files
    model, scaler, feature_columns = load_models()
    emp_median, nr_median = load_thresholds() # and economic condition thresholds

    # also need to apply CSS theme so can use dark and light 
    apply_theme()

    # SIDEBAR
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.2rem;">
            <div style="font-size:2rem; margin-bottom:0.3rem;">🤖</div>
            <div style="font-size:1.4rem; font-weight:700; background: linear-gradient(135deg, #5EFCE8, #6C63FF);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">BankConvert AI</div>
            <div style="font-size:0.75rem; color:rgba(255,255,255,0.4) !important; margin-top:0.2rem;
            letter-spacing:0.08em; text-transform:uppercase;">ML Prediction Engine</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---") # divider line for neat

        # Theme toggle but show current state first
        theme_label = "🌙 Dark Mode" if get_theme() == "dark" else "☀️ Light Mode"
        if st.button(theme_label, key="theme_toggle"):            
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun() # Rerun to apply new theme

        st.markdown("---")

        # MODEL STATS so that can know 
        st.markdown("""
        <div class="sb-card"><span class="sb-label">Model</span><span class="sb-value">Random Forest</span></div>
        <div class="sb-card"><span class="sb-label">F1-Score</span><span class="sb-value">48.58%</span></div>
        <div class="sb-card"><span class="sb-label">Recall</span><span class="sb-value">52.37%</span></div>
        <div class="sb-card"><span class="sb-label">Test Results</span><span class="sb-value">729 / 1,392</span></div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # for me 
        st.markdown("### Developer")
        st.markdown("**Melissa Kuah** (2404487G)  \nML for Developers (P01) \nTemasek Polytechnic")

        st.markdown("---")

        # Model improvement summary so that can pitch better 
        st.markdown("### Improvement")
        st.markdown("""
        Recall: **31.68% => 52.37%**  
        Finding **+288 more** subscribers  
        F1: **39.52% => 48.58%**
        """)

    # HERO HEADER
    st.markdown("""
    <div class="hero">
        <h1>BankConvert AI</h1>
        <div class="subtitle">Predict term deposit subscription likelihood before the call</div>
        <div class="badge">🤖 Random Forest · Tuned · class_weight='balanced'</div>
    </div>
    """, unsafe_allow_html=True)

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(["🔮  Predict", "📊  Performance", "🧠  How It Works", "📋  About"])

    # TAB 1: PREDICT
    # To allow user to input customer data and get a prediction
    with tab1:
        # model must be loaded successfully first
        if model is None:
            st.warning("⚠️ Model not loaded. Ensure best_model.pkl, scaler.pkl, and feature_columns.pkl are in the same directory as this app.")
            return # cant continue

        st.markdown("""
        <div class="section-header">
            <h3>Enter Customer Information</h3>
            <p>Fill in the customer details below to predict their likelihood of subscribing to a term deposit.
            Adjust the sliders and dropdowns, then click <strong>Run Prediction</strong> to see results.</p>
        </div>
        """, unsafe_allow_html=True)

        # Demographics Section
        st.markdown('<div class="section-label">👤 Customer Demographics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3) 

        with col1: 
            age = st.slider("Age", 18, 95, 35, help="Customer's age in years") 
            job = st.selectbox("Occupation", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                                              'management', 'retired', 'self-employed', 'services',
                                              'student', 'technician', 'unemployed', 'unknown'],
                               help="Customer's job type") 
        with col2: 
            marital = st.selectbox("Marital Status", ['single', 'married', 'divorced', 'unknown'])
            education = st.selectbox("Education Level", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                                         'illiterate', 'professional.course',
                                                         'university.degree', 'unknown'],
                                     help="Highest education completed")
        with col3: 
            default = st.selectbox("Credit Default", ['no', 'yes', 'unknown'],
                                   help="Has the customer defaulted on credit?")
            housing = st.selectbox("Housing Loan", ['no', 'yes', 'unknown'],
                                   help="Does the customer have a housing loan?")

        st.markdown("---") 

        # Contact & Campaign Section
        st.markdown('<div class="section-label">📞 Contact & Campaign Details</div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)

        with col4:
            loan = st.selectbox("Personal Loan", ['no', 'yes', 'unknown'],
                                help="Does the customer have a personal loan?")
            contact = st.selectbox("Contact Method", ['cellular', 'telephone'],
                                   help="How was the customer contacted?")
        with col5:
            month = st.selectbox("Last Contact Month", ['jan','feb','mar','apr','may','jun',
                                                        'jul','aug','sep','oct','nov','dec'],
                                 help="Month of last contact in current campaign")
            day_of_week = st.selectbox("Last Contact Day", ['mon','tue','wed','thu','fri'],
                                       help="Day of last contact in current campaign")
        with col6:
            pdays = st.slider("Days Since Last Contact", 0, 999, 999,
                              help="Number of days since last contact from a previous campaign. 999 = customer was never contacted before")
            previous = st.slider("Previous Campaign Contacts", 0, 10, 0,
                                 help="How many times was this customer contacted in previous campaigns?")

        st.markdown("---")

        # Economic Indicators Section
        st.markdown('<div class="section-label">🌍 Economic Indicators & Previous Outcome</div>', unsafe_allow_html=True)
        col7, col8 = st.columns(2) 

        with col7:
            poutcome = st.selectbox("Previous Campaign Outcome", ['nonexistent', 'failure', 'success'],
                                    help="Outcome of the previous marketing campaign for this customer")
            emp_var_rate = st.slider("Employment Variation Rate", -3.5, 1.5, 1.1, 0.1,
                                     help="Quarterly employment rate change. Negative = declining economy")
            cons_price_idx = st.slider("Consumer Price Index", 92.0, 95.0, 93.75, 0.01,
                                       help="Monthly indicator of consumer goods price changes")
        with col8:
            cons_conf_idx = st.slider("Consumer Confidence Index", -51.0, -26.0, -40.0, 0.1,
                                      help="Monthly indicator of consumer economic confidence. More negative = less confident")
            euribor3m = st.slider("Euribor 3-Month Rate", 0.5, 5.5, 4.5, 0.01,
                                   help="European interbank lending rate. Lower rate = looser monetary policy")
            nr_employed = st.slider("Employed (thousands)", 4900.0, 5300.0, 5100.0, 1.0,
                                    help="Quarterly average number of employees in Portugal (thousands)")

        st.markdown("---")

        # Predict Button
        if st.button("Run Prediction"):

            # INPUT VALIDATION
            # To warn user potentially contradictory inputs
            has_warning = False

            # Check: pdays set but previous = 0 
            if pdays != 999 and previous == 0:
                st.warning("⚠️ **Input Check:** You set days since last contact but previous contacts is 0. If the customer was contacted before, previous contacts should be ≥ 1.")
                has_warning = True

            # Check: previous outcome = success but never contacted
            if poutcome == 'success' and pdays == 999:
                st.warning("⚠️ **Input Check:** Previous outcome is 'success' but days since contact is 999 (never contacted). These are contradictory - please verify.")
                has_warning = True

            # Check: previous outcome set but previous contacts = 0
            if poutcome != 'nonexistent' and previous == 0:
                st.warning("⚠️ **Input Check:** Previous outcome is set but previous contacts is 0. If there was a previous campaign, contacts should be ≥ 1.")
                has_warning = True

            # Check: very young person but if job retired
            if age < 25 and job == 'retired':
                st.warning("⚠️ **Input Check:** Customer is under 25 but listed as retired - please verify age and occupation.")
                has_warning = True

            # Check: elderly person listed as student
            if age > 65 and job == 'student':
                st.warning("⚠️ **Input Check:** Customer is over 65 but listed as student - please verify age and occupation.")
                has_warning = True

            # data frame need for user input
            
            input_data = pd.DataFrame({
                'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
                'default': [default], 'housing': [housing], 'loan': [loan], 'contact': [contact],
                'month': [month], 'day_of_week': [day_of_week], 'pdays': [pdays],
                'previous': [previous], 'poutcome': [poutcome], 'emp.var.rate': [emp_var_rate],
                'cons.price.idx': [cons_price_idx], 'cons.conf.idx': [cons_conf_idx],
                'euribor3m': [euribor3m], 'nr.employed': [nr_employed]
            })

            try:
                # RUN PREDICTION
                with st.spinner("Analysing customer profile..."):
                    
                    processed = preprocess_input(input_data, feature_columns, scaler, emp_median, nr_median)
                    
                    prediction = model.predict(processed)[0]
                    # model probability after predict
                    probability = model.predict_proba(processed)[0][1] if hasattr(model, 'predict_proba') else (0.7 if prediction == 1 else 0.3)

                # CUSTOMER PROFILE SUMMARY                
                st.markdown("---")
                pdays_display = "Never contacted" if pdays == 999 else f"{pdays} days ago"
                st.markdown(f"""
                <div class="profile-summary">
                    <div class="profile-title">👤 Customer Profile Summary</div>
                    <div class="profile-grid">
                        <div class="profile-item">
                            <span class="p-label">Age</span>
                            <span class="p-value">{age}</span>
                        </div>
                        <div class="profile-item">
                            <span class="p-label">Job</span>
                            <span class="p-value">{job}</span>
                        </div>
                        <div class="profile-item">
                            <span class="p-label">Marital</span>
                            <span class="p-value">{marital}</span>
                        </div>
                        <div class="profile-item">
                            <span class="p-label">Education</span>
                            <span class="p-value">{education}</span>
                        </div>
                        <div class="profile-item">
                            <span class="p-label">Credit Default</span>
                            <span class="p-value">{default}</span>
                        </div>
                        <div class="profile-item">
                            <span class="p-label">Housing Loan</span>
                            <span class="p-value">{housing}</span>
                        </div>
                        <div class="profile-item">
                            <span class="p-label">Personal Loan</span>
                            <span class="p-value">{loan}</span>
                        </div>
                        <div class="profile-item">
                            <span class="p-label">Contact</span>
                            <span class="p-value">{contact}</span>
                        </div>
                        <div class="profile-item">
                            <span class="p-label">Last Contact</span>
                            <span class="p-value">{pdays_display}</span>
                        </div>
                        <div class="profile-item">
                            <span class="p-label">Previous Outcome</span>
                            <span class="p-value">{poutcome}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # PREDICTION RESULTS
                st.markdown("### Prediction Results")
                r1, r2 = st.columns([1, 1]) 

                with r1: 
                    st.plotly_chart(create_gauge_chart(probability), use_container_width=True)

                with r2: 
                    if prediction == 1: # YES
                        st.markdown(f"""
                        <div class="result-yes">
                            <h2>✅ LIKELY TO SUBSCRIBE</h2>
                            <div class="conf">{probability*100:.1f}% confidence</div>
                            <p>High potential - prioritise this customer for follow-up calls</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else: # NO
                        st.markdown(f"""
                        <div class="result-no">
                            <h2>❌ UNLIKELY TO SUBSCRIBE</h2>
                            <div class="conf">{(1-probability)*100:.1f}% confidence (non-subscription)</div>
                            <p>Low potential - deprioritise and focus on higher-probability prospects</p>
                        </div>
                        """, unsafe_allow_html=True)
                # RECOMMENDED ACTIONS
                st.markdown("---")
                actions = generate_recommendations(
                    prediction, probability, age, job, poutcome, emp_var_rate,
                    pdays, previous, default, housing, loan, contact, education
                )
                action_label = "🟢 Recommended Actions - High Potential" if prediction == 1 else "🔴 Recommended Actions - Low Potential"
            
                actions_html = "".join([
                    f'<div class="action-item"><span class="action-bullet">→</span><span class="action-text">{a}</span></div>'
                    for a in actions
                ])
                st.markdown(f"""
                <div class="action-card">
                    <div class="action-title">{action_label}</div>
                    {actions_html}
                </div>
                """, unsafe_allow_html=True)

                # BUSINESS INSIGHT CARDS
                st.markdown("### Business Insights")
                i1, i2 = st.columns(2)
                with i1: 
                    st.markdown(f"""
                    <div class="card">
                        <h4>📋 Campaign History</h4>
                        <p>Previous outcome: <strong>{poutcome}</strong><br>
                        {'✅ Prior success - strong positive signal for subscription' if poutcome == 'success'
                         else '📌 No prior success - focus on relationship building first'}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with i2: 
                    st.markdown(f"""
                    <div class="card">
                        <h4>🌍 Economic Context</h4>
                        <p>Employment Variation Rate: <strong>{emp_var_rate}</strong><br>
                        {'⚠️ Strong economy - customers may prefer higher-risk investments over term deposits' if emp_var_rate > 0
                         else '✅ Weaker economy - customers seek safe investments like term deposits'}</p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                # anything goes wrong during prediction
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("💡 Please ensure all model files (best_model.pkl, scaler.pkl, feature_columns.pkl) are present and the input values are valid.")





    # TAB 2: PERFORMANCE
    with tab2:
        st.markdown("""
        <div class="section-header">
            <h3>Model Performance</h3>
            <p>Comparison of model metrics across iterative improvements, showcasing how tuning and class balancing boosted recall to find more subscribers.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-row">
            <div class="metric-pill">
                <div class="value">48.58%</div>
                <div class="label">F1-Score</div>
            </div>
            <div class="metric-pill">
                <div class="value">52.37%</div>
                <div class="label">Recall</div>
            </div>
            <div class="metric-pill">
                <div class="value">729</div>
                <div class="label">Subscribers Found</div>
            </div>
            <div class="metric-pill">
                <div class="value">+65%</div>
                <div class="label">Recall Improvement</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Before vs After comparison
        st.markdown("#### Before vs After Tuning")
        st.markdown("""
        | Metric | Iteration 1 (Default) | Iteration 3 (Tuned) | Change |
        |--------|----------------------|---------------------|--------|
        | **F1-Score** | 39.52% | **48.58%** | +22.9% |
        | **Recall** | 31.68% | **52.37%** | +65.3% |
        | **Subscribers Found** | ~441 | **729** | +288 more |
        """)

        st.markdown("---")

        
        st.markdown("#### Iterative Improvement Journey")
        st.markdown("""
        <div class="journey">
            <div class="num">0</div>
            <div class="content">
                <div class="title">Baseline - DummyClassifier</div>
                <div class="desc">Always predicts "No". 88.73% accuracy but 0% recall - completely misses all subscribers.</div>
            </div>
        </div>
        <div class="journey">
            <div class="num">1</div>
            <div class="content">
                <div class="title">4 Default Models Compared</div>
                <div class="desc">Logistic Regression, Decision Tree, Random Forest, Gradient Boosting. Random Forest best: F1 = 0.3952, Recall = 0.3168</div>
            </div>
        </div>
        <div class="journey">
            <div class="num">2</div>
            <div class="content">
                <div class="title">class_weight='balanced' Applied</div>
                <div class="desc">Forces models to pay equal attention to minority "Yes" class. Dramatically improved recall across all models.</div>
            </div>
        </div>
        <div class="journey">
            <div class="num">3</div>
            <div class="content">
                <div class="title">RandomizedSearchCV Hyperparameter Tuning</div>
                <div class="desc">Optimised hyperparameters with 5-fold cross-validation. Final: Random Forest (tuned) - F1 = 0.4858, Recall = 0.5237. Now finds 729 out of 1,392 subscribers.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1: # show user top predictive features
            st.markdown("""
            <div class="card">
                <h4>🏆 Top Predictive Features</h4>
                <p>
                <strong>1. euribor3m</strong> - European interbank interest rate<br>
                <strong>2. nr.employed</strong> - Employment level in economy<br>
                <strong>3. emp.var.rate</strong> - Employment variation rate<br>
                <strong>4. cons.conf.idx</strong> - Consumer confidence index<br>
                <strong>5. age</strong> - Customer age<br><br>
                <em>Economic indicators dominate ~40% of total importance - this makes business sense since economic conditions heavily influence whether customers invest in term deposits.</em>
                </p>
            </div>
            """, unsafe_allow_html=True)


        with c2: # ngineered features importance
            st.markdown("""
            <div class="card">
                <h4>🔧 Engineered Features (~11.5% Total)</h4>
                <p>
                <strong>age_group</strong> - 3.1% importance<br>
                <strong>contact_recency</strong> - 2.9% importance<br>
                <strong>contacted_before</strong> - 2.3% importance<br>
                <strong>economic_condition</strong> - 1.7% importance<br>
                <strong>prev_success</strong> - 1.5% importance<br><br>
                <em>All 5 engineered features add meaningful predictive signal that raw features alone cannot capture.</em>
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")


        st.markdown("""
        <div class="card">
            <h4>⚖️ Why Accuracy Decreased (And Why That's OK)</h4>
            <p>Accuracy dropped from ~89% (baseline) to ~87% (final model). This is <strong>acceptable</strong> because:</p>
            <p>
            • The baseline achieves 89% accuracy by always predicting "No" - it finds <strong>zero</strong> subscribers<br>
            • Our model sacrifices 2% accuracy to find <strong>729 real subscribers</strong> which is +288 more !<br>
            • Cost of missing a subscriber (lost term deposit revenue) far exceeds the cost of an extra phone call
            </p>
        </div>
        """, unsafe_allow_html=True)

    # TAB 3: HOW IT WORKS
    with tab3:
        st.markdown("""
        <div class="section-header">
            <h3>How BankConvert AI Works</h3>
            <p>A breakdown of the ML pipeline - from business problem to deployed prediction model.</p>
        </div>
        """, unsafe_allow_html=True)

        # Business problem card
        st.markdown("""
        <div class="card">
            <h4>🎯 The Business Problem</h4>
            <p>Banks use phone campaigns to promote term deposits, but only <strong>~11%</strong> of customers subscribe.
            Relationship Managers waste hours calling uninterested customers. BankConvert AI predicts <strong>before the call</strong>
            which customers are most likely to subscribe - allowing RMs to prioritise their time and increase conversion rates.</p>
        </div>
        """, unsafe_allow_html=True)

        # ML Pipeline Workflow Diagram
        st.markdown("""
        <div class="pipeline">
            <div class="step">
                <div class="step-icon">📊</div>
                <div class="step-label">Customer Data</div>
                <div class="step-desc">18 features input</div>
            </div>
            <div class="arrow">→</div>
            <div class="step">
                <div class="step-icon">🧹</div>
                <div class="step-label">Preprocessing</div>
                <div class="step-desc">Clean & encode</div>
            </div>
            <div class="arrow">→</div>
            <div class="step">
                <div class="step-icon">🔧</div>
                <div class="step-label">Feature Eng.</div>
                <div class="step-desc">5 new features</div>
            </div>
            <div class="arrow">→</div>
            <div class="step">
                <div class="step-icon">🤖</div>
                <div class="step-label">ML Model</div>
                <div class="step-desc">Random Forest</div>
            </div>
            <div class="arrow">→</div>
            <div class="step">
                <div class="step-icon">✅</div>
                <div class="step-label">Prediction</div>
                <div class="step-desc">Yes / No + %</div>
            </div>
            <div class="arrow">→</div>
            <div class="step">
                <div class="step-icon">📞</div>
                <div class="step-label">RM Action</div>
                <div class="step-desc">Prioritised calls</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="card">
                <h4>📊 Data Preprocessing</h4>
                <p>
                • <strong>One-Hot Encoding</strong> for all categorical variables<br>
                • <strong>StandardScaler</strong> for numerical features<br>
                • <strong>70/30 train-test split</strong> with stratification<br>
                • Removed <strong>duration</strong> & <strong>campaign</strong> (data leakage prevention)
                </p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="card">
                <h4>🔧 Feature Engineering (5 New Features)</h4>
                <p>
                • <strong>age_group</strong> - Young / Middle / Senior / Elderly<br>
                • <strong>contacted_before</strong> - Was customer contacted before? (binary)<br>
                • <strong>prev_success</strong> - Did previous campaign succeed? (binary)<br>
                • <strong>economic_condition</strong> - Good / Neutral / Bad (median-based)<br>
                • <strong>contact_recency</strong> - Never / Recent / Medium / Long
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h4>⚠️ Data Leakage Prevention</h4>
            <p>Two features were <strong>removed</strong> because they are only available <strong>after</strong> the call ends:</p>
            <p>
            • <strong>duration</strong> - Call length is only known after the call ends<br>
            • <strong>campaign</strong> - Total number of contacts is only known after all calls are made
            </p>
            <p>Since the goal is to predict subscription <strong>before making the call</strong>, including these features would
            create unrealistically high accuracy that cannot be replicated in real-world deployment.</p>
        </div>
        """, unsafe_allow_html=True)


        # Model selection & tuning card
        st.markdown("""
        <div class="card">
            <h4>🤖 Model Selection & Tuning</h4>
            <p>
            • <strong>4 algorithms compared:</strong> Logistic Regression, Decision Tree, Random Forest, Gradient Boosting<br>
            • <strong>Class imbalance handling:</strong> class_weight='balanced' to address 89/11 class split<br>
            • <strong>Hyperparameter tuning:</strong> RandomizedSearchCV with 5-fold cross-validation<br>
            • <strong>Winner:</strong> Random Forest (tuned) - best balance of F1-Score and Recall
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Before vs After Comparison
        st.markdown("---")
        st.markdown("""
        <div class="compare-container">
            <div class="compare-side compare-before">
                <div class="compare-icon">❌</div>
                <div class="compare-title">Without BankConvert AI</div>
                <div class="compare-stat">
                    <strong>Random calling</strong> from full customer list<br>
                    Only <strong>11% conversion rate</strong><br>
                    E.g. 500 calls → ~55 subscribers<br>
                    <strong>Wasted calls</strong> on uninterested customers<br>
                    RMs spend hours with low return
                </div>
            </div>
            <div class="compare-vs">VS</div>
            <div class="compare-side compare-after">
                <div class="compare-icon">✅</div>
                <div class="compare-title">With BankConvert AI</div>
                <div class="compare-stat">
                    <strong>Prioritised calling</strong> based on ML predictions<br>
                    <strong>52% recall</strong> on flagged customers<br>
                    Focus on top prospects first<br>
                    <strong>Fewer wasted calls</strong>, higher conversion<br>
                    RMs maximise their valuable time
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # TAB 4: ABOUT
    with tab4:
        st.markdown("""
        <div class="section-header">
            <h3>About This Project</h3>
            <p>Project details, dataset information, and technical specifications.</p>
        </div>
        """, unsafe_allow_html=True)

        # Project information card
        st.markdown("""
        <div class="card">
            <h4>📌 Project Information</h4>
            <p>
            <strong>Module:</strong> Machine Learning for Developers<br>
            <strong>Institution:</strong> Temasek Polytechnic<br>
            <strong>Developer:</strong> Melissa Kuah (2404487G)<br>
            <strong>Final Model:</strong> Random Forest (tuned) with class_weight='balanced'<br>
            <strong>Key Metrics:</strong> F1-Score = 48.58% · Recall = 52.37% · Finds 729/1,392 subscribers
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h4>📂 Dataset</h4>
            <p>
            <strong>UCI Bank Marketing Dataset</strong><br>
            • 41,188 real customer interactions from a Portuguese banking institution<br>
            • Period: May 2008 - November 2010<br>
            • Source: UCI Machine Learning Repository
            </p>
            <p><em>https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing</em></p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="card">
                <h4>🛠️ Tech Stack</h4>
                <p>
                <strong>ML Library:</strong> scikit-learn<br>
                &nbsp;&nbsp;- RandomForestClassifier<br>
                &nbsp;&nbsp;- StandardScaler<br>
                &nbsp;&nbsp;- RandomizedSearchCV<br>
                <strong>Web App:</strong> Streamlit + Plotly<br>
                <strong>Data:</strong> pandas, numpy<br>
                <strong>Imports:</strong> joblib (.pkl files)
                </p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="card">
                <h4>🧪 ML Techniques Used</h4>
                <p>
                <strong>1.</strong> One-Hot Encoding + StandardScaler<br>
                <strong>2.</strong> Stratified train-test split (70/30)<br>
                <strong>3.</strong> 5 engineered features from dataset<br>
                <strong>4.</strong> class_weight='balanced' for imbalanced dataset<br>
                <strong>5.</strong> RandomizedSearchCV (5-fold CV)<br>
                <strong>6.</strong> F1 & Recall prioritised over Accuracy
                </p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
#---------------------------------------------------------------------------------------------------------