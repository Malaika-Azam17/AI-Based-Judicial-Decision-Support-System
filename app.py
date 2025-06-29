import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
from openai import OpenAI
import together
import pyperclip  # For clipboard functionality

import google.generativeai as genai
from datetime import datetime, timedelta

# Initialize Together AI
together.api_key = "tgp_v1_0S3p3b-KDrtTrDyX9LZ44wMHze4gZQRbRMvVKFPcEwg"  # Replace with your actual API key

# Together AI integration with better error handling
def generate_ai_reasoning(user_input, prediction, raw_data_path):
    try:
        # Prepare context
        context = f"""As a Supreme Court Legal Advisor, analyze this criminal case sentencing prediction and provide:
        1. Relevant laws in bullet points
        2. References to 2-3 similar cases
        3. Identification of mitigating/aggravating factors
        4. Keep analysis under 500 words

        Predicted Sentence: {prediction}
        Case Factors:
        """
        context += "\n".join(f"- {k}: {v}" for k,v in user_input.items())
        
        # Generate response using Together AI
        response = together.Complete.create(
            prompt=context,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            max_tokens=1000,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Debug: Print raw response for troubleshooting
        print("Raw API Response:", response)
        
        # Handle different possible response formats
        if isinstance(response, dict):
            if 'output' in response and 'choices' in response['output'] and len(response['output']['choices']) > 0:
                return response['output']['choices'][0]['text']
            elif 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['text']
            else:
                return "⚠️ AI Analysis: Unable to parse API response format. Please check API documentation."
        elif isinstance(response, str):
            return response
        else:
            return "⚠️ AI Analysis: Unexpected response type received from API."
        
    except Exception as e:
        error_msg = f"⚠️ AI Service Error: {str(e)}"
        print(error_msg)  # Log the error for debugging
        return error_msg + "\nPlease try again later or contact support."

def together_ai_case_analysis(case_details):
    try:
        prompt = f"""Perform a comprehensive legal analysis of this criminal case:
        
        Case Details:
        {case_details}

        Provide:
        1. Potential legal arguments for both prosecution and defense
        2. Relevant case law precedents
        3. Sentencing guidelines analysis
        4. Risk assessment factors
        5. Recommendations for judicial considerations
        
        Structure your response with clear headings and bullet points."""
        
        response = together.Complete.create(
            prompt=prompt,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            max_tokens=1500,
            temperature=0.6,
            top_k=60,
            top_p=0.85
        )
        
        # Debug: Print raw response for troubleshooting
        print("Raw API Response:", response)
        
        # Handle different possible response formats
        if isinstance(response, dict):
            if 'output' in response and 'choices' in response['output'] and len(response['output']['choices']) > 0:
                return response['output']['choices'][0]['text']
            elif 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['text']
            else:
                return "⚠️ AI Analysis: Unable to parse API response format. Please check API documentation."
        elif isinstance(response, str):
            return response
        else:
            return "⚠️ AI Analysis: Unexpected response type received from API."
    
    except Exception as e:
        error_msg = f"AI Service Error: {str(e)}"
        print(error_msg)  # Log the error for debugging
        return error_msg + "\nPlease try again later."

# Load model and encoders
model = joblib.load("../models/random_forest_model.pkl")
encoders = joblib.load("../models/label_encoders.pkl")

# Sample datasets (replace with your actual data paths)
RAW_DATA_PATH = "../data/Pakistan_Crime_Dataset.csv"
PROCESSED_DATA_PATH = "../data/Pakistan_Crime_Dataset_Balanced.csv"


# Streamlit title config
st.set_page_config(
    page_title="AI Judicial Decision Support",
    layout="wide",
    page_icon="⚖️"
)

# Initialize session state for settings
if 'notification_preference' not in st.session_state:
    st.session_state.notification_preference = "Email"
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = True
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'profile_data' not in st.session_state:
      st.session_state.profile_data = {
        "name": "Judge John Doe",
        "id": "JD123456",
        "court": "District Court",
        "email": "john.doe@judicial.gov",
        "qualifications": [
            "Juris Doctor, Harvard Law School, 1995",
            "Bachelor of Arts, Yale University, 1992",
            "Certified Mediator, 2000"
        ],
        "achievements": [
            {"title": "Outstanding Jurist Award", "year": "2020", "icon": "🏆"},
            {"title": "Published Author", "year": "Criminal Law Review", "icon": "📜"},
            {"title": "Guest Lecturer", "year": "Stanford Law School", "icon": "🎓"}
        ],
        "profile_image": None
    }



def get_theme_css():
    theme = st.session_state.get('theme', 'Light')
    return f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

            :root {{
                /* Base Colors */
                --primary: #09b7ba;
                --secondary: #09b7ba;
                --accent: #4fc3f7;
                --danger: #09b7ba;
                --success: #10b981;

                /* Theme-Specific Colors */
                --background: {('#1a1a2e' if theme == 'Dark' else '#f8f9fa')};
                --card: {('#2a2a3e' if theme == 'Dark' else '#ffffff')};
                --text: {('#e0e0e0' if theme == 'Dark' else '#2c3e50')};
                --sidebar-bg: #2ea89e;
                --metric-bg: {('#3a3a4e' if theme == 'Dark' else '#e6f3ff')};
                --table-header-bg: {('#3a3a4e' if theme == 'Dark' else '#e6f3ff')};
                --table-row-bg: {('#2a2a3e' if theme == 'Dark' else '#f9fcff')};
                --table-row-hover: {('#3a3a4e' if theme == 'Dark' else '#e6f3ff')};

                /* Typography */
                --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                --font-size-base: 16px;
                --font-size-heading: 1.75rem;
                --font-size-subheading: 1.25rem;
            }}

            /* Global Styles */
            .stApp {{
                background-color: var(--background);
                font-family: var(--font-family);
                font-size: var(--font-size-base);
                color: var(--text);
                line-height: 1.6;
            }}

            /* Header Styles */
            .header-style {{
                text-align: center;
                color: var(--secondary);
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: -10px;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
                animation: fadeIn 0.5s ease-out;
            }}

            .subheader {{
                text-align: center;
                color:#09b7ba !important;
                font-size: var(--font-size-subheading);
                margin-bottom:2rem;
                font-weight: 500;
            }}

            /* Sidebar Styles */
            [data-testid="stSidebar"] {{
                background: var(--sidebar-bg);
                padding: 1.5rem;
                border-right: 1px solid {('#4a4a6a' if theme == 'Dark' else '#d1d5db')};
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            }}

            [data-testid="stSidebar"] h1 {{
                color: white !important;
                font-size: 1.8rem !important;
                margin-bottom: 1.5rem !important;
                text-align: center;
                font-weight: 700;
            }}

            [data-testid="stSidebarNavLink"], .sidebar .stRadio label {{
                color: white !important;
                font-weight: 500;
                font-size: 1.1rem !important;
                padding: 0.75rem 1rem;
                border-radius: 8px;
                transition: all 0.3s ease;
                margin: 0.25rem 0;
                border: 1px solid transparent;
                background: rgba(255,255,255,0.1);
            }}

            [data-testid="stSidebarNavLink"]:hover, .sidebar .stRadio label:hover {{
                background-color: rgba(79, 195, 247, 0.3);
                border-color: var(--accent);
                transform: translateY(-2px);
            }}
            
            /* Card Styles */
            .card, .profile-card, .guideline-card, .PremiumTab, .edit-form, .logout-card {{
                background: {('linear-gradient(135deg, var(--card) 0%, #3a3a4e 100%)' if theme == 'Dark' else 'linear-gradient(135deg, var(--card) 0%, #e6f3ff 100%)')};
                border-radius: 16px;
                padding: 2rem;
                box-shadow: 0 6px 18px rgba(0,0,0,0.1);
                margin-bottom: 1.5rem;
                border: 1px solid {('rgba(79, 195, 247, 0.2)' if theme == 'Dark' else '#e2e8f0')};
                transition: all 0.3s ease;
                animation: slideIn 0.5s ease-out;
            }}

            .card:hover, .profile-card:hover, .guideline-card:hover, .PremiumTab:hover {{
                transform: translateY(-4px);
                box-shadow: 0 12px 24px rgba(0,0,0,0.15);
            }}

            .card h3, .profile-card h3, .guideline-card h3 {{
                color: var(--accent);
                font-size: var(--font-size-heading);
                margin-bottom: 1rem;
                position: relative;
            }}

            .card h3::after, .profile-card h3::after, .guideline-card h3::after {{
                content: '';
                position: absolute;
                bottom: -4px;
                left: 0;
                width: 60px;
                height: 3px;
                background: var(--accent);
            }}

            .achievement-card, .qualification-card, .faq-item {{
                background: {('rgba(79, 195, 247, 0.1)' if theme == 'Dark' else 'white')};
                border-radius: 10px;
                padding: 1rem 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 3px 8px rgba(0,0,0,0.05);
                border-left: 4px solid var(--accent);
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 1rem;
            }}

            .achievement-card:hover, .qualification-card:hover, .faq-item:hover {{
                transform: translateY(-3px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.1);
                background: {('rgba(79, 195, 247, 0.2)' if theme == 'Dark' else '#f8f9fa')};
            }}

            .achievement-icon {{
                font-size: 1.3rem;
                margin-right: 0.75rem;
            }}

        

            /* Tab Styles */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.75rem;
                background: {('#2a2a3e' if theme == 'Dark' else '#e5e7eb')};
                padding: 0.5rem;
                border-radius: 12px;
            }}

            .stTabs [data-baseweb="tab"] {{
                height: 44px;
                padding: 0 1.5rem;
                background-color: {('#3a3a4e' if theme == 'Dark' else '#f3f4f6')};
                border-radius: 8px;
                border: none;
                font-weight: 600;
                color: var(--text);
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1rem;
            }}

            .stTabs [data-baseweb="tab"]:hover {{
                background-color: {('#4a4a6a' if theme == 'Dark' else '#e5e7eb')};
                transform: translateY(-2px);
            }}

            .stTabs [aria-selected="true"] {{
                background-color: var(--accent) !important;
                color: white !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}

            /* Metric Styles */
            [data-testid="stMetricValue"] {{
                font-size: 1.1rem !important;
                color: {('#4fc3f7' if theme == 'Dark' else '#166088')} !important;
            }}

            [data-testid="stMetricLabel"] {{
                font-size: 0.9rem !important;
                color: var(--primary) !important;
            }}

            .metric-card {{
                background: var(--metric-bg);
                border-radius: 10px;
                padding: 1rem;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                border: 1px solid {('#4a4a6a' if theme == 'Dark' else '#d1e9ff')};
                transition: transform 0.3s ease;
            }}

            .metric-card:hover {{
                transform: translateY(-2px);
            }}

          

            /* Success and Error Messages */
            .success-message {{
                background: {('rgba(46, 125, 50, 0.2)' if theme == 'Dark' else '#d1fae5')};
                color: var(--success);
                padding: 1rem 1.5rem;
                border-radius: 8px;
                margin: 1rem 0;
                border-left: 4px solid var(--success);
                font-weight: 500;
                animation: fadeIn 0.5s ease-out;
            }}

            .error-message {{
                background: {('rgba(230, 57, 70, 0.2)' if theme == 'Dark' else '#fee2e2')};
                color: var(--danger);
                padding: 1rem 1.5rem;
                border-radius: 8px;
                margin: 1rem 0;
                border-left: 4px solid var(--danger);
                font-weight: 500;
                animation: fadeIn 0.5s ease-out;
            }}

            /* Feature Box */
            .feature-box {{
                background: {('rgba(79, 195, 247, 0.2)' if theme == 'Dark' else 'rgba(79, 195, 247, 0.1)')};
                border-left: 4px solid var(--accent);
                padding: 1rem;
                margin-bottom: 1rem;
                border-radius: 0 8px 8px 0;
                animation: fadeIn 0.5s ease-out;
            }}

            /* Links */
            a {{
                color: var(--accent);
                text-decoration: none;
                font-weight: 500;
                transition: color 0.3s ease;
            }}

            a:hover {{
                color: #3da8d8;
                text-decoration: underline;
            }}

            /* Responsive Design */
            @media (max-width: 768px) {{
                .profile-image-container {{
                    width: 120px;
                    height: 120px;
                }}

                .camera-icon {{
                    width: 30px;
                    height: 30px;
                    font-size: 14px;
                }}

                .header-style {{
                    font-size: 2rem;
                }}

                .stTabs [data-baseweb="tab"] {{
                    font-size: 0.9rem;
                    padding: 0 1rem;
                }}

            }}

            
        </style>
    """

# Apply theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)

# Header
st.markdown("<div class='header-style'>AI Judicial Decision Support System</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Advanced Analytics for Sentencing Outcome Predictions</div>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("<h1>Navigation Menu</h1>", unsafe_allow_html=True)
tabs = [
    "Dashboard",
    "New Case Prediction", 
    "Prediction History",
    "Legal Guidelines",
    "Data Explorer",
    "Judicial Notes",
    "Profile"
]
choice = st.sidebar.radio("", tabs, label_visibility="collapsed")

# File for history
HISTORY_FILE = "prediction_history.csv"
NOTES_FILE = "notes.csv"

if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["Timestamp"] + [col for col in encoders if col != "Sentencing Outcome"] + ["Prediction"]).to_csv(HISTORY_FILE, index=False)

# State management for history deletion
if 'history_deleted' not in st.session_state:
    st.session_state.history_deleted = False

if not os.path.exists(NOTES_FILE):
    pd.DataFrame(columns=["Timestamp", "CaseID", "Note"]).to_csv(NOTES_FILE, index=False)


# Initialize session state for clipboard if not exists
if 'clipboard_msg' not in st.session_state:
    st.session_state.clipboard_msg = None



# ---------------------- Dashboard Tab ------------------------
if choice == "Dashboard":
    st.subheader("Judicial Analytics Dashboard")
    st.markdown("Real-time insights into case predictions and system performance")
    
    # Load data for dashboard
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
    else:
        history_df = pd.DataFrame()
    
    # Metrics row with updated styling
    col1, col2, col3, col4 = st.columns(4)
    
    st.markdown("""
    <style>
        /* Enhanced metric styling */
        [data-testid="stMetric"] {
            font-family: 'Inter', sans-serif;
            padding: 20px 10px !important;
            border-radius: 12px !important;
            background-color: #f8f9fa !important;
            border-left: 5px solid #09b7ba !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        [data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }
        
        [data-testid="stMetricLabel"] > div {
            font-size: 16px !important;
            font-weight: 600 !important;
            color: #4a4a4a !important;
            margin-bottom: 8px !important;
        }
        
        [data-testid="stMetricValue"] > div {
            font-size: 13px !important;
            font-weight: 700 !important;
            color: #09b7ba !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    with col1:
        total_predictions = len(history_df) if not history_df.empty else 0
        st.metric("Total Predictions", total_predictions)
    
    with col2:
        if not history_df.empty:
            most_common = history_df["Prediction"].mode()[0]
            st.metric("Most Common Outcome", most_common)
        else:
            st.metric("Most Common Outcome", "N/A")
    
    with col3:
        if not history_df.empty:
            latest_date = pd.to_datetime(history_df["Timestamp"]).max().strftime("%Y-%m-%d")
            st.metric("Last Prediction", latest_date)
        else:
            st.metric("Last Prediction", "N/A")
    
    with col4:
        if os.path.exists(RAW_DATA_PATH):
            raw_df = pd.read_csv(RAW_DATA_PATH)
            st.metric("Total Cases in Database", len(raw_df))
        else:
            st.metric("Total Cases in Database", "N/A")
    
    st.markdown("---")
    
    # Charts row with larger sizes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Outcome Distribution")
        if not history_df.empty:
            fig = px.pie(
                history_df,
                names="Prediction",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.3,
                width=300,  # Increased width
                height=500  # Increased height
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                # textfont_size=16,  # Larger font
                marker=dict(line=dict(color='#ffffff', width=1))
            )
            fig.update_layout(
                margin=dict(l=10, r=20, t=10, b=10),  # Adjust margins
                legend=dict(
                    font=dict(size=8),  # Larger legend
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction data available yet")
    
    # History Table Section with improved styling
    st.subheader("📋 Recent Prediction History")
    
    if not history_df.empty:
        # Limit to 10 most recent records for dashboard display
        display_df = history_df.sort_values("Timestamp", ascending=False).head(10)
        
        # Custom CSS for the table
        st.markdown("""
        <style>
            .stDataFrame {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            
            .stDataFrame th {
                background-color: #09b7ba !important;
                color: white !important;
                font-size: 15px !important;
                font-weight: 600 !important;
                padding: 12px !important;
            }
            
            .stDataFrame td {
                font-size: 14px !important;
                padding: 10px !important;
            }
            
            .stDataFrame tr:nth-child(even) {
                background-color: #f8f9fa !important;
            }
            
            .stDataFrame tr:hover {
                background-color: rgba(9, 183, 186, 0.1) !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Display the table with custom styling
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=(len(display_df) + 1) * 35 + 3,  # Dynamic height
            column_config={
                "Timestamp": st.column_config.DatetimeColumn(
                    "Timestamp",
                    format="YYYY-MM-DD HH:mm:ss",
                    width="medium"
                ),
                "Prediction": st.column_config.TextColumn(
                    "Prediction",
                    width="medium"
                ),
                "Reasoning": st.column_config.TextColumn(
                    "Reasoning",
                    width="large",
                    help="AI-generated legal analysis"
                )
            }
        )
        
        # View Full History button with improved styling
        st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <button onclick="window.location.href='/?nav=Prediction+History'" 
                    style="
                        background-color: #09b7ba;
                        color: white;
                        border: none;
                        padding: 12px 24px;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 16px;
                        font-weight: 500;
                        transition: all 0.3s ease;
                    "
                    onmouseover="this.style.backgroundColor='#008a8d'; this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.2)';"
                    onmouseout="this.style.backgroundColor='#09b7ba'; this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                View Full History →
            </button>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No prediction history available yet.")
            
# ---------------------- New Case Prediction Tab ------------------------

if choice == "New Case Prediction":
    with st.container():
        st.subheader("Case Details Form")
        st.markdown("Provide the case information to predict the sentencing outcome.")

        user_input = {}
        cols = st.columns(2)
        i = 0

        with st.form("prediction_form"):
            for feature in encoders:
                if feature != "Sentencing Outcome":
                    options = encoders[feature].classes_.tolist()
                    with cols[i % 2]:
                        with st.container():
                            st.markdown(
                                f"""
                                <div style='
                                    background-color: #09b7ba;
                                    color: white;
                                    padding: 10px;
                                    border-radius: 5px;
                                    margin-bottom: 10px;
                                '>
                                    <strong>{feature}</strong>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            user_input[feature] = st.selectbox(
                                f"Select {feature}",
                                options,
                                key=feature,
                                label_visibility="collapsed"
                            )
                    i += 1

            submitted = st.form_submit_button("Predict Sentencing Outcome")

        if submitted:
            with st.spinner("Analyzing case details..."):
                # Prepare input for model
                input_df = pd.DataFrame([user_input])
                for col in input_df.columns:
                    input_df[col] = encoders[col].transform(input_df[col])

                # Get prediction
                prediction = model.predict(input_df)[0]
                outcome = encoders["Sentencing Outcome"].inverse_transform([prediction])[0]

                # Display prediction result
                st.balloons()
                st.markdown(
                    f"""
                    <div style='
                        padding: 15px;
                        border: 2px solid #09b7ba;
                        border-radius: 5px;
                        background-color: #09b7ba;
                        color: white;
                        margin: 20px 0;
                        font-size: 18px;
                        font-weight: bold;
                        text-align: center;
                    '>
                        Predicted Outcome: {outcome}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Generate AI reasoning using Together AI
                with st.spinner("Generating detailed legal analysis..."):
                    ai_reasoning = generate_ai_reasoning(user_input, outcome, RAW_DATA_PATH)
                    
                    # Additional Together AI analysis
                    case_details = "\n".join(f"{k}: {v}" for k, v in user_input.items())
                    advanced_analysis = together_ai_case_analysis(case_details)

                # Display AI reasoning
                with st.expander("📜 AI-Powered Legal Analysis", expanded=True):
                    st.markdown(f"""
                    <div style='
                        background-color: #f8f9fa;
                        padding: 16px;
                        border-radius: 8px;
                        border-left: 4px solid #09b7ba;
                        margin-bottom: 16px;
                    '>
                        <h4>Basic Analysis</h4>
                        {ai_reasoning}
                        
                       
                        {advanced_analysis}
                    </div>
                    """, unsafe_allow_html=True)

                    # Add action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📋 Copy Analysis", key="copy_analysis"):
                            pyperclip.copy(f"{ai_reasoning}\n\n{advanced_analysis}")
                            st.toast("Analysis copied to clipboard!", icon="📋")
                    with col2:
                        if st.button("🗂️ Save as Note", key="save_note"):
                            notes_df = pd.read_csv(NOTES_FILE) if os.path.exists(NOTES_FILE) else pd.DataFrame(columns=["Timestamp", "CaseID", "Note"])
                            new_note = {
                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "CaseID": f"PRED-{datetime.now().strftime('%Y%m%d')}",
                                "Note": f"Prediction: {outcome}\n\nBasic Analysis:\n{ai_reasoning}\n\nAdvanced Analysis:\n{advanced_analysis}"
                            }
                            notes_df = pd.concat([notes_df, pd.DataFrame([new_note])], ignore_index=True)
                            notes_df.to_csv(NOTES_FILE, index=False)
                            st.toast("Analysis saved to notes!", icon="🗂️")

                # Save to history
                history_row = {
                    **user_input,
                    "Prediction": outcome,
                    "Reasoning": f"{ai_reasoning}\n\n{advanced_analysis}",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                history_df = pd.read_csv(HISTORY_FILE) if os.path.exists(HISTORY_FILE) else pd.DataFrame(columns=list(history_row.keys()))
                history_df = pd.concat([history_df, pd.DataFrame([history_row])], ignore_index=True)
                history_df.to_csv(HISTORY_FILE, index=False)

# Add a new section to the Legal Guidelines tab for AI-assisted research
    
 


# ---------------------- Prediction History Tab ------------------------
elif choice == "Prediction History":
    with st.container():
        st.subheader("Prediction History")
        st.markdown("Review all previous predictions made by the system.")
        
        if os.path.exists(HISTORY_FILE):
            history_df = pd.read_csv(HISTORY_FILE)
            
            # History management section
            with st.expander("🔧 History Management", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Delete Options**")
                    if st.button("🗑️ Delete ALL History", 
                               key="delete_all", 
                               help="Permanently delete all prediction history", 
                               use_container_width=True):
                        try:
                            os.remove(HISTORY_FILE)
                            st.session_state.history_deleted = True
                            st.success("All prediction history has been deleted!")
                            # Create empty history file
                            pd.DataFrame(columns=["Timestamp"] + [col for col in encoders if col != "Sentencing Outcome"] + ["Prediction"]).to_csv(HISTORY_FILE, index=False)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting history: {e}")
                
                with col2:
                    st.markdown("**Export Options**")
                    if st.button("📤 Export History as CSV", use_container_width=True):
                        if not history_df.empty:
                            csv = history_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="💾 Download CSV",
                                data=csv,
                                file_name=f"judicial_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime='text/csv',
                                use_container_width=True
                            )
                        else:
                            st.warning("No history data available to export.")
                
                with col3:
                    st.markdown("**Search Filters**")
                    search_term = st.text_input("Search cases...", 
                                               placeholder="Enter keywords",
                                               label_visibility="collapsed")
            
            # Check if history exists after potential deletion
            if st.session_state.history_deleted or not os.path.exists(HISTORY_FILE) or history_df.empty:
                st.info("No prediction history available yet.")
            else:
                # Apply search filter if any
                if search_term:
                    mask = history_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
                    history_df = history_df[mask]
                
                # Analytics cards
                st.markdown("""
                <style>
                    [data-testid="stMetric"] {
                        background: #f8f9fa;
                        border-radius: 10px;
                        padding: 15px;
                        border-left: 4px solid #09b7ba;
                    }
                    [data-testid="stMetricLabel"] {
                        font-size: 14px !important;
                    }
                    [data-testid="stMetricValue"] {
                        font-size: 24px !important;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(history_df))
                with col2:
                    if not history_df.empty:
                        most_common = history_df["Prediction"].mode()[0]
                        st.metric("Most Common Outcome", most_common)
                    else:
                        st.metric("Most Common Outcome", "N/A")
                with col3:
                    if not history_df.empty:
                        latest_date = pd.to_datetime(history_df["Timestamp"]).max().strftime("%Y-%m-%d")
                        st.metric("Last Prediction", latest_date)
                    else:
                        st.metric("Last Prediction", "N/A")
                
                if not history_df.empty:
                    # Main data table with improved styling
                    st.markdown("""
                    <style>
                        .stDataFrame {
                            border-radius: 8px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                        }
                        .stDataFrame div[data-testid="stDataFrameContainer"] {
                            height: 500px;
                        }
                        .stDataFrame th {
                            background-color: #09b7ba !important;
                            color: white !important;
                            position: sticky;
                            top: 0;
                        }
                        .stDataFrame tr:nth-child(even) {
                            background-color: #f8f9fa;
                        }
                        .stDataFrame tr:hover {
                            background-color: rgba(9, 183, 186, 0.1) !important;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(
                        history_df.sort_values("Timestamp", ascending=False),
                        use_container_width=True,
                        height=500,
                        hide_index=True,
                        column_config={
                            "Timestamp": st.column_config.DatetimeColumn(
                                "Timestamp",
                                format="YYYY-MM-DD HH:mm:ss",
                                width="medium"
                            ),
                            "Prediction": st.column_config.TextColumn(
                                "Prediction",
                                width="medium"
                            ),
                            "Reasoning": st.column_config.TextColumn(
                                "Reasoning",
                                width="large",
                                help="AI-generated legal analysis"
                            )
                        }
                    )
                    
                    # Visualization tabs
                    tab1 = st.tabs(["📊 Outcome Distribution"])
                    
                    with tab1:
                        st.subheader("Outcome Distribution")
                        fig = px.pie(
                            history_df,
                            names="Prediction",
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            hole=0.3,
                            width=600,
                            height=500
                        )
                        fig.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            textfont_size=16,
                            marker=dict(line=dict(color='#ffffff', width=1))
                        )
                        fig.update_layout(
                            margin=dict(l=20, r=20, t=40, b=20),
                            legend=dict(
                                font=dict(size=14),
                                orientation="h",
                                yanchor="bottom",
                                y=-0.1,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
        else:
            st.info("No prediction history available yet.")
# ---------------------- Case Analytics Tab ------------------------
elif choice == "Case Analytics":
    st.subheader("🔍 Case Analytics")
    st.markdown("Explore patterns and relationships in judicial decisions")
    
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        
        if not history_df.empty:
            # Feature selection
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Select X-axis feature", 
                                     [col for col in history_df.columns if col not in ['Timestamp', 'Prediction']])
            with col2:
                y_axis = st.selectbox("Select Y-axis feature", 
                                     [col for col in history_df.columns if col not in ['Timestamp', 'Prediction']])
            
            # Interactive cross-analysis
            st.subheader("Feature Correlation Analysis")
            cross_tab = pd.crosstab(history_df[x_axis], history_df[y_axis])
            st.dataframe(cross_tab.style.background_gradient(cmap='Blues'), 
                        use_container_width=True)
            
            # Visualization
            fig = px.density_heatmap(history_df, x=x_axis, y=y_axis, 
                                   color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            
            # Outcome by feature analysis
            st.subheader("Outcome by Feature Analysis")
            selected_feature = st.selectbox("Select feature to analyze outcomes", 
                                          [col for col in history_df.columns if col not in ['Timestamp', 'Prediction']])
            
            outcome_dist = pd.crosstab(history_df[selected_feature], history_df['Prediction'])
            fig = px.bar(outcome_dist, barmode='group', 
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data available for analytics. Please make some predictions first.")
    else:
        st.info("No prediction history available yet.")

# ---------------------- Legal Guidelines Tab ------------------------
elif choice == "Legal Guidelines":
    st.subheader("📚 Legal Guidelines")
    st.markdown("Reference materials and sentencing guidelines")
    
    tab1, tab2, tab3 = st.tabs(["Sentencing Framework", "Case Law References", "AI Legal Research"])
    
    with tab1:
        st.markdown("### Standard Sentencing Guidelines")
        st.markdown("Explore the structured guidelines for judicial sentencing decisions.")
        
        # Card-based layout for sentencing guidelines
        guidelines = [
            {
                "title": "Probation",
                "description": [
                    "Applicable for first-time offenders",
                    "Non-violent crimes",
                    "Low risk of recidivism"
                ],
                "link": "https://www.uscourts.gov/services-forms/probation-and-pretrial-services"
            },
            {
                "title": "Jail <1 Year",
                "description": [
                    "Repeat offenders",
                    "Minor violent offenses",
                    "Property crimes with aggravating factors"
                ],
                "link": "https://www.bjs.gov/index.cfm?ty=tp&tid=23"
            },
            {
                "title": "Prison 1-5 Years",
                "description": [
                    "Serious violent offenses",
                    "Repeat offenders with escalating severity",
                    "Major drug offenses"
                ],
                "link": "https://www.ussc.gov/guidelines"
            },
            {
                "title": "Prison 5+ Years",
                "description": [
                    "Severe violent crimes",
                    "Habitual offenders",
                    "Crimes with special circumstances"
                ],
                "link": "https://www.justice.gov/criminal/criminal-fraud/sentencing"
            }
        ]
        
        cols = st.columns(2)
        for idx, guideline in enumerate(guidelines):
            with cols[idx % 2]:
                st.markdown(f"""
                    <div class='guideline-card' style='padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;'>
                        <h3>{guideline['title']}</h3>
                        <ul>
                            {"".join(f"<li>{item}</li>" for item in guideline['description'])}
                        </ul>
                        <div style='text-align: center;'>
                            <a href='{guideline['link']}' target='_blank'>
                                <button style='
                                    background-color: #66b3ff;
                                    color: white;
                                    border: none;
                                    padding: 10px 20px;
                                    text-align: center;
                                    text-decoration: none;
                                    display: inline-block;
                                    font-size: 16px;
                                    margin: 10px 2px;
                                    cursor: pointer;
                                    border-radius: 5px;
                                    transition: all 0.3s ease;
                                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                                ' 
                                onmouseover="this.style.backgroundColor='#3399ff'; this.style.transform='scale(1.05)';" 
                                onmouseout="this.style.backgroundColor='#66b3ff'; this.style.transform='scale(1)';"
                                onclick="this.style.backgroundColor='#0080ff'; this.style.boxShadow='0 0 10px rgba(0,0,0,0.3)'; setTimeout(() => {{this.style.backgroundColor='#66b3ff'; this.style.boxShadow='0 2px 5px rgba(0,0,0,0.2)';}}, 300);">
                                    Learn More
                                </button>
                            </a>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Relevant Case Law Precedents")
        st.markdown("Key legal precedents influencing sentencing decisions.")
        
        # Card-based layout for case law references
        precedents = [
            {
                "title": "State v. Johnson (2018)",
                "description": [
                    "Established standards for drug offense sentencing",
                    "Differentiated between possession vs distribution"
                ],
                "link": "https://www.law.cornell.edu/wex/drug_trafficking"
            },
            {
                "title": "People v. Smith (2020)",
                "description": [
                    "Clarified sentencing for white-collar crimes",
                    "Considered financial impact as aggravating factor"
                ],
                "link": "https://www.justice.gov/criminal/criminal-fraud/white-collar-crime"
            },
            {
                "title": "State v. Williams (2021)",
                "description": [
                    "Updated guidelines for juvenile sentencing",
                    "Emphasized rehabilitation over punishment"
                ],
                "link": "https://ojjdp.ojp.gov/programs/juvenile-justice"
            }
        ]
        
        cols = st.columns(2)
        for idx, precedent in enumerate(precedents):
            with cols[idx % 2]:
                st.markdown(f"""
                    <div class='guideline-card' style='padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;'>
                        <h3>{precedent['title']}</h3>
                        <ul>
                            {"".join(f"<li>{item}</li>" for item in precedent['description'])}
                        </ul>
                        <div style='text-align: center;'>
                            <a href='{precedent['link']}' target='_blank'>
                                <button style='
                                    background-color: #66b3ff;
                                    color: white;
                                    border: none;
                                    padding: 10px 20px;
                                    text-align: center;
                                    text-decoration: none;
                                    display: inline-block;
                                    font-size: 16px;
                                    margin: 10px 2px;
                                    cursor: pointer;
                                    border-radius: 5px;
                                    transition: all 0.3s ease;
                                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                                ' 
                                onmouseover="this.style.backgroundColor='#3399ff'; this.style.transform='scale(1.05)';" 
                                onmouseout="this.style.backgroundColor='#66b3ff'; this.style.transform='scale(1)';"
                                onclick="this.style.backgroundColor='#0080ff'; this.style.boxShadow='0 0 10px rgba(0,0,0,0.3)'; setTimeout(() => {{this.style.backgroundColor='#66b3ff'; this.style.boxShadow='0 2px 5px rgba(0,0,0,0.2)';}}, 300);">
                                    Learn More
                                </button>
                            </a>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 🤖 AI-Powered Legal Research Assistant")
        st.markdown("Get instant analysis of legal questions using Together AI's advanced models.")
        
        research_query = st.text_area("Enter your legal research question:", 
                                    placeholder="E.g.: What are the key considerations for sentencing in white-collar fraud cases?",
                                    height=150)
        
        if st.button("Research with AI", type="primary"):
            if research_query:
                with st.spinner("Conducting legal research..."):
                    prompt = f"""As a legal research assistant, provide a comprehensive answer to this query:
                    
                    Query: {research_query}
                    
                    Include in your response:
                    1. Relevant statutes and laws
                    2. Key case precedents
                    3. Judicial interpretations
                    4. Current legal trends
                    5. Potential counterarguments
                    
                    Structure your response with clear headings and citations where appropriate."""
                    
                    try:
                        response = together.Complete.create(
                            prompt=prompt,
                            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                            max_tokens=1500,
                            temperature=0.5,
                            top_k=50,
                            top_p=0.8
                        )
                        
                        # Handle response format
                        if isinstance(response, dict):
                            if 'output' in response and 'choices' in response['output']:
                                research_result = response['output']['choices'][0]['text']
                            elif 'choices' in response:
                                research_result = response['choices'][0]['text']
                            else:
                                research_result = "Unable to parse API response format"
                        else:
                            research_result = response
                        
                        st.markdown(f"""
                        <div style='
                            background-color: #f8f9fa;
                            padding: 16px;
                            border-radius: 8px;
                            border-left: 4px solid #09b7ba;
                            margin-top: 20px;
                        '>
                            {research_result}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add action buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📋 Copy Research", key="copy_research"):
                                pyperclip.copy(research_result)
                                st.toast("Research copied to clipboard!", icon="📋")
                        with col2:
                            if st.button("💾 Save to Notes", key="save_research"):
                                notes_df = pd.read_csv(NOTES_FILE) if os.path.exists(NOTES_FILE) else pd.DataFrame(columns=["Timestamp", "CaseID", "Note"])
                                new_note = {
                                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "CaseID": "RESEARCH",
                                    "Note": f"Research Query: {research_query}\n\nResults:\n{research_result}"
                                }
                                notes_df = pd.concat([notes_df, pd.DataFrame([new_note])], ignore_index=True)
                                notes_df.to_csv(NOTES_FILE, index=False)
                                st.toast("Research saved to notes!", icon="💾")
                    except Exception as e:
                        st.error(f"AI service error: {str(e)}. Please try again later.")
            else:
                st.warning("Please enter a research question")
# ---------------------- Data Explorer Tab ------------------------
elif choice == "Data Explorer":
    with st.container():
        st.subheader("🔍 Data Explorer")
        st.markdown("Explore the raw and processed datasets used for model training.")
        
        tab1, tab2 = st.tabs(["Raw Dataset", "Processed Dataset"])
        
        with tab1:
            if os.path.exists(RAW_DATA_PATH):
                raw_df = pd.read_csv(RAW_DATA_PATH)
                
                # Dynamic filtering
                st.markdown("### Dynamic Data Exploration")
                col1, col2 = st.columns(2)
                with col1:
                    filter_column = st.selectbox("Filter by column", raw_df.columns)
                with col2:
                    filter_value = st.selectbox("Filter value", raw_df[filter_column].unique())
                
                filtered_df = raw_df[raw_df[filter_column] == filter_value]
                
                # Show record count
                st.markdown(f"**Showing {len(filtered_df)} records**")
                
                # Convert filtered dataframe to HTML with custom styling
                table_html = f"""
                <style>
                    .custom-table-container {{
                        max-height: 200px;
                        overflow-y: auto;
                        margin: 10px 0;
                        border-radius: 1px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    }}
                    .custom-table {{
                        width: 100%;
                        border-collapse: separate;
                        border-spacing: 0;
                        font-family: Arial, sans-serif;
                    }}
                    .custom-table thead th {{
                        position: sticky;
                        top: 0;
                        background-color: #09b7ba !important;
                        color: white !important;
                        font-weight: 600;
                        padding: 1px 1px;
                        text-align: center;
                        font-size:15px;
                        border: 2px solid white !important;
                       
                    }}
                    .custom-table tbody td {{
                        padding: 1px 1px;
                        font-size:13px;
                        text-align: center;
                        border: 2px solid #09b7ba !important;
                        min-width: 150px !important; 
                         
                    }}
                    .custom-table tbody tr:nth-child(even) {{
                        background-color: #f8f9fa;
                    }}
                    .custom-table tbody tr:hover {{
                        background-color: rgba(9, 183, 186, 0.05);
                    }}
                    .custom-table tbody td:last-child {{
                        border-right: none;
                    }}
                    .custom-table tbody tr:last-child td {{
                        border-bottom: none;
                    }}
                  

                </style>
                <div class="custom-table-container">
                    <table class="custom-table">
                        <thead>
                            <tr>
                                {''.join(f'<th>{col}</th>' for col in filtered_df.columns)}
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(
                                f'<tr>{"".join(f"<td>{row[col]}</td>" for col in filtered_df.columns)}</tr>'
                                for _, row in filtered_df.iterrows()  # Removed .head(100) to show all records
                            )}
                        </tbody>
                    </table>
                </div>
                """
                st.markdown(table_html, unsafe_allow_html=True)
                
                # Show dataset info
                with st.expander("🔍 Dataset Information"):
                    st.write(f"**Shape:** {raw_df.shape}")
                    st.write("**Columns:**")
                    for col in raw_df.columns:
                        st.write(f"- {col} ({raw_df[col].dtype})")
                    
                    st.write("**Summary Statistics:**")
                    # Convert summary stats to HTML with same styling
                    summary_df = raw_df.describe(include='all').fillna('-')
                    summary_html = f"""
                    <div class="custom-table-container">
                        <table class="custom-table">
                            <thead>
                                <tr>
                                    {''.join(f'<th>{col}</th>' for col in summary_df.columns)}
                                </tr>
                            </thead>
                            <tbody>
                                {''.join(
                                    f'<tr>{"".join(f"<td>{row[col]}</td>" for col in summary_df.columns)}</tr>'
                                    for _, row in summary_df.iterrows()
                                )}
                            </tbody>
                        </table>
                    </div>
                    """
                    st.markdown(summary_html, unsafe_allow_html=True)
            else:
                st.warning("Raw dataset file not found.")
        
        with tab2:
            if os.path.exists(PROCESSED_DATA_PATH):
                processed_df = pd.read_csv(PROCESSED_DATA_PATH)
                
                # Interactive visualization
                st.markdown("### Processed Data Visualization")
                viz_col = st.selectbox("Select column to visualize", processed_df.columns)
                
                if pd.api.types.is_numeric_dtype(processed_df[viz_col]):
                    fig = px.histogram(processed_df, x=viz_col, color_discrete_sequence=['#4fc3f7'])
                else:
                    fig = px.bar(processed_df[viz_col].value_counts(), 
                                color_discrete_sequence=['#4fc3f7'])
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show record count
                st.markdown(f"**Showing {len(processed_df)} records**")
                
                # Convert processed dataframe to HTML with custom styling
                table_html = f"""
                <style>
                    .custom-table-container {{
                        max-height: 500px;
                        overflow-y: auto;
                        margin: 10px 0;
                        border-radius: 1px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    }}
                    .custom-table {{
                        width: 100%;
                        border-collapse: separate;
                        border-spacing: 0;
                        font-family: Arial, sans-serif;
                    }}
                    .custom-table thead th {{
                        position: sticky;
                        top: 0;
                        background-color: #09b7ba !important;
                        color: white !important;
                        font-weight: 600;
                        padding: 3px 1px;
                        text-align: center;
                        font-size:15px;
                        border: 2px solid white !important;
                    }}
                    .custom-table tbody td {{
                        padding: 1px 1px;
                        font-size:13px;
                        text-align: center;
                        border: 2px solid #09b7ba !important;
                        min-width: 150px !important;  
                    }}
                    .custom-table tbody tr:nth-child(even) {{
                        background-color: #f8f9fa;
                    }}
                    .custom-table tbody tr:hover {{
                        background-color: rgba(9, 183, 186, 0.05);
                    }}
                    .custom-table tbody td:last-child {{
                        border-right: none;
                    }}
                    .custom-table tbody tr:last-child td {{
                        border-bottom: none;
                    }}
                </style>
                <div class="custom-table-container">
                    <table class="custom-table">
                        <thead>
                            <tr>
                                {''.join(f'<th>{col}</th>' for col in processed_df.columns)}
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(
                                f'<tr>{"".join(f"<td>{row[col]}</td>" for col in processed_df.columns)}</tr>'
                                for _, row in processed_df.iterrows()  # Removed .head(100) to show all records
                            )}
                        </tbody>
                    </table>
                </div>
                """
                st.markdown(table_html, unsafe_allow_html=True)
                
                # Show processing info
                with st.expander("🔧 Processing Details"):
                    st.write("**Processing Steps:**")
                    st.write("1. Missing value imputation")
                    st.write("2. Categorical variable encoding")
                    st.write("3. Feature scaling")
                    st.write("4. Target variable transformation")
                    
                    st.write("**Feature Information:**")
                    for col in processed_df.columns:
                        st.write(f"- {col} ({processed_df[col].dtype})")       
            else:
                st.warning("Processed dataset file not found.")
# ---------------------- Profile Tab ------------------------
elif choice == "Profile":
    st.subheader("👤 My Profile")
    st.markdown("Manage your account, settings, and support options")
    
    # Main profile tabs
    tab1, tab2, tab3 = st.tabs(["⚙️ Settings", "📞 Contact Us", "🔒 Logout"])
    
    with tab1:
        st.markdown("<div class='PremiumTab'>", unsafe_allow_html=True)
        st.markdown("### System Settings")
        
        # Settings subtabs
        subtab1, subtab2, subtab3 = st.tabs(["General", "Profile", "About"])
        
        with subtab1:
            st.markdown("#### General Settings")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Language Preference**")
                language_options = ["English", "Spanish", "French", "German"]
                st.session_state.language = st.selectbox(
                    "Select Language",
                    language_options,
                    index=language_options.index(st.session_state.language),
                    key="language_select"
                )
                if st.button("Save Language", key="save_language", type="primary"):
                    st.markdown("<div class='success-message'>Language preference updated successfully!</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Theme Preference**")
                theme_options = ["Light", "Dark", "System"]
                st.session_state.theme = st.selectbox(
                    "Select Theme",
                    theme_options,
                    index=theme_options.index(st.session_state.theme),
                    key="theme_select"
                )
                if st.button("Apply Theme", key="apply_theme", type="primary"):
                    st.markdown("<div class='success-message'>Theme preference updated successfully!</div>", unsafe_allow_html=True)
        
        with subtab2:
            st.markdown("#### Judge Profile")
            
            if 'edit_mode' not in st.session_state:
                st.session_state.edit_mode = False
            
            if not st.session_state.edit_mode:
                # View mode
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Profile placeholder with icon
                    st.markdown("""
                    <div style="text-align: center; margin-bottom: 20px;">
                        <div style="font-size: 48px; margin-bottom: 10px;">👨‍⚖️</div>
                        <div style="font-weight: bold; color: var(--primary);">Judge Profile</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='profile-info'>", unsafe_allow_html=True)
                    st.markdown(f"**Name:** {st.session_state.profile_data['name']}")
                    st.markdown(f"**ID:** {st.session_state.profile_data['id']}")
                    st.markdown(f"**Court:** {st.session_state.profile_data['court']}")
                    st.markdown(f"**Email:** {st.session_state.profile_data['email']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("#### Qualifications")
                for qual in st.session_state.profile_data['qualifications']:
                    st.markdown(f"- {qual}")
                
                st.markdown("#### Achievements")
                for achievement in st.session_state.profile_data['achievements']:
                    st.markdown(
                        f"<div class='achievement-card'>"
                        f"<span class='achievement-icon'>{achievement['icon']}</span>"
                        f"{achievement['title']} - {achievement['year']}"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                
                if st.button("Edit Profile Details", key="edit_profile_details", type="primary"):
                    st.session_state.edit_mode = True
                    st.rerun()
            
            else:
                # Edit mode
                with st.form("profile_edit_form"):
                    st.markdown("<div class='edit-form'>", unsafe_allow_html=True)
                    st.markdown("### Edit Profile Information")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Profile icon selector
                        st.markdown("**Select Profile Icon**")
                        icon_options = ["👨‍⚖️", "👩‍⚖️", "🧑‍⚖️", "⚖️", "🏛️"]
                        st.session_state.profile_icon = st.selectbox(
                            "Choose an icon",
                            icon_options,
                            index=icon_options.index(st.session_state.get('profile_icon', '👨‍⚖️')),
                            key="profile_icon_select"
                        )
                    
                    with col2:
                        st.session_state.profile_data['name'] = st.text_input("Full Name", 
                                                                              value=st.session_state.profile_data['name'])
                        st.session_state.profile_data['id'] = st.text_input("Judge ID", 
                                                                          value=st.session_state.profile_data['id'])
                        st.session_state.profile_data['court'] = st.text_input("Court", 
                                                                             value=st.session_state.profile_data['court'])
                        st.session_state.profile_data['email'] = st.text_input("Email", 
                                                                             value=st.session_state.profile_data['email'])
                    
                    st.markdown("#### Edit Qualifications")
                    new_qualifications = []
                    
                    for i, qual in enumerate(st.session_state.profile_data['qualifications']):
                        new_qual = st.text_input(f"Qualification {i+1}", value=qual, key=f"qual_{i}")
                        new_qualifications.append(new_qual)
                    
                    add_qual_col1, add_qual_col2 = st.columns([1, 3])
                    with add_qual_col1:
                        if st.form_submit_button("➕ Add Qualification"):
                            st.session_state.profile_data['qualifications'].append("")
                            st.rerun()
                    
                    st.session_state.profile_data['qualifications'] = [q for q in new_qualifications if q]
                    
                    st.markdown("#### Edit Achievements")
                    new_achievements = []
                    
                    for i, achievement in enumerate(st.session_state.profile_data['achievements']):
                        ach_col1, ach_col2, ach_col3 = st.columns([1, 3, 2])
                        with ach_col1:
                            icon = st.text_input("Icon", value=achievement['icon'], key=f"ach_icon_{i}")
                        with ach_col2:
                            title = st.text_input("Title", value=achievement['title'], key=f"ach_title_{i}")
                        with ach_col3:
                            year = st.text_input("Year/Details", value=achievement['year'], key=f"ach_year_{i}")
                        new_achievements.append({"icon": icon, "title": title, "year": year})
                    
                    st.session_state.profile_data['achievements'] = new_achievements
                    
                    add_ach_col1, add_ach_col2 = st.columns([1, 3])
                    with add_ach_col1:
                        if st.form_submit_button("➕ Add Achievement"):
                            st.session_state.profile_data['achievements'].append({"icon": "🏅", "title": "", "year": ""})
                            st.rerun()
                    
                    submit_col1, submit_col2, submit_col3 = st.columns([1, 1, 2])
                    with submit_col1:
                        if st.form_submit_button("💾 Save Changes", type="primary"):
                            st.session_state.edit_mode = False
                            st.markdown("<div class='success-message'>Profile updated successfully!</div>", unsafe_allow_html=True)
                            st.rerun()
                    with submit_col2:
                        if st.form_submit_button("❌ Cancel"):
                            st.session_state.edit_mode = False
                            st.rerun()
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        
        with subtab3:
            st.markdown("#### About the System")
            with st.expander("Terms of Use", expanded=False):
                st.markdown("""
                **Terms of Use**
                
                1. **Usage Restrictions**: This system is for authorized judicial personnel only.
                2. **Data Privacy**: All case data must be handled in accordance with federal privacy laws.
                3. **System Access**: Unauthorized access attempts may result in account suspension.
                4. **Liability**: The system provides advisory predictions and is not liable for judicial outcomes.
                
                Last Updated: May 13, 2025
                """)
            
            with st.expander("Privacy Policy", expanded=False):
                st.markdown("""
                **Privacy Policy**
                
                1. **Data Collection**: We collect only necessary user and case data for system functionality.
                2. **Data Storage**: All data is stored on secure, encrypted servers.
                3. **Data Sharing**: No personal data is shared with third parties without consent.
                4. **User Rights**: Users may request data deletion subject to legal retention requirements.
                
                Last Updated: May 13, 2025
                """)
            
            with st.expander("System Information", expanded=False):
                st.markdown("""
                **Version**: 2.4.1  
                **Last Updated**: May 15, 2025  
                **Developed By**: Judicial AI Solutions  
                **License**: Proprietary  
                **Support**: support@judicialai.gov
                """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='PremiumTab'>", unsafe_allow_html=True)
        st.markdown("### Contact Us")
        
        with st.expander("Frequently Asked Questions (FAQs)", expanded=True):
            faqs = [
                {
                    "question": "How accurate are the predictions?",
                    "answer": "Our machine learning models achieve 85-92% accuracy based on historical case data and continuously improve with more usage."
                },
                {
                    "question": "Can I edit past predictions?",
                    "answer": "Past predictions are immutable to maintain data integrity, but you can export them or add case notes for future reference."
                },
                {
                    "question": "How do I report a bug or issue?",
                    "answer": "Use the contact form below or email us directly at support@judicialai.gov. Please include screenshots if possible."
                },
                {
                    "question": "Is my data secure and private?",
                    "answer": "Yes, we use AES-256 encryption, regular security audits, and comply with all judicial data protection standards."
                },
                {
                    "question": "How often is the system updated?",
                    "answer": "We release minor updates weekly and major feature updates quarterly. All updates are thoroughly tested before deployment."
                }
            ]
            for faq in faqs:
                st.markdown(f"""
                <div class='faq-item'>
                    <strong style='color: #1e40af;'>{faq['question']}</strong><br>
                    <span style='color: #4b5563;'>{faq['answer']}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("#### Contact Support")
        contact_col1, contact_col2 = st.columns([2, 1])
        
        with contact_col1:
            with st.form("support_form"):
                email_subject = st.text_input("Subject*", placeholder="Briefly describe your issue", key="email_subject")
                email_body = st.text_area("Message*", placeholder="Please provide detailed information about your question or issue", height=180, key="email_body")
                
                submitted = st.form_submit_button("✉️ Send Message", type="primary")
                if submitted:
                    if email_subject and email_body:
                        st.markdown("""
                        <div class='success-message'>
                            <div style='display: flex; align-items: center;'>
                                <span style='font-size: 20px; margin-right: 10px;'>✓</span>
                                <div>
                                    <strong>Message Sent Successfully!</strong><br>
                                    Our support team will respond within 24 hours.
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Please complete all required fields (marked with *)")
        
        with contact_col2:
            # st.markdown("<div class='profile-info'>", unsafe_allow_html=True)
            st.markdown("**🕒 Support Hours**")
            st.markdown("Monday - Friday: 8:00 AM - 6:00 PM")
            st.markdown("Saturday: 9:00 AM - 2:00 PM")
            st.markdown("**📧 Email Support**")
            st.markdown("help@judicialai.gov")
            st.markdown("**📍 Headquarters**")
            st.markdown("123 Justice Ave, Suite 500<br>Washington, DC 20001", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        # st.markdown("<div class='PremiumTab'>", unsafe_allow_html=True)
        # st.markdown("<div class='logout-card'>", unsafe_allow_html=True)
        st.markdown("### 🔒 Logout")
        st.markdown("You are about to end your current session. Please confirm:")
        
        if st.session_state.logged_in:
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button(" Yes, Logout", key="logout_btn", type="primary"):
                    st.session_state.logged_in = False
                    st.markdown("""
                    <div class='success-message'>
                        <div style='display: flex; align-items: center;'>
                            <span style='font-size: 20px; margin-right: 10px;'>👋</span>
                            <div>
                                <strong>Logged Out Successfully</strong><br>
                                Redirecting to login page...
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            # with col2:
            #     if st.button(" Cancel", key="cancel_logout"):
            #         st.rerun()
        else:
            st.warning("You are currently logged out")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", placeholder="Enter your password", type="password")
                submitted = st.form_submit_button("🔑 Login", type="primary")
                if submitted:
                    if username and password:  # Add actual authentication logic here
                        st.session_state.logged_in = True
                        st.markdown("""
                        <div class='success-message'>
                            <div style='display: flex; align-items: center;'>
                                <span style='font-size: 20px; margin-right: 10px;'>✓</span>
                                <div>
                                    <strong>Login Successful!</strong><br>
                                    Welcome back to the Judicial Decision Support System.
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.error("Please provide both username and password")
        
        st.markdown("<div style='margin-top: 30px; text-align: center; color: #64748b;'>Judicial Decision Support System v2.4.1</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)




        # Judicial Notes Tab



# ------------------------ Judicial Notes Tabs ------------------------
elif choice == "Judicial Notes":
    st.subheader("📝 Judicial Notes Notebook")
    st.markdown("""
    <div style='
        font-size: 16px;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4f2f8 100%);
        padding: 16px 24px;
        border-left: 4px solid #09b7ba;
        margin-bottom: 24px;
        # border-radius: 0 8px 8px 0;
        color: #2d3748;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        font-family: "Segoe UI", Roboto, sans-serif;
    '>
        <span style="font-weight:600;">DIGITAL NOTEBOOK</span><br>
        Create and manage personal case notes in your private, secure workspace. All notes are encrypted at rest.
                <br></br>
    </div>
""", unsafe_allow_html=True)

   # Load existing notes or initialize
    notes_df = pd.read_csv(NOTES_FILE) if os.path.exists(NOTES_FILE) else pd.DataFrame(columns=["Timestamp", "CaseID", "Note"])

   # ------------------- New Note Entry -------------------
    with st.container():
     with st.form("new_note_form"):
        st.markdown("""
            <div style='
                background: linear-gradient(135deg, #09b7ba 0%, #007b83 100%);
                color: white;
                padding: 16px 24px;
                font-size: 18px;
                font-weight: 600;
                border-radius: 12px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                display: flex;
                align-items: center;
                gap: 12px;
            '>
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 20h9"></path>
                    <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
                </svg>
                NEW NOTE ENTRY
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 3])
        with col1:
            case_id = st.text_input("**Case ID** (Optional)", placeholder="e.g. CR-2023-0456", help="Reference number for your case")
        with col2:
            note_text = st.text_area("**Note Content**", placeholder="Type your detailed notes here...\n• Use bullet points\n• Add key observations\n• Include important dates", height=150)

        submitted = st.form_submit_button("💾 Save Note", use_container_width=True)
        if submitted:
            if note_text:
                new_note = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "CaseID": case_id if case_id else "N/A",
                    "Note": note_text
                }
                notes_df = pd.concat([notes_df, pd.DataFrame([new_note])], ignore_index=True)
                notes_df.to_csv(NOTES_FILE, index=False)
                st.toast("✅ Note saved successfully!", icon="✔️")
                st.rerun()
            else:
                st.warning("Please enter note content before saving")

     # Divider
    st.markdown("""
    <div style='
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(9,183,186,0.5), transparent);
        margin: 24px 0;
    '></div>
""", unsafe_allow_html=True)

    # ------------------- Display Notes -------------------
    st.markdown("""
    <div style='
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 24px 0 16px 0;
    '>
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#09b7ba" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
        <h4 style='margin:0;color:#2d3748;'>Your Notes</h4>
    </div>
""", unsafe_allow_html=True)

    search_term = st.text_input("", placeholder="🔍 Search notes by Case ID or content...", 
                          help="Filter through your notes", 
                          label_visibility="collapsed")

    filtered_notes = notes_df[
    notes_df["CaseID"].str.contains(search_term, case=False, na=False) |
    notes_df["Note"].str.contains(search_term, case=False, na=False)
    ] if search_term else notes_df

    if not filtered_notes.empty:
      for _, row in filtered_notes.sort_values("Timestamp", ascending=False).iterrows():
        with st.expander(f"🗓️ {row['Timestamp']} | Case: {row['CaseID']}", expanded=False):
            st.markdown(f"""
                <div style='
                    background: #f8fafc;
                    padding: 16px;
                    border-radius: 8px;
                    border-left: 3px solid #09b7ba;
                    margin: 8px 0;
                    white-space: pre-wrap;
                    color: #4a5568;
                    font-size: 15px;
                    line-height: 1.6;
                '>
                    {row['Note']}
                </div>
            """, unsafe_allow_html=True)
            
            # Add action buttons
            col_view, col_del = st.columns([1, 1])
            with col_view:
                if st.button("📋 Copy", key=f"copy_{_}", use_container_width=True):
                    pyperclip.copy(row['Note'])
                    st.toast("Copied to clipboard!", icon="📋")
            with col_del:
                if st.button("🗑️ Delete", key=f"del_{_}", use_container_width=True, type="secondary"):
                    notes_df = notes_df.drop(index=_)
                    notes_df.to_csv(NOTES_FILE, index=False)
                    st.toast("Note deleted!", icon="🗑️")
                    st.rerun()
    else:
        st.markdown("""
        <div style='
            text-align: center;
            padding: 40px 20px;
            background: #f8fafc;
            border-radius: 12px;
            margin: 20px 0;
        '>
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#09b7ba" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            <h4 style='margin:16px 0 8px 0;color:#2d3748;'>No notes found</h4>
            <p style='color:#718096;margin:0;'>Start by adding your first note above</p>
        </div>
    """, unsafe_allow_html=True)

     # ------------------- Export Buttons -------------------
    st.markdown("""
    <div style='
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 32px 0 16px 0;
    '>
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#09b7ba" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        <h4 style='margin:0;color:#2d3748;'>Export Notes</h4>
    </div>
""", unsafe_allow_html=True)

    # Prepare text data for download
    txt_data = "\n\n".join(
    f"Timestamp: {row['Timestamp']}\nCase ID: {row['CaseID']}\nNote:\n{row['Note']}"
    for _, row in filtered_notes.iterrows()
)

   # Single download button for TXT format
    st.download_button(
    "📑 Download TXT File", 
    data=txt_data, 
    file_name="judicial_notes.txt", 
    mime="text/plain",
    use_container_width=True,
    help="Export all notes as a plain text document"
)

    # Add some empty space at the bottom
    st.markdown("<div style='margin-bottom:40px;'></div>", unsafe_allow_html=True)