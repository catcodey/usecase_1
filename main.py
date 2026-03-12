import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
import pandas as pd
import re

# --- 1. CONFIGURATION ---
PROJECT_ID = "transcript-summarizer-490013"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(
    "gemini-2.0-flash",
    system_instruction="Keep every response extremely brief. Summaries must be 5 short bullets. All follow-ups under 50 words."
)

# --- 2. SESSION STATE ---
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""
if "input_text_val" not in st.session_state:
    st.session_state.input_text_val = ""
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --- 3. HELPER FUNCTIONS ---
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    fillers = r'\b(um|uh|ah|er|basically|actually|you know|sort of|like)\b'
    return re.sub(fillers, '', text, flags=re.IGNORECASE).strip()

def extract_data(files_list):
    combined_text = ""
    for f in files_list:
        if f.name.endswith('.txt'):
            combined_text += f.read().decode("utf-8") + "\n"
        elif f.name.endswith('.xlsx'):
            df = pd.read_excel(f)
            combined_text += " ".join(df.astype(str).values.flatten()) + "\n"
    return combined_text

# --- 4. UI LAYOUT ---
st.set_page_config(page_title="AI Transcript Analyser", layout="wide")
st.title("Transcript Analysis Dashboard")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Input")
    
    # Check states for mutual exclusion
    has_text = len(st.session_state.input_text_val.strip()) > 0
    
    # 1. File Uploader
    # We use a key that we can increment to force a clear
    uploaded_files = st.file_uploader(
        "Upload TXT/XLSX (Multiple allowed)", 
        type=["txt", "xlsx"], 
        accept_multiple_files=True,
        disabled=has_text,
        key=f"uploader_{st.session_state.uploader_key}"
    )
    
    has_files = uploaded_files is not None and len(uploaded_files) > 0

    if has_files:
        st.info("Files uploaded. Text area is now disabled.")
    
    st.markdown("---")

    # 2. Text Area
    manual_input = st.text_area(
        "Paste Transcript:", 
        height=300, 
        value=st.session_state.input_text_val,
        disabled=has_files,
        placeholder="Paste your text here..."
    )
    # Update session state with manual input
    st.session_state.input_text_val = manual_input
    
    if has_text:
        st.info("Text detected. File upload is now disabled.")

    # Action Buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🚀 Generate", use_container_width=True):
            raw_data = ""
            if has_files:
                raw_data = extract_data(uploaded_files)
            elif has_text:
                raw_data = manual_input
                
            if raw_data:
                cleaned = clean_text(raw_data)
                st.session_state.chat_session = model.start_chat()
                st.session_state.messages = [] 
                
                try:
                    response = st.session_state.chat_session.send_message(f"Summarize in 5 short points:\n\n{cleaned}")
                    st.session_state.summary_text = response.text
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please provide data first.")

    with btn_col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.input_text_val = ""
            st.session_state.summary_text = ""
            st.session_state.messages = []
            st.session_state.uploader_key += 1 # This clears the file uploader widget
            st.rerun()

with col2:
    st.subheader("Summary & Chat")
    chat_container = st.container(height=450)
    with chat_container:
        if not st.session_state.messages:
            st.write("Results will appear here.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Follow-up..."):
        if st.session_state.summary_text:
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                response = st.session_state.chat_session.send_message(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please generate a summary first!")

    if st.session_state.summary_text:
        st.download_button("💾 Download Summary", st.session_state.summary_text, "summary.txt")
