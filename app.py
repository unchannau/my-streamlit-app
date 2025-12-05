import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import re
import json
import os
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from time import sleep

# -------------------------------------------------
# 1) Streamlit Setup
# -------------------------------------------------
st.set_page_config(page_title="JP Vocab Extractor", layout="centered")

with st.sidebar:
    st.header("💌\u2003Gemini API Key Setup", divider="red")
    api_key_input = st.text_input("Gemini API Key (required)", type="password")
    st.caption("This data will not be saved.")

# Optional logo
try:
    st.image("images/image.png", width="content")
except Exception:
    pass

load_dotenv()

# -------------------------------------------------
# 2) Load API Key
# -------------------------------------------------
API_KEY = api_key_input or os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.0-flash"
model = None

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        st.error(f"Gemini configuration failed: {e}")
else:
    st.warning("Please enter the API Key in the sidebar.")

# -------------------------------------------------
# 3) Helper Functions
# -------------------------------------------------
def is_japanese(text: str) -> bool:
    jp_chars = re.findall(r"[ぁ-ゔァ-ヴー々〆〤一-龥]", text)
    return len(jp_chars) >= 15

def build_prompt(lyrics: str, num_words: int) -> str:
    return f"""
You are a professional Japanese language teacher.

Extract {num_words} Japanese vocabulary words from the song lyrics below.
Return ONLY valid JSON. No explanations, no markdown, no commentary.

Requirements:
- furigana must be correct
- English translation must be accurate
- JLPT level must be N5–N1
- Example sentence must be natural and relevant.
- Example sentence must follow this EXACT one-line pattern (no newlines allowed):

Example pattern (strict):
"<Japanese sentence> (<Romaji>) - <English sentence>"

Correct JSON format:
{{
    "vocab": [
        {{
            "word": "...",
            "furigana": "...",
            "translation": "...",
            "jlpt": "...",
            "example": "..."
        }}
    ]
}}

Lyrics (first 500 chars only):
{lyrics[:500]}
"""

def clean_gemini_output(text: str) -> str:
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return text

def call_gemini(prompt: str, model, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except GoogleAPIError as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                st.warning(f"Gemini Error: {e}. Retrying in {wait} seconds...")
                sleep(wait)
            else:
                st.error(f"API Error: {e}")
                st.stop()
        except Exception as e:
            st.error(f"Unexpected Error: {e}")
            st.stop()
    return ""

def parse_json_safely(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        st.error(f"❌\u2003Invalid JSON returned by AI: {e}")
        return {}

# -------------------------------------------------
# 4) UI Intro
# -------------------------------------------------
st.subheader("⏯\u2003Vocabulary Extractor from Japanese song")
st.markdown('''
***Want to learn Japanese through the songs you love ?***  
Just paste your favorite lyrics, and we'll gently gather all the useful words for you !

You'll get **furigana**, **translations**, **JLPT levels**, and **example sentences** — all in one place\u2003🔖
''')

col1, col2 = st.columns([1,2], gap="medium")
with col1:
    st.subheader("💬\u2003Vocab Count")
    num_words = st.number_input("Number of words to extract:", min_value=1, max_value=20, value=10, step=1)
with col2:
    st.subheader("🎵\u2003Japanese Lyrics")
    lyrics = st.text_area("Japanese lyrics:", height=200, placeholder="ここに歌詞を貼り付けてください...")

# Prevent double-click processing
if "busy" not in st.session_state:
    st.session_state["busy"] = False
def is_busy(): return st.session_state.get("busy", False)
def set_busy(x): st.session_state["busy"] = x

# -------------------------------------------------
# 5) Main Process Button
# -------------------------------------------------
if st.button("Process", disabled=not model or is_busy()):
    if not API_KEY:
        st.error("Please enter a valid API Key before use.")
        st.stop()
    if not lyrics or len(lyrics.strip()) < num_words * 4:
        st.error("❌\u2003Please enter lyrics of sufficient length.")
        st.stop()
    if not is_japanese(lyrics):
        st.error("❌\u2003Please enter Japanese lyrics only.")
        st.stop()

    st.success("☁️\u2003Lyrics checked successfully")
    set_busy(True)

    # Build prompt
    prompt = build_prompt(lyrics, num_words)

    # Gemini Call
    with st.spinner(f"Processing with Gemini ({MODEL_NAME})..."):
        raw_output = call_gemini(prompt, model)

    cleaned = clean_gemini_output(raw_output)
    data = parse_json_safely(cleaned)
    vocab_list = data.get("vocab", [])

    # Debug info
    # with st.expander("Check process"):
        # st.text_area("Gemini Raw Output", cleaned, height=200)

    if not vocab_list:
        st.error("No vocabulary returned.")
        set_busy(False)
        st.stop()

    # -------------------------------------------------
    # 6) Vocabulary Tabs
    # -------------------------------------------------
    st.subheader("📕\u2003Japanese Vocabulary Extracted")
    tabs = st.tabs(["DataFrame View", "Card View"])

    # Tab 1: DataFrame
    with tabs[0]:
        df = pd.DataFrame(vocab_list)
        df.index = range(1, len(df) + 1)
        color_map = {'N5': '#F3F0FF','N4': '#F0FFFA','N3': '#FDFFF0','N2': '#FFFAF0','N1': '#FFF0F5'}
        def color_jlpt(val):
            return f"background-color: {color_map.get(val, 'white')}"
        styled_df = df.style.map(color_jlpt, subset=['jlpt'])
        st.dataframe(styled_df, width='stretch', height=450)

    # Tab 2: Card View
    with tabs[1]:
        for item in vocab_list:
            st.markdown(f"""
            <div style="background-color:{color_map.get(item['jlpt'],'#fff')};
                        padding:12px; border-radius:10px; margin-bottom:8px;
                        box-shadow:2px 2px 5px rgba(0,0,0,0.1)">
                <b>Word:</b> {item['word']} &nbsp;&nbsp;
                <b>Furigana:</b> {item['furigana']} &nbsp;&nbsp;
                <b>JLPT:</b> {item['jlpt']} <br>
                <b>Translation:</b> {item['translation']} <br>
                <b>Example:</b> {item['example']}
            </div>
            """, unsafe_allow_html=True)

    # Download CSV
    df_csv = pd.DataFrame(vocab_list)
    csv = df_csv.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾\u2003Download CSV",
        data=csv,
        file_name="japanese_vocab.csv",
        mime="text/csv"
    )

    set_busy(False)

# Footer
st.markdown("---")
st.caption("Developed to help you learn Japanese from your favorite songs.")
