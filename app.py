import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import re
import json
import os
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from time import sleep

# -----------------------------------------------
# 1) Streamlit Setup
# -----------------------------------------------
st.set_page_config(page_title="UtaVocab2", layout="centered")

with st.sidebar:
    st.header(":love_letter: Gemini API Key Setup", divider="red")
    api_key_input = st.text_input("Gemini API Key (required)", type="password")
    st.caption("This data will not be saved.")

# Optional logo
try:
    st.image("images/image.png", width="content")
except:
    pass

load_dotenv()

# -----------------------------------------------
# No cache — model must load from each user's key
# -----------------------------------------------
def load_model_direct(api_key: str):
    """Load Gemini model fresh from user's API key."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


API_KEY = api_key_input or os.getenv("GEMINI_API_KEY")

model = None
if API_KEY:
    try:
        model = load_model_direct(API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {e}")
else:
    st.warning("Please enter the API Key in the sidebar.")

# -----------------------------------------------
# 3) Helper Functions
# -----------------------------------------------
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
- Example sentence must follow this EXACT pattern:

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

Lyrics (first 500 chars):
{lyrics[:500]}
"""


def clean_gemini_output(text: str) -> str:
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return text


def call_gemini(prompt: str, model):
    """Call Gemini ONE TIME. If 429 → stop immediately."""
    try:
        response = model.generate_content(prompt)
        return response.text

    except GoogleAPIError as e:
        if "429" in str(e):
            st.error(":x: You exceeded your quota. Please wait or use another API key.")
            st.stop()
        else:
            st.error(f"API Error: {e}")
            st.stop()

    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        st.stop()


def parse_json_safely(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        st.error(f":x: AI returned invalid JSON: {e}")
        return {}

# -----------------------------------------------
# 4) UI Intro
# -----------------------------------------------
st.subheader(":play_or_pause_button: Vocabulary Extractor for Japanese songs")
st.markdown('''
***Want to learn Japanese through the songs you love ?***  
Just paste your favorite lyrics, and we'll gently gather all the useful words for you !

You'll get **furigana**, **translations**, **JLPT levels**, and **example sentences** — all in one place :bookmark:
''')

col1, col2 = st.columns([1,2], gap="medium")
with col1:
    st.subheader(":speech_balloon: Vocab Count")
    num_words = st.number_input("Number of words to extract:", min_value=1, max_value=20, value=10, step=1)
with col2:
    st.subheader(":musical_note: Japanese Lyrics")
    lyrics = st.text_area("Japanese lyrics:", height=200, placeholder="ここに歌詞を貼り付けてください...")

# -----------------------------------------------
# 5) Process Button
# -----------------------------------------------
if st.button("Process", disabled=not model):
    if not API_KEY:
        st.error("Please enter a valid API Key before use.")
        st.stop()

    if not lyrics or len(lyrics.strip()) < num_words * 4:
        st.error(":x: Please enter lyrics of sufficient length.")
        st.stop()

    if not is_japanese(lyrics):
        st.error(":x: Please enter Japanese lyrics only.")
        st.stop()

    st.success(":cloud: Lyrics validated")
    prompt = build_prompt(lyrics, num_words)

    with st.spinner(f"Processing with Gemini 2.5 Flash..."):
        raw_output = call_gemini(prompt, model)

    cleaned = clean_gemini_output(raw_output)
    data = parse_json_safely(cleaned)
    vocab_list = data.get("vocab", [])

    if not vocab_list:
        st.error("No vocabulary returned.")
        st.stop()

    # -------------------------------------------
    # 6) Vocabulary Tabs
    # -------------------------------------------
    st.subheader(":closed_book: Japanese Vocabulary Extracted")
    tabs = st.tabs(["DataFrame View", "Card View", "Practice View"])

    color_map = {'N5': '#F3F0FF','N4': '#F0FFFA','N3': '#FDFFF0','N2': '#FFFAF0','N1': '#FFF0F5'}

    # Tab 1: DataFrame
    with tabs[0]:
        df = pd.DataFrame(vocab_list)
        df.index = range(1, len(df) + 1)
        def color_jlpt(val):
            return f"background-color: {color_map.get(val, 'white')}"
        styled = df.style.map(color_jlpt, subset=['jlpt'])
        st.dataframe(styled, width='stretch', height=450)

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

    # Tab 3: Practice View
    with tabs[2]:
        for item in vocab_list:
            jlpt_color = color_map.get(item["jlpt"])
            st.markdown(f"""
                <div style="border-left: 4px solid {jlpt_color}; padding-left:8px; margin: 4px 0;">
                    <span style="font-size:17px; font-weight:600;">{item['word']}</span>
                    <span style="font-size:13px; opacity:0.7;"> ({item['jlpt']})</span>
                </div>
            """, unsafe_allow_html=True)
            with st.expander("Reveal"):
                st.markdown(f"""
                **Furigana:** {item['furigana']}  
                **Meaning:** {item['translation']}  
                **Example:** {item['example']}
                """)

    # Export CSV
    df_csv = pd.DataFrame(vocab_list)
    st.download_button(
        label=":floppy_disk: Download CSV",
        data=df_csv.to_csv(index=False).encode("utf-8"),
        file_name="japanese_vocab.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("Developed to help you learn Japanese from your favorite songs.")
