import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import re
import json
from time import sleep
import os

# Google Gemini (new SDK)
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

# -------------------------------
# Streamlit App Setup
# -------------------------------
st.set_page_config(page_title="Japanese Vocab Extractor", layout="centered")

# Sidebar (API Key)
with st.sidebar:
    st.header("💌\u2003Gemini API Key Setup", divider="red")
    api_key = st.text_input("Gemini API Key (required)", type="password")
    st.caption("This data will not be saved.")

# my logo
st.image("images/image.png", width="content")

load_dotenv()

MODEL_NAME = "gemini-2.0-flash"
model = None

# --------------------------
# Configure Gemini
# --------------------------
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        st.error(f"Gemini configuration failed: {e}")
else:
    st.warning("Please enter the API Key in the sidebar.")

# -------------------------------
# Function: Check Japanese
# -------------------------------
def is_japanese(text):
    japanese_chars = re.findall(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", text)
    ratio = len(japanese_chars) / max(len(text), 1)
    return ratio > 0.2

# -------------------------------
# Streamlit UI
# -------------------------------

multi = ''' *Want to learn Japanese through the songs you love ?*  
just paste your favorite lyrics, and we'll magically pull out all the useful vocabulary for you !

you'll get **furigana**, **translations**, **jlpt levels** and **example sentences** -- all in one place\u2003🔖'''

st.markdown(multi)

num_words = st.slider("Vocabulary count :", 1, 20, 10)

lyrics = st.text_area("Japanese lyrics :", height=200)

if st.button("Process", disabled=not model):

    if not api_key:
        st.error("Please enter a valid API Key before use.")
        st.stop()

    if not lyrics or len(lyrics.strip()) < 10:
        st.error("❌ Please enter lyrics of sufficient length.")
        st.stop()

    if not is_japanese(lyrics):
        st.error("❌ Please enter Japanese lyrics only.")
        st.stop()

    st.success("☁️\u2003Lyrics checked successfully")
    display_lyrics = lyrics.replace("\n", " / ")
    st.info(f"Lyrics used : **{display_lyrics}**")

    # -------------------------------
    # Prompt for Gemini (JSON only, max 500 chars)
    # -------------------------------
    
    prompt = f"""
You are a professional Japanese language teacher.

Extract {num_words} Japanese vocabulary words from the song lyrics below.
Return ONLY valid JSON. No explanations, no markdown, no commentary.

Output format:

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

Lyrics (use only the first 500 characters if very long):
{lyrics[:500]}
"""

    # -------------------------------
    # Call Gemini (with retry)
    # -------------------------------
    max_retries = 3
    response = None

    with st.spinner(f"Processing with Gemini ({MODEL_NAME})..."):
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                break
            except GoogleAPIError as e:
                if attempt < max_retries - 1:
                    st.warning(f"Gemini Error: {e}. Retrying in {2**(attempt+1)} seconds...")
                    sleep(2**(attempt+1))
                else:
                    st.error(f"API Error: {e}")
                    st.stop()
            except Exception as e:
                st.error(f"Unexpected Error: {e}")
                st.stop()

    text = response.text.strip()

    # ลบ markdown หรือคำว่า json ข้างหน้า
    if text.startswith("```"):
        text = text.strip("```").strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()  # ตัด 'json' ออก

    # แสดง raw output ก่อน parse
    # st.text_area("Cleaned Gemini Output", text, height=200)

    # แปลงเป็น dict
    try:
        data = json.loads(text)
        vocab_list = data.get("vocab", [])
    except Exception as e:
        st.error(f"❌ Invalid JSON: {e}")
        st.stop()


    # -------------------------------
    # Parse JSON safely
    # -------------------------------
    if not text:
        st.error("AI did not return any data.")
        st.stop()


    # -------------------------------
    # Display DataFrame
    # -------------------------------
    df = pd.DataFrame(vocab_list)
    df.index = range(1, len(df) + 1)

    st.subheader("📕\u2003Japanese Vocabulary Extracted")

    # JLPT Color Map
    color_map = {
        'N5': '#F3F0FF',
        'N4': '#F0FFFA',
        'N3': '#FDFFF0',
        'N2': '#FFFAF0',
        'N1': '#FFF0F5'
    }

    def color_jlpt(val):
        return f'background-color: {color_map.get(val, "white")}'

    styled_df = df.style.applymap(color_jlpt, subset=['jlpt'])
    st.dataframe(styled_df, use_container_width=True, height=450)

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾\u2003Download CSV",
        data=csv,
        file_name="japanese_vocab.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("Developed by Gemini AI to help you learn Japanese from your favorite songs.")


