import streamlit as st
from googletrans import Translator
from gtts import gTTS
#import base64te
from io import BytesIO
import tempfile
from languages import languages

#streamlit pages code -->

st.set_page_config(
page_title=  "Hey Lexi",
page_icon="👋🏻"
)

st.sidebar.success("Select a page above.")
# Text to Text code start from here -> 

st.title("Language Translation App")

source_text = st.text_area("Enter text to translate:")
target_language = st.selectbox("Select target language:", languages)

translate = st.button('Translate')

if translate:
    translator = Translator()
    # Translate the source text to the selected target language
    out = translator.translate(source_text, dest=target_language)

    # Display the translated text
    st.write(out.text)

    # Generate the audio from the translated text
    tts = gTTS(out.text, lang=target_language)

    # Save the audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        tts.save(fp.name)
        fp.seek(0)
        # Convert the audio to a base64 format for embedding in Streamlit
        audio_bytes = fp.read()

    # Display a speaker icon with a button to play the audio
    if st.button("🔊 Click to listen to the translation"):
        st.audio(audio_bytes, format='audio/mp3')
