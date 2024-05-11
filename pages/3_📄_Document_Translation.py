import streamlit as st
from PyPDF2 import PdfReader
from googletrans import Translator
import tempfile

# Function to translate PDF content to a specified language
def translate_pdf(pdf_reader, target_language):
    translator = Translator()
    translated_pages = []

    # Iterate over each page and translate the text
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()

        if text:
            # Translate the text to the target language
            translated = translator.translate(text, dest=target_language)
            translated_pages.append(translated.text)
        else:
            translated_pages.append("No text found on this page.")

    return translated_pages

st.title("PDF Translator")
st.write("Upload a PDF file, and it will be translated to the selected language.")

# File uploader to upload PDF files
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# Dropdown to select the target language
language_options = {
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-cn",
    # Add more languages as desired
}

selected_language = st.selectbox("Select language for translation:", list(language_options.keys()))

if uploaded_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Create a PDF reader
    pdf_reader = PdfReader(temp_file_path)

    # Translate the PDF content
    translated_content = translate_pdf(pdf_reader, language_options[selected_language])

    # Display the translated content
    st.write(f"Translated content into {selected_language}:")
    for page_num, content in enumerate(translated_content):
        st.markdown(f"### Page {page_num + 1}")
        st.text(content)

    # Remove the temporary file to clean up
    import os
    os.remove(temp_file_path)
