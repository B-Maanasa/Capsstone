import cv2
import easyocr
import numpy as np
import streamlit as st
from textblob import TextBlob
from googletrans import Translator, LANGUAGES

# Initialize EasyOCR reader and Google Translator
reader = easyocr.Reader(['en'])  # Set the languages for OCR
translator = Translator()

def translate_text(text, target_lang):
    translation = translator.translate(text, dest=target_lang)
    return translation.text

def add_transparent_text(image, text, position, detected_lang, font_scale=0.8, thickness=2):
    # Create a copy of the image
    overlay = image.copy()
    
    # Get text size and position
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_width, text_height = text_size
    
    # Define background rectangle position
    background_position = (
        position[0],
        position[1] - text_height - 10
    )
    
    # Draw semi-transparent grey rectangle as background
    cv2.rectangle(
        overlay,
        background_position,
        (background_position[0] + text_width, background_position[1] + text_height),
        (128, 128, 128),
        -1
    )
    
    # Add text to the overlay image (100% white color)
    cv2.putText(
        overlay,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),  # White color
        thickness
    )
    
    # Display detected language on top left corner
    cv2.putText(
        overlay,
        f"Translated Language: {detected_lang.upper()}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),  # White color
        thickness
    )
    
    return overlay

def capture_and_process(target_lang='en'):
    st.title("Live OCR Translation")

    cap = cv2.VideoCapture(0)  # Open the default camera (0)

    while True:
        ret, frame = cap.read()  # Capture frame from the camera
        if not ret:
            break
        
        # Use EasyOCR to extract text from the frame
        result = reader.readtext(frame)
        
        # Process the detected text
        processed_frame = frame.copy()  # Initialize processed frame

        for detection in result:
            text = detection[1]
            bbox = detection[0]  # Bounding box of the detected text
            
            try:
                # Detect the language of the extracted text using TextBlob
                detected_lang = TextBlob(text).detect_language()
            except:
                detected_lang = 'en'  # Default to 'en' if language detection fails
            
            # Translate the text to the target language if it's different from the source language
            if detected_lang != target_lang:
                translated_text = translate_text(text, target_lang)
            else:
                translated_text = text
            
            # Display the translated text on the input image
            text_position = (int(bbox[0][0]), int(bbox[0][1]))  # Position of the detected text
            processed_frame = add_transparent_text(
                processed_frame,
                translated_text,
                text_position,
                detected_lang
            )

        # Display the processed frame using Streamlit
        st.image(processed_frame, channels="BGR")

        # Check for key press (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

def start_translation(target_lang='en'):
    capture_and_process(target_lang)

if __name__ == "__main__":
    # Get the supported languages for translation
    supported_languages = list(LANGUAGES.values())
    
    # Create a Streamlit dropdown to select the target language
    selected_language = st.sidebar.selectbox("Select Target Language", supported_languages, index=0)
    # Start the translation process
    start_translation(selected_language)
