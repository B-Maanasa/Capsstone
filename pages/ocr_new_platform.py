import streamlit as st
import numpy as np
from textblob import TextBlob
from googletrans import Translator, LANGUAGES
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, ClientSettings
import easyocr

# Initialize EasyOCR reader and Google Translator
reader = easyocr.Reader(['en'])  # Set the languages for OCR
translator = Translator()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def transform(self, frame):
        # Convert the frame to BGR format
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Use EasyOCR to extract text from the frame
        result = self.reader.readtext(frame_bgr)

        # Process the detected text
        processed_frame = frame.copy()

        for detection in result:
            text = detection[1]
            bbox = detection[0]

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
            cv2.putText(
                processed_frame,
                translated_text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),  # White color
                2
            )

        # Convert the processed frame back to RGB format
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        return processed_frame_rgb

def translate_text(text, target_lang):
    translation = translator.translate(text, dest=target_lang)
    return translation.text

def main():
    st.title("Live OCR Translation with WebRTC")

    # Get the supported languages for translation
    supported_languages = list(LANGUAGES.values())
    
    # Create a Streamlit dropdown to select the target language
    selected_language = st.selectbox("Select Target Language", supported_languages, index=0)

    # Initialize WebRTC client settings
    rtc_configuration = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    # Start WebRTC streaming
    webrtc_ctx = webrtc_streamer(
        key="example",
        client_settings=ClientSettings(rtc_configuration=rtc_configuration),
        video_transformer_factory=VideoTransformer,
    )

    if webrtc_ctx.video_transformer:
        st.write("WebRTC connection established. Start the camera to perform live OCR translation.")
        st.button("Start Camera")

        # Process frames when the camera is started
        if st.button("Start Camera"):
            st.write("Camera is on. Performing live OCR translation...")
            webrtc_ctx.video_transformer.run()

    else:
        st.warning("WebRTC connection failed. Please check your network settings.")

if __name__ == "__main__":
    main()
