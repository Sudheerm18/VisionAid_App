import streamlit as st
from PIL import Image, ImageDraw
import pyttsx3
import os
import pytesseract
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
import torch
import cv2

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize Google Generative AI with API Key
GEMINI_API_KEY = 'AIzaSyDOyDLUizbDCM6S5QvmlI1ZxhybCaGdess'  # Replace with your valid API key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Streamlit Styling
st.markdown(
    """
    <style>
     .main-title {
        font-size: 60px;  /* Bigger font */
        font-weight: bold;
        text-align: center;
        color: red;  /* Red color for app name */
        margin-top: -20px;
     }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .feature-header {
        font-size: 24px;
        color: #333;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main Title
st.markdown('<div class="main-title">VisionAId</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Empowering lives with AI: Image descriptions, text extraction, object detection, and voice guidance!</div>', unsafe_allow_html=True)

# Sidebar with Overview
st.sidebar.title("ℹ️ About VisionAId")
st.sidebar.markdown(
    """
    📌 **Features**
    - 🔍 **Describe Scene**: AI insights about the image, including objects and suggestions.
    - 📝 **Extract Text**: Extract visible text using OCR.
    - 🔊 **Text-to-Speech**: Hear the extracted text aloud.
    - 🚧 **Object Detection**: Highlight objects detected in the image.

    💡 **How it helps**:
    Assists visually impaired users by providing scene descriptions, text extraction, and speech.

    🤖 **Powered by**:
    - **Google Gemini API** for scene understanding.
    - **Tesseract OCR** for text extraction.
    - **pyttsx3** for text-to-speech.
    - **YOLOv5** for object detection.
    """
)

# Sidebar Instructions
st.sidebar.text_area("📜 Instructions", "1. Upload an image. 2. Choose a feature: Describe Scene, Extract Text, Listen to Text, or Detect Objects.")

# Functions for functionality
def extract_text_from_image(image):
    """Extracts text from the given image using OCR."""
    return pytesseract.image_to_string(image)

def text_to_speech(text):
    """Converts the given text to speech."""
    engine.say(text)
    engine.runAndWait()

def generate_scene_description(input_prompt, image_data):
    """Generates a scene description using Google Generative AI."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

def input_image_setup(uploaded_file):
    """Prepares the uploaded image for processing."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data,
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")

def detect_objects(image):
    """Detect objects in an image using YOLOv5."""
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    img = Image.open(image).convert("RGB")
    img.save("temp.jpg")  # Save for OpenCV processing
    img_cv = cv2.imread("temp.jpg")
    results = model(img_cv)

    # Annotate and save the image
    annotated_path = "static/annotated_images/annotated_image.jpg"
    os.makedirs("static/annotated_images", exist_ok=True)
    results.save(save_dir="static/annotated_images")

    # Summarize detected objects
    objects_detected = results.pandas().xyxy[0]['name'].value_counts().to_dict()
    summary = f"Detected objects include: {', '.join(objects_detected.keys())}."

    return annotated_path, summary

# Upload Image Section
st.markdown("<h3 class='feature-header'>📤 Upload an Image</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop or browse an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

# Buttons Section
st.markdown("<h3 class='feature-header'>⚙️ Features</h3>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

scene_button = col1.button("🔍 Describe Scene")
ocr_button = col2.button("📝 Extract Text")
tts_button = col3.button("🔊 Text-to-Speech")
object_button = col4.button("🚧 Detect Objects")

# Input Prompt for Scene Understanding
input_prompt = """
You are an AI assistant helping visually impaired individuals by describing the scene in the image. Provide:
1. List of items detected in the image with their purpose.
2. Overall description of the image.
3. Suggestions for actions or precautions for the visually impaired.
"""

# Process user interactions
if uploaded_file:
    image_data = input_image_setup(uploaded_file)

    if scene_button:
        with st.spinner("Generating scene description..."):
            response = generate_scene_description(input_prompt, image_data)
            st.markdown("<h3 class='feature-header'>🔍 Scene Description</h3>", unsafe_allow_html=True)
            st.write(response)

    if ocr_button:
        with st.spinner("Extracting text from the image..."):
            text = extract_text_from_image(image)
            st.markdown("<h3 class='feature-header'>📝 Extracted Text</h3>", unsafe_allow_html=True)
            st.text_area("Extracted Text", text, height=150)

    if tts_button:
        with st.spinner("Converting text to speech..."):
            text = extract_text_from_image(image)
            if text.strip():
                text_to_speech(text)
                st.success("✅ Text-to-Speech Conversion Completed!")
            else:
                st.warning("No text found to convert.")

    if object_button:
        with st.spinner("Detecting objects in the image..."):
            annotated_path, summary = detect_objects(uploaded_file)
            st.markdown("<h3 class='feature-header'>🚧 Detected Objects</h3>", unsafe_allow_html=True)
            if annotated_path and os.path.exists(annotated_path):
                st.image(annotated_path, caption="Detected Objects", use_container_width=True)
                st.write(summary)
            else:
                st.warning("No objects detected.")

# Footer
st.markdown(
    """
    <hr>
    <footer style="text-align:center;">
        <p>Powered by <strong>Google Gemini API</strong> | ©️ 2024 VisionAId | Built with ❤️ using Streamlit</p>
    </footer>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <hr>
    <footer style="text-align:center;">
        <p>Powered by <strong>Google Gemini API</strong> | ©️ 2024 VisionAId</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
