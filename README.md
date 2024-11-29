# VisionAid_App


# VisionAId: AI-Powered Accessibility Application

VisionAId is an AI-powered application designed to enhance accessibility for visually impaired users. With features such as scene description, text extraction, text-to-speech conversion, and object detection, VisionAId empowers users to interact with their surroundings more effectively.

---

## 🌟 Features
1. **🔍 Describe Scene**: Provides AI-generated insights about the scene in an image.
2. **📝 Extract Text**: Uses OCR to extract visible text from images.
3. **🔊 Text-to-Speech**: Converts extracted text into speech for enhanced accessibility.
4. **🛑 Object Detection**: Identifies and highlights objects in an image for better understanding of the surroundings.

---

## 🚀 How It Works
1. **Upload an Image**: Users upload an image in JPG, JPEG, or PNG format.
2. **Choose a Feature**: 
   - **Scene Description**: AI describes the scene and provides insights.
   - **Text Extraction**: Extracts readable text using Tesseract OCR.
   - **Text-to-Speech**: Converts text to speech using `pyttsx3`.
   - **Object Detection**: Detects objects in the image using YOLOv5.
3. **Receive Results**: Outputs are displayed and audio is generated as required.

---

## 🛠️ Installation and Setup
### Prerequisites
- Python 3.8 or above
- Virtual Environment (recommended)
- pip (Python package manager)

### Steps
1. **Clone the Repository**  
   Download or clone the repository to your local machine.

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv visionaid_env
   source visionaid_env/bin/activate  # On Windows: visionaid_env\Scripts\activate
