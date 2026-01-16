Generative AI-based Multilingual Voice Assistant

A web-based AI voice assistant that answers user queries by retrieving information from PDF-based FAQs and responding in real-time using voice. The system supports multiple Indian languages and provides an intuitive customer-support experience.

 Features

-  PDF-based FAQ understanding
-  Retrieval Augmented Generation (RAG)
-  Speech-to-Text using Whisper
-  Text-to-Speech voice responses
-  Multilingual support:
  - English
  - Hindi
  - Telugu
  - Tamil
  - Kannada
  - Malayalam
  - Bengali
-  Real-time responses

 Tech Stack

 Component : Technology 
  Backend   :Python, Flask 
 Frontend  : Streamlit 
 LLM       : OpenAI API 
 Speech-to-Text : Whisper 
 Text-to-Speech : gTTS 
 Retrieval : RAG 
 PDF Parsing : PyMuPDF 

System Architecture

1. User speaks a query
2. Whisper converts audio → text
3. Query is matched with PDF content using RAG
4. OpenAI generates contextual response
5. Response converted to voice using TTS
6. Audio response played to the user



 Project Structure:
 
 BankAssistantPOC/
│
├── BankPOC.pdf # Banking FAQ document
├── final_app_test.py # Main application file
├── htmlTemplates.py # HTML templates for UI
├── requirements2.txt # Python dependencies
└── README.md


 Setup & Installation

1. Clone the repository

git clone https://github.com/asiyx/BankAssistantPOC.git
cd BankAssistantPOC

2. Create a virtual environment
python -m venv venv

Activate it:
Windows: venv\Scripts\activate
Mac/Linux: source venv/bin/activate

3. Install dependencies
pip install -r requirements2.txt

4. Set Environment Variables
Set your API key (example for OpenAI):

Windows:
set OPENAI_API_KEY=your_api_key_here

Mac/Linux: export OPENAI_API_KEY=your_api_key_here

Running the Application
python final_app_test.py

The application will start locally.

Open your browser and go to:

http://localhost:5000

Usage

Upload or load the banking FAQ PDF.
Ask questions using voice or text.
The assistant retrieves relevant content from the PDF.
AI generates accurate responses and replies via text and voice.

Future Enhancements

Authentication for users
Deployment on cloud (AWS / Azure)
Support for more document formats
