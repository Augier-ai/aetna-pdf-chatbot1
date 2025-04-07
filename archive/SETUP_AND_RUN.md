# Aetna PDF Chatbot Setup and Running Guide

This guide will walk you through setting up and running the Aetna PDF Chatbot project.

## Prerequisites

- Python 3.9+ installed
- Git installed
- Internet connection to download the PDF and dependencies

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd aetna-pdf-chatbot
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root with the following content:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Replace `your_openai_api_key_here` with your actual OpenAI API key.

## Running the Application

1. **Run the application**

   ```bash
   python app.py
   ```

   This will:
   - Download the Aetna PDF if it doesn't exist locally
   - Process the PDF and create embeddings
   - Start the Gradio web interface

2. **Access the web interface**

   Open your browser and go to:
   ```
   http://localhost:7860
   ```

3. **Optional command-line arguments**

   ```bash
   # Force rebuild of vector store
   python app.py --rebuild

   # Run on a different port
   python app.py --port 8000

   # Create a public link (temporary)
   python app.py --share
   ```

## Development Workflow

1. **Run the notebook**

   For development and experimentation, you can use the Jupyter notebook:

   ```bash
   jupyter notebook notebooks/development.ipynb
   ```

   Make sure to set your OpenAI API key in the notebook.

2. **Run tests**

   ```bash
   pytest tests/
   ```

## Troubleshooting

1. **PDF download issues**

   If the PDF fails to download, you can manually download it from:
   ```
   https://www.aetnabetterhealth.com/content/dam/aetna/medicaid/illinois/pdf/ABHIL_Member_Handbook.pdf
   ```

   Place it in the `data/pdf/` directory with the filename `ABHIL_Member_Handbook.pdf`.

2. **OpenAI API issues**

   - Verify your API key is correct
   - Check your OpenAI account has sufficient credits
   - Ensure you have internet connectivity

3. **Vector store issues**

   If you encounter issues with the vector store, try rebuilding it:

   ```bash
   python app.py --rebuild
   ```

## Deployment

For deployment instructions, refer to the `DEVELOPMENT.md` file which includes information about deploying to AWS SageMaker.
