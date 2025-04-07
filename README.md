# Aetna PDF Chatbot

A conversational chatbot that interacts with the Aetna Better Health Illinois Member Handbook PDF, allowing users to ask questions and receive relevant information from the document. This project uses Retrieval-Augmented Generation (RAG) to provide accurate, contextual responses based on the content of the handbook.

## Features

- PDF ingestion and parsing into searchable content
- Intelligent text chunking with semantic boundaries
- Vector embeddings for efficient similarity search
- Retrieval-Augmented Generation (RAG) for accurate responses
- Natural language understanding of user queries
- Contextual responses based on PDF content
- Conversation history management
- Secure API key handling
- User-friendly Gradio interface

## Technology Stack

- **Language**: Python 3.9+
- **UI Framework**: Gradio
- **PDF Processing**: PyPDF2
- **LLM Framework**: OpenAI API
- **Vector Database**: ChromaDB
- **LLM**: OpenAI GPT-4
- **Deployment Options**: Vercel, AWS SageMaker

## Project Structure

```
aetna-pdf-chatbot/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── pdf_processor.py    # PDF processing utilities
│   ├── vector_store.py     # Vector database operations
│   ├── llm_service.py      # LLM integration with OpenAI
│   ├── chat_interface.py   # Gradio chat interface
│   └── utils.py            # Helper utilities
├── app.py                  # Main application entry point
├── data/
│   ├── pdf/                # PDF document storage
│   └── vector_db/          # Vector database storage
├── tests/
│   ├── __init__.py
│   ├── test_pdf_processor.py
│   ├── test_vector_store.py
│   └── test_llm_service.py
├── notebooks/
│   └── development.ipynb   # Experimentation notebook
├── .env.example
├── .gitignore
├── requirements.txt
├── DEVELOPMENT.md
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- Git
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd aetna-pdf-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file and add your OpenAI API key.

### Running the Application

Run the optimized RAG-enhanced version:

```bash
python rag_app.py
```

The application will be available at http://localhost:7860

### Using the Chatbot

1. The application will automatically download the Aetna Better Health Illinois Member Handbook PDF if it doesn't exist locally.
2. It will process the PDF, create text chunks, and build a vector database for efficient retrieval.
3. Ask questions about the handbook in the chat interface.
4. The system will retrieve the most relevant sections and generate accurate responses.

## AWS Scaling Strategy

A simple AWS architecture for scaling this application:

```
Users → API Gateway → Lambda Functions → OpenSearch
                           ↓
                         S3 (PDFs)
```

### Simple AWS Components

1. **Lambda Functions**:
   - Serverless compute for handling requests
   - Automatic scaling with no server management
   - Pay only for what you use

2. **OpenSearch Service**:
   - Managed service for vector database
   - Replaces local ChromaDB
   - Scales automatically with demand

3. **S3**:
   - Stores PDF documents
   - Highly durable and available

4. **API Gateway**:
   - Handles all API requests
   - Built-in throttling and monitoring

### Scaling Benefits

- **Zero Infrastructure Management**: No servers to manage
- **Automatic Scaling**: Handles traffic spikes without manual intervention
- **Cost Efficiency**: Pay only for resources used
- *