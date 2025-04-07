FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/pdf data/vector_db

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 7860

# Run the application
CMD ["python", "rag_app.py"]
