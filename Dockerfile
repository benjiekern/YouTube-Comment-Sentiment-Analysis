# Use a lightweight Python base image
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH=/app/src:${PYTHONPATH}

# Set working directory
WORKDIR /app

# Copy only requirements first for better Docker caching
COPY requirements.txt .


# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy the rest of the app
COPY src/ ./src/
COPY mlruns /app/mlruns
COPY config.yaml .

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
