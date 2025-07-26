# This file contains the instructions to build a Docker image for the application.

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No API key is needed for a local model
# ENV OPENAI_API_KEY="your_openai_api_key_here"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]