# 1. Start with Python 3.11 (Fixes the networkx error)
FROM python:3.11

# 2. Set up the working directory
WORKDIR /app

# 3. Copy requirements and install them securely
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Download the SpaCy English Language Model
RUN python -m spacy download en_core_web_sm

# 5. Copy your main.py code into the server
COPY . .

# 6. Start the server on Hugging Face's required port (7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]