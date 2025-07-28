# Dockerfile for Challenge 1b - Intelligent Document Analyst
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for PyMuPDF and NLTK
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application
COPY main.py .

# Copy sample collections for testing
COPY ["Collection 1/", "./Collection_1/"]
COPY ["Collection 2/", "./Collection_2/"]
COPY ["Collection 3/", "./Collection_3/"]

# Create output directory
RUN mkdir -p /app/output

# Set the entrypoint to run the main.py script
ENTRYPOINT ["python", "main.py"]

# Default command arguments - can be overridden
CMD ["Collection_1/challenge1b_input.json", "output/collection1_output.json"]
