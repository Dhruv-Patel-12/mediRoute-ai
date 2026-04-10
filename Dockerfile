# Use official Python lightweight image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# System dependencies (for healthchecks)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (This will copy model.pkl and features.pkl)
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to ensure container runs correctly
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the application (Serving ONLY)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]