FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (ffmpeg is required for moviepy and pydub)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the default Streamlit port (or Render dynamic port)
EXPOSE 8000

# Start the Streamlit app
CMD sh -c "streamlit run app.py --server.port ${PORT:-8000} --server.address 0.0.0.0"
