# Use a lightweight Python image
FROM python:3.10-slim

# 1. Install the missing system libraries MediaPipe needs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgles2 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory
WORKDIR /app

# 3. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your code
COPY . .

# 5. Run the app with Gunicorn
CMD ["gunicorn", "app:app", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:10000"]