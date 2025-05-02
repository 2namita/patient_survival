# Use an official Python base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app.py .
COPY xgboost-model.pkl .

# Expose the port Gradio will run on
EXPOSE 7860

# Start the app
CMD ["python", "app.py"]
