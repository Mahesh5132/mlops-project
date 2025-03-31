FROM python:3.11.4

# Set the working directory
WORKDIR /app

# Copy the necessary files
COPY app_requirements.txt .
COPY model.pkl .
COPY app.py .
COPY templates/ templates/

# Install dependencies
RUN pip install --no-cache-dir -r app_requirements.txt

# Expose the port the Flask app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
