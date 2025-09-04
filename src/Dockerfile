FROM python:3.12-slim

# Set the working directory in the image
WORKDIR /app

# Copy the requirements file into the image
COPY requirements.txt .

# Install the packages in the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the image
COPY connection/ ./connection/
COPY params/ ./params/
COPY utils/ ./utils/
COPY config.yaml .
COPY main.py .

# Set the default command
CMD ["python", "-u", "main.py"]
