# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies (optional)
# If your script has dependencies, you can use a requirements.txt file to install them
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y \
build-essential \
libssl-dev \
libffi-dev \
python3-dev
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container (optional, only if your app uses networking)
# EXPOSE 80

# Run script.py when the container launches
CMD ["python", "./major.py"]
