# Use the official Python image from the Docker Hub
FROM python:3.9

# Set environment variables
ENV PIP_NO_CACHE_DIR=1

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Install the dependencies
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app_st.py", "--server.port=8501", "--server.address=0.0.0.0"]
