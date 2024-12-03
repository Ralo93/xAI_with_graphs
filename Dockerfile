# Use a lightweight Python base image
FROM python:3.12

# Set the working directory
WORKDIR /code

# Copy the requirements file from our folder into /code folder in the container
COPY ./requirements.txt /code/requirements.txt

# Copy the app code
COPY ./app /code/app

# Install the requirements.txt file inside the container
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


# Expose the port for the FastAPI app
EXPOSE 8080

# Define the command to run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
