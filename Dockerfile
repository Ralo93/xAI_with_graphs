FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/code/src:$PYTHONPATH"

# Set working directory
WORKDIR /code

# Copy requirements
COPY requirements/base.txt requirements/base.txt
COPY requirements/prod.txt requirements/prod.txt

# Install production dependencies
RUN pip install --no-cache-dir -r requirements/prod.txt

# Copy source code
COPY src /code/src

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]