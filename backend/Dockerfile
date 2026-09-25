FROM python:3.11-slim

WORKDIR /app

# Copy requirements first, separately from the rest of the code -- this is
# a deliberate ordering, not accidental. Docker caches each instruction as
# a "layer"; if requirements.txt hasn't changed, Docker reuses the cached
# pip install layer even when your actual code changes, so rebuilds after
# a small code edit take seconds instead of reinstalling every package
# from scratch every time.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]