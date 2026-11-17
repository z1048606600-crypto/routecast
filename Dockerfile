FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home app && mkdir -p runtime && chown -R app:app /app
USER app
EXPOSE 8001
CMD ["uvicorn", "routecast.api:app", "--host", "0.0.0.0", "--port", "8001"]
