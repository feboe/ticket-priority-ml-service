FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    API_BASE_URL=http://127.0.0.1:8000

COPY requirements-app.txt ./requirements-app.txt
RUN pip install --no-cache-dir -r requirements-app.txt

COPY . .

EXPOSE 8000 8501

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
