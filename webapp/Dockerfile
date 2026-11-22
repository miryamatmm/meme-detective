FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY .env .env
RUN pip3 install --upgrade pip --default-timeout=1000
RUN pip3 install --no-cache-dir -r requirements.txt --default-timeout=1000

# Port par défaut de Streamlit
EXPOSE 8501

# Commande pour lancer Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]