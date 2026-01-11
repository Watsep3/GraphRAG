# Base image
FROM python:3.10-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Exposer le port Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Commande de démarrage
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]