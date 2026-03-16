FROM python:3.12-slim

WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY api.py cpu_power_model.py data_loader.py ./

# cpu_data.dat is bind-mounted from the host at runtime
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
