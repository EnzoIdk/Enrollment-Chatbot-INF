# Enrollment-Chatbot-INF
Chatbot para consulta directa en periodos de matrícula por parte de estudiantes de Ingeniería Informática en la PUCP

## Lectura de chats de Discord
Primero establecer y conectarte a tu venv, luego instalar las dependencias
```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```
Despertar a FloresAI para extracción de dataset inicial
```bash
py '.\discord\extraccion_Q&A.py'
```

Anonimización de dataset extraída
```bash
py '.\discord\anonymization.py'
```

Despertar a FloresAI para resolución de consultas por Discord
```bash
py '.\main.py'
```

## Arquitectura del modelo (Temporalmente)
<img width="1119" height="770" alt="image" src="https://github.com/user-attachments/assets/28d2f03b-548d-4390-8b97-490cad0a9801" />
