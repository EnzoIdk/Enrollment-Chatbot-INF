# Enrollment-Chatbot-INF
Chatbot para consulta directa en periodos de matrícula por parte de estudiantes de Ingeniería Informática en la PUCP

## Lectura de chats de Discord
Primero establecer y conectarte a tu venv, luego instalar las dependencias
```bash
pip install -r requirements.txt
```
Dirigirse a la carpeta de discord
```bash
cd .\discord\
```
Despertar a FloresAI (bot)
```bash
py '.\extraccion_Q&A.py'
```

Este bot por ahora devuelve un dataset para el entrenamiento agrupando las preguntas y respuestas (a Luis Flores) que encuentre en el canal que lo llamen.

## Arquitectura del modelo (Temporalmente)
<img width="1119" height="770" alt="image" src="https://github.com/user-attachments/assets/28d2f03b-548d-4390-8b97-490cad0a9801" />
