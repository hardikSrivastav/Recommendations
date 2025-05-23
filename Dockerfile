# syntax=docker/dockerfile:1
FROM python:3.10-alpine
WORKDIR /code
ENV FLASK_APP=application
ENV FLASK_RUN_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers
COPY server/requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8000
COPY . .
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]