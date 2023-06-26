FROM python

WORKDIR /usr/src/app

COPY pyproject.toml pdm.lock README.md .
COPY map_generation map_generation

RUN pip3 install .[prod]

COPY models/model models/model
COPY examples examples
EXPOSE 5000

CMD ["streamlit", "run", "map_generation/app/app.py", "--server.port", "5000"]