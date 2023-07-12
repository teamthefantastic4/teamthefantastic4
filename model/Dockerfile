FROM python:3.8.13 as base

ARG UID
ARG GID

# Add a non-root user
RUN addgroup --gid $GID app
RUN adduser --disabled-login --geco '' --uid $UID --gid $GID app

# Append /home/app/.local/bin/ to PATH variable because
# gunicorn is installed there.
ENV PYTHONPATH=$PYTHONPATH:/src/
ENV PATH=$PATH:/home/app/.local/bin/

ENV PYTHONPATH=$PYTHONPATH:/src/

COPY ./ /src/

WORKDIR /src

# add current user and group inside of the src directory recursively
# user app and group app 
RUN chown -R app:app /src/
# From Alan Bauzá
# add read and write permissions for users
# give only read permissions for group
# recursively in src directory
RUN chmod u+rw,g=r -R /src/

USER app

ADD requirements.txt .
RUN pip3 install -r requirements.txt

FROM base as test
RUN ["pytest", "-v", "/src/tests"]

FROM base as build
ENTRYPOINT ["python3", "/src/ml_service.py"]

