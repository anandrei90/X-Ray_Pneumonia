# import python base image
FROM python:3.12-slim

# create working directory (root)
WORKDIR /app

# copy requirements into root dir
COPY requirements.txt .

# install requirements with pip
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install tensorflow-cpu==2.16.1

# copy backend contents into backend root ("." =  /app/backend).
# do this selectively, i.e. manually chose folders!
COPY api/api_client.py ./api/api_client.py
COPY api/api_helpers.py ./api/api_helpers.py
COPY api/api_server.py ./api/api_server.py
COPY data/test ./data/test
COPY data/helpers.py ./data/helpers.py
COPY unified_experiment/mlartifacts ./unified_experiment/mlartifacts
COPY unified_experiment/mlruns ./unified_experiment/mlruns

# copy start script from "docker" to parent directory (inro root of WORKDIR!)
COPY docker/run_app_docker.sh /app/run_app_docker.sh

# open communication ports
EXPOSE 8000 8080

# command ran during the docker building process
CMD ["bash", "run_app_docker.sh"]