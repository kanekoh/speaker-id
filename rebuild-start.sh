#!/bin/bash

podman stop speaker-id
podman rm speaker-id
lsof -i:8082 | tail -n1 | awk '{print $2}' | xargs kill -9
podman build -t speaker-id .
podman run -d -p 8082:8000 --name speaker-id -e SPEAKERID_API_KEY=${SPEAKER_API_KEY} -v $(pwd)/profiles:/app/profiles speaker-id
