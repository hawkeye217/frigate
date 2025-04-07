#!/bin/bash

# Stop and remove existing container
docker stop frigate-memx
docker rm frigate-memx

# Build the new Docker image
sudo docker build -t frigate-memx -f docker/main/Dockerfile .

# Run the new container
sudo docker run -d \
  --name frigate-memx \
  --restart=unless-stopped \
  --mount type=tmpfs,target=/tmp/cache,tmpfs-size=1000000000 \
  --shm-size=256m \
  -v /home/memryx/final/Frigate_MemryX/config:/config \
  -e FRIGATE_RTSP_PASSWORD='password' \
  --add-host=host.docker.internal:host-gateway \
  --privileged=true \
  -p 8971:8971 \
  -p 8554:8554 \
  -p 5000:5000 \
  -p 8555:8555/tcp \
  -p 8555:8555/udp \
  --device /dev/memx0 frigate-memx

echo "Frigate container restarted successfully."
