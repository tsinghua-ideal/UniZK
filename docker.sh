#!/bin/bash

# Define the image name and container name
IMAGE_NAME="ubuntu:22.04"
CONTAINER_NAME="unizk"
HOST_FOLDER="."
CONTAINER_FOLDER="/mnt/data"
SCRIPT_NAME="dependency.sh"

chmod +x "$HOST_FOLDER/$SCRIPT_NAME"

# Pull the Ubuntu image
echo "Pulling the Ubuntu image..."
docker pull $IMAGE_NAME

# Create and start a new container, mounting the host folder to the container
echo "Creating and starting the container..."
docker run --name $CONTAINER_NAME -d -v $HOST_FOLDER:$CONTAINER_FOLDER $IMAGE_NAME tail -f /dev/null

# Execute the .sh file inside the container
echo "Executing $SCRIPT_NAME inside the container..."
# docker exec $CONTAINER_NAME bash "$CONTAINER_FOLDER/$SCRIPT_NAME"

# Enter the container's bash
echo "Entering the container's bash..."
docker exec -it $CONTAINER_NAME bash -c "cp -r /mnt/data /UniZK"

