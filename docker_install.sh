#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Updating package lists..."
sudo apt-get update

echo "Installing CA certificates and curl..."
sudo apt-get install -y ca-certificates curl

echo "Creating the /etc/apt/keyrings directory..."
sudo install -m 0755 -d /etc/apt/keyrings

echo "Downloading and saving the Docker GPG key..."
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc

echo "Setting permissions for Docker GPG key..."
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "Adding the Docker repository to APT sources..."
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "Updating package lists for Docker repository..."
sudo apt-get update

echo "Installing Docker Engine, CLI, containerd, and related plugins..."
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "Starting Docker service..."
sudo systemctl start docker

echo "Enabling Docker service to start on boot..."
sudo systemctl enable docker

echo "Creating docker group (if not exists) and adding current user..."
sudo groupadd docker || true
sudo usermod -aG docker $USER

echo "Docker installation complete!"
