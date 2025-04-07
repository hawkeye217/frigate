#!/bin/bash

set -e  # Exit on error
set -o pipefail

echo "Starting MemryX driver and runtime installation..."

# Detect architecture
arch=$(uname -m)

if [[ -d /sys/memx0/ ]]; then
    echo "Existing functional MX3 driver found. Skipping driver re-install."
else

    # Purge existing packages and repo
    echo "Removing old MemryX installations..."
    sudo apt purge -y memx-* || true
    sudo rm -f /etc/apt/sources.list.d/memryx.list /etc/apt/trusted.gpg.d/memryx.asc

    # Install kernel headers
    echo "Installing kernel headers for: $(uname -r)"
    sudo apt update
    sudo apt install -y linux-headers-$(uname -r)

    # Add MemryX key and repo
    echo "Adding MemryX GPG key and repository..."
    wget -qO- https://developer.memryx.com/deb/memryx.asc | sudo tee /etc/apt/trusted.gpg.d/memryx.asc >/dev/null
    echo 'deb https://developer.memryx.com/deb stable main' | sudo tee /etc/apt/sources.list.d/memryx.list >/dev/null

    # Update and install packages
    echo "Installing memx-drivers..."
    sudo apt update
    sudo apt install -y memx-drivers

    # ARM-specific board setup
    if [[ "$arch" == "aarch64" || "$arch" == "arm64" ]]; then
        echo " Running ARM board setup..."
        sudo mx_arm_setup
    fi

    echo -e "\n\n\033[1;31mYOU MUST RESTART YOUR COMPUTER NOW\033[0m\n\n"
fi

# Install mxa-manager
echo "Installing mxa-manager..."
sudo apt install -y memx-accl mxa-manager


echo "MemryX installation complete!"
