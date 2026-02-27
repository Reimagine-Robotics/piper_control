#!/bin/bash

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

error() {
  echo -e "[${RED}error${NC}]: ${1}" >&2
  exit 1
}

warning() {
  echo -e "[${YELLOW}warning${NC}]: ${1}" >&2
}

info() {
  echo -e "[${GREEN}info${NC}]: ${1}" >&2
}

usage() {
    echo "Usage: $0 -i <can_interface> [-n <desired_name>] [-b <bitrate>] [-h]"
    echo ""
    echo "Options:"
    echo "  -i <interface>    CAN interface to configure (required, e.g., can0, can1)"
    echo "  -n <name>         Desired name for the interface (optional, defaults to original name)"
    echo "  -b <bitrate>      CAN bitrate in bps (optional, default: 1000000)"
    echo "  -h                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -i can0 -n vehicle-can -b 1000000"
    echo "  $0 -i can1 -n engine-can -b 500000"
    echo "  $0 -i can0 -b 250000  # Keep original name, just set bitrate"
    echo "  $0 -i can0            # Use defaults (original name, 1Mbps bitrate)"
    exit 0
}

# Default values
BITRATE="1000000"
CAN_INTERFACE=""
DESIRED_NAME=""
RENAME_INTERFACE=false

# Parse arguments using getopts
while getopts "i:n:b:h" opt; do
    case $opt in
        i) CAN_INTERFACE="$OPTARG" ;;
        n) DESIRED_NAME="$OPTARG"; RENAME_INTERFACE=true ;;
        b) BITRATE="$OPTARG" ;;
        h) usage ;;
        \?) error "Invalid option: -$OPTARG. Use -h for help." ;;
        :) error "Option -$OPTARG requires an argument." ;;
    esac
done

# Shift positional parameters
shift $((OPTIND-1))

# Check required parameters
if [ -z "$CAN_INTERFACE" ]; then
    error "CAN interface is required. Use -i <interface>. Use -h for help."
fi

# If no desired name specified, use the original interface name
if [ -z "$DESIRED_NAME" ]; then
    DESIRED_NAME="$CAN_INTERFACE"
    info "No custom can interface name specified, keeping original name: $DESIRED_NAME"
else
    # Validate desired interface name format (allow alphanumeric, hyphens, underscores)
    if ! [[ "$DESIRED_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        error "Invalid desired interface name '$DESIRED_NAME'. Use only letters, numbers, hyphens, and underscores."
    fi
    # Check length (Linux interface names are limited to 15 characters)
    if [ ${#DESIRED_NAME} -gt 15 ]; then
        error "Desired interface name '$DESIRED_NAME' is too long (max 15 characters)"
    fi
fi

# Check if running as root (always needed for creating udev rules)
if [ "$EUID" -ne 0 ]; then
    error "Please run with sudo to automatically save the udev rule"
fi

# Validate bitrate
if ! [[ "$BITRATE" =~ ^[0-9]+$ ]]; then
    error "Bitrate must be a number (e.g., 1000000 for 1Mbps)"
fi

# Check if the CAN interface exists
if [ ! -d "/sys/class/net/$CAN_INTERFACE" ]; then
    error "CAN interface '$CAN_INTERFACE' not found"
fi

info "Configuration:"
echo "  Source interface: $CAN_INTERFACE"
echo "  Target name: $DESIRED_NAME"
echo "  Bitrate: $BITRATE bps"
echo ""

# Always create udev rule for persistent configuration
info "Creating udev rule for persistent CAN interface configuration..."

# Get device information
DEVICE_PATH=$(readlink -f "/sys/class/net/$CAN_INTERFACE")
if [ ! -d "$DEVICE_PATH" ]; then
    error "Could not find device path for $CAN_INTERFACE"
fi

# Extract vendor and product IDs
VENDOR=$(udevadm info -p "$DEVICE_PATH" -q property | grep ID_VENDOR_ID | cut -d= -f2)
PRODUCT=$(udevadm info -p "$DEVICE_PATH" -q property | grep ID_MODEL_ID | cut -d= -f2)
SERIAL=$(udevadm info -p "$DEVICE_PATH" -q property | grep ID_SERIAL_SHORT | cut -d= -f2)

info "Detected device information:"
echo "  Interface: $CAN_INTERFACE"
echo "  Vendor ID: $VENDOR"
echo "  Product ID: $PRODUCT"
echo "  Serial: $SERIAL"
echo ""

# Validate that we have all required device identifiers
if [ -z "$VENDOR" ] || [ -z "$PRODUCT" ] || [ -z "$SERIAL" ]; then
    error "Could not detect all required identifiers for device $CAN_INTERFACE:
  Vendor ID: ${VENDOR:-NOT FOUND}
  Product ID: ${PRODUCT:-NOT FOUND}
  Serial: ${SERIAL:-NOT FOUND}"
fi

# Generate udev rule content with conditional NAME attribute
info "Creating rule with vendor, product, and serial for unique identification"
if [ "$RENAME_INTERFACE" = true ]; then
    UDEV_RULE="SUBSYSTEM==\"net\", ATTRS{idVendor}==\"$VENDOR\", ATTRS{idProduct}==\"$PRODUCT\", ATTRS{serial}==\"$SERIAL\", ACTION==\"add\", NAME=\"$DESIRED_NAME\", RUN+=\"/bin/sh -c 'ip link set $DESIRED_NAME down; ip link set $DESIRED_NAME type can bitrate $BITRATE; ip link set $DESIRED_NAME up'\""
else
    UDEV_RULE="SUBSYSTEM==\"net\", ATTRS{idVendor}==\"$VENDOR\", ATTRS{idProduct}==\"$PRODUCT\", ATTRS{serial}==\"$SERIAL\", ACTION==\"add\", RUN+=\"/bin/sh -c 'ip link set \$env{INTERFACE} down; ip link set \$env{INTERFACE} type can bitrate $BITRATE; ip link set \$env{INTERFACE} up'\""
fi

# Define the udev rule file path
UDEV_RULE_FILE="/etc/udev/rules.d/99-can-$DESIRED_NAME.rules"

# Save the udev rule
echo "# CAN interface rule for $DESIRED_NAME" > "$UDEV_RULE_FILE"
echo "# Generated on $(date)" >> "$UDEV_RULE_FILE"
echo "# Interface: $CAN_INTERFACE -> $DESIRED_NAME" >> "$UDEV_RULE_FILE"
echo "# Bitrate: $BITRATE" >> "$UDEV_RULE_FILE"
echo "" >> "$UDEV_RULE_FILE"
echo "$UDEV_RULE" >> "$UDEV_RULE_FILE"

if [ $? -eq 0 ]; then
    info "Udev rule saved successfully"
else
    error "Failed to save udev rule"
fi

# Set proper permissions
chmod 644 "$UDEV_RULE_FILE"

# Reload udev rules
info "Reloading udev rules..."
sudo udevadm control --reload-rules && sudo udevadm trigger

echo ""
info "Setup complete! Your CAN interface should now be available as '$DESIRED_NAME'"
info "To test: unplug and replug your CAN adapter, then run 'ip link show $DESIRED_NAME'"
echo ""
info "To remove this rule later, delete: $UDEV_RULE_FILE"
