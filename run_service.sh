#!/usr/bin/env bash

rm -r mech

# Load env vars
set -o allexport; source .1env; set +o allexport

GNOSIS_RPC_0="$ETHEREUM_LEDGER_RPC_0"
GNOSIS_RPC_1="$ETHEREUM_LEDGER_RPC_1"
GNOSIS_RPC_2="$ETHEREUM_LEDGER_RPC_2"
GNOSIS_RPC_3="$ETHEREUM_LEDGER_RPC_3"

# Remove previous builds
# if [ -d "mech" ]; then
#     echo $PASSWORD | sudo -S sudo rm -Rf mech;
# fi

# Push packages and fetch service
# make formatters
# make generators
make clean

autonomy push-all

autonomy fetch --local --service valory/mech && cd mech

# Build the image
autonomy build-image

# Copy keys and build the deployment
cp $PWD/../keys.json .

autonomy deploy build -ltm

# Run the deployment
autonomy deploy run --build-dir abci_build/