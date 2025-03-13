#!/usr/bin/env bash

rm -r mech

# Load env vars
set -o allexport; source .1env; set +o allexport

ETHEREUM_LEDGER_RPC_1="$ETHEREUM_LEDGER_RPC_0"
ETHEREUM_LEDGER_RPC_2="$ETHEREUM_LEDGER_RPC_0"
ETHEREUM_LEDGER_RPC_3="$ETHEREUM_LEDGER_RPC_0"
GNOSIS_RPC_0="$ETHEREUM_LEDGER_RPC_0"
GNOSIS_RPC_1="$ETHEREUM_LEDGER_RPC_0"
GNOSIS_RPC_2="$ETHEREUM_LEDGER_RPC_0"
GNOSIS_RPC_3="$ETHEREUM_LEDGER_RPC_0"

if [[ "$NETWORK"=="base" ]]; then
    MECH_MARKETPLACE_ADDRESS="0x735FAAb1c4Ec41128c367AFb5c3baC73509f70bB"
    HASH_CHECKPOINT_ADDRESS="0x694e62BDF7Ff510A4EE66662cf4866A961a31653"
    AGENT_REGISTRY_ADDRESS="0x3C1fF68f5aa342D296d4DEe4Bb1cACCA912D95fE"
    ETHEREUM_LEDGER_CHAIN_ID=8453
elif [[ "$NETWORK" == "gnosis" ]]; then
    MECH_MARKETPLACE_ADDRESS="0x735FAAb1c4Ec41128c367AFb5c3baC73509f70bB"
    HASH_CHECKPOINT_ADDRESS="0x694e62BDF7Ff510A4EE66662cf4866A961a31653"
    AGENT_REGISTRY_ADDRESS="0xE49CB081e8d96920C38aA7AB90cb0294ab4Bc8EA"
    ETHEREUM_LEDGER_CHAIN_ID=100
fi

DIR_NAME="tmp"
mkdir -p "$DIR_NAME"
LOG_DIR=$(realpath "$DIR_NAME")

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
cd mech
build_dir=$(ls -d abci_build_????/ 2>/dev/null || echo "abci_build")
autonomy deploy run --build-dir "$build_dir"