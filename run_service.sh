#!/usr/bin/env bash

rm -r mech

# Load env vars
set -o allexport; source .1env; set +o allexport

export ETHEREUM_LEDGER_RPC_1="$ETHEREUM_LEDGER_RPC_0"
export ETHEREUM_LEDGER_RPC_2="$ETHEREUM_LEDGER_RPC_0"
export ETHEREUM_LEDGER_RPC_3="$ETHEREUM_LEDGER_RPC_0"
export GNOSIS_RPC_0="$ETHEREUM_LEDGER_RPC_0"
export GNOSIS_RPC_1="$ETHEREUM_LEDGER_RPC_0"
export GNOSIS_RPC_2="$ETHEREUM_LEDGER_RPC_0"
export GNOSIS_RPC_3="$ETHEREUM_LEDGER_RPC_0"
export MECH_TO_CONFIG="{\"$MECH_ADDRESS\":{\"use_dynamic_pricing\":false,\"is_marketplace_mech\":true}}"
export SKILL_MECH_ABCI_MODELS_PARAMS_ARGS_MECH_TO_SUBSCRIPTION="{\"$MECH_ADDRESS\":{\"tokenAddress\":\"0x0000000000000000000000000000000000000000\",\"tokenId\":\"1\"}}"

if [[ "$NETWORK"=="base" ]]; then
    export MECH_MARKETPLACE_ADDRESS="0x735FAAb1c4Ec41128c367AFb5c3baC73509f70bB"
    export CHECKPOINT_ADDRESS="0x694e62BDF7Ff510A4EE66662cf4866A961a31653"
    export AGENT_REGISTRY_ADDRESS="0x3C1fF68f5aa342D296d4DEe4Bb1cACCA912D95fE"
    export ETHEREUM_LEDGER_CHAIN_ID=8453
    if [["$USE_SUBSCRIPTION"]]; then 
        export MECH_TO_SUBSCRIPTION="{\"$MECH_ADDRESS\":{\"tokenAddress\":\"0xd5318d1A17819F65771B6c9277534C08Dd765498\",\"tokenId\":\"0x6f74c18fae7e5c3589b99d7cd0ba317593f00dee53c81a2ba4ac2244232f99da\"}}"
    fi 
elif [[ "$NETWORK" == "gnosis" ]]; then
    export MECH_MARKETPLACE_ADDRESS="0x735FAAb1c4Ec41128c367AFb5c3baC73509f70bB"
    export CHECKPOINT_ADDRESS="0x694e62BDF7Ff510A4EE66662cf4866A961a31653"
    export AGENT_REGISTRY_ADDRESS="0xE49CB081e8d96920C38aA7AB90cb0294ab4Bc8EA"
    export ETHEREUM_LEDGER_CHAIN_ID=100
    if [["$USE_SUBSCRIPTION"]]; then 
        export MECH_TO_SUBSCRIPTION="{\"$MECH_ADDRESS\":{\"tokenAddress\":\"0x1b5DeaD7309b56ca7663b3301A503e077Be18cba\",\"tokenId\":\"0xb0b28402e5a7229804579d4ac55b98a1dd94660d7a7eb4add78e5ca856f2aab7\"}}"
    fi 
fi

DIR_NAME="tmp"
mkdir -p "$DIR_NAME"
export LOG_DIR=$(realpath "$DIR_NAME")

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

autonomy deploy build -ltm --n "$NUM_AGENTS"

# Run the deployment
autonomy deploy run --build-dir abci_build/
cd mech
build_dir=$(ls -d abci_build_????/ 2>/dev/null || echo "abci_build")
autonomy deploy run --build-dir "$build_dir"