cleanup() {
    echo "Terminating tendermint..."
    if kill -0 "$tm_subprocess_pid" 2>/dev/null; then
        kill "$tm_subprocess_pid"
        wait "$tm_subprocess_pid" 2>/dev/null
    fi
    echo "Tendermint terminated"
}

# Link cleanup to the exit signal
trap cleanup EXIT

# Remove previous agent if exists
if test -d agent; then
  echo "Removing previous agent build"
  sudo rm -r agent
fi

# Remove empty directories to avoid wrong hashes
find . -empty -type d -delete
make clean

# Ensure hashes are updated
autonomy packages lock

# Fetch the agent
autonomy fetch --local --agent valory/mech --alias agent

# Copy and add the keys, env and issue certificates
cd agent
cp $PWD/../.1env .
cp $PWD/../ethereum_private_key.txt .
autonomy add-key ethereum ethereum_private_key.txt
autonomy issue-certificates

set -o allexport
source .1env
set +o allexport


# Copy env variables
export SKILL_MECH_ABCI_MODELS_PARAMS_ARGS_ON_CHAIN_SERVICE_ID="$ON_CHAIN_SERVICE_ID"
export SKILL_TASK_EXECUTION_MODELS_PARAMS_ARGS_NUM_AGENTS="$NUM_AGENTS"
export SKILL_TASK_EXECUTION_MODELS_PARAMS_ARGS_TOOLS_TO_PACKAGE_HASH="$TOOLS_TO_PACKAGE_HASH"
export SKILL_TASK_EXECUTION_MODELS_PARAMS_ARGS_API_KEYS="$API_KEYS"
export CONNECTION_LEDGER_CONFIG_LEDGER_APIS_ETHEREUM_ADDRESS="$ETHEREUM_LEDGER_RPC_0"
export SKILL_MECH_ABCI_MODELS_PARAMS_ARGS_SETUP_ALL_PARTICIPANTS="$ALL_PARTICIPANTS"
export SKILL_MECH_ABCI_MODELS_PARAMS_ARGS_RESET_PAUSE_DURATION="$RESET_PAUSE_DURATION"
export SKILL_MECH_ABCI_MODELS_PARAMS_ARGS_SETUP_SAFE_CONTRACT_ADDRESS="$SAFE_CONTRACT_ADDRESS"
export SKILL_TASK_EXECUTION_MODELS_PARAMS_ARGS_MECH_TO_CONFIG="$MECH_TO_CONFIG"
export SKILL_MECH_ABCI_MODELS_PARAMS_ARGS_MECH_TO_SUBSCRIPTION="$MECH_TO_SUBSCRIPTION"

export CONNECTION_LEDGER_CONFIG_LEDGER_APIS_GNOSIS_ADDRESS="$CONNECTION_LEDGER_CONFIG_LEDGER_APIS_ETHEREUM_ADDRESS"

if [[ "$NETWORK"=="base" ]]; then
    export CONNECTION_LEDGER_CONFIG_LEDGER_APIS_ETHEREUM_CHAIN_ID=8453
    export SKILL_TASK_EXECUTION_MODELS_PARAMS_ARGS_MECH_MARKETPLACE_ADDRESS="0x735FAAb1c4Ec41128c367AFb5c3baC73509f70bB"
    export SKILL_MECH_ABCI_MODELS_PARAMS_ARGS_HASH_CHECKPOINT_ADDRESS="0x694e62BDF7Ff510A4EE66662cf4866A961a31653"
    export SKILL_MECH_ABCI_MODELS_PARAMS_ARGS_AGENT_REGISTRY_ADDRESS="0x3C1fF68f5aa342D296d4DEe4Bb1cACCA912D95fE"
elif [[ "$NETWORK" == "gnosis" ]]; then
    export CONNECTION_LEDGER_CONFIG_LEDGER_APIS_ETHEREUM_CHAIN_ID=100
    export SKILL_TASK_EXECUTION_MODELS_PARAMS_ARGS_MECH_MARKETPLACE_ADDRESS="0x735FAAb1c4Ec41128c367AFb5c3baC73509f70bB"
    export SKILL_MECH_ABCI_MODELS_PARAMS_ARGS_HASH_CHECKPOINT_ADDRESS="0x694e62BDF7Ff510A4EE66662cf4866A961a31653"
    export SKILL_MECH_ABCI_MODELS_PARAMS_ARGS_AGENT_REGISTRY_ADDRESS="0xE49CB081e8d96920C38aA7AB90cb0294ab4Bc8EA"
fi


DIR_NAME="tmp"
mkdir -p "$DIR_NAME"
export SKILL_MECH_ABCI_MODELS_BENCHMARK_TOOL_ARGS_LOG_DIR=$(realpath "$DIR_NAME")


# Run tendermint
rm -r ~/.tendermint
tendermint init > /dev/null 2>&1
echo "Starting Tendermint..."
tendermint node --proxy_app=tcp://127.0.0.1:26658 --rpc.laddr=tcp://127.0.0.1:26657 --p2p.laddr=tcp://0.0.0.0:26656 --p2p.seeds= --consensus.create_empty_blocks=true > /dev/null 2>&1 &
tm_subprocess_pid=$!


# Run the agent
aea -s run --env .1env
