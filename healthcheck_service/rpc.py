# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

import json
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from web3 import Web3
from web3.types import BlockIdentifier


logging.basicConfig(
    filename="rpc_monitor.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.Formatter.converter = time.gmtime

load_dotenv()

class RPCCheckHandler:
    """RPCcheck handler."""

    def __init__(self, *args, **kwargs) -> None:

        self.rpc_endpoint = os.getenv("RPC_ENDPOINT", "http://localhost:8545")
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_endpoint))
        self.mech_contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(os.getenv("MECH_CONTRACT_ADDRESS")),
            abi=self._get_abi(name="abi.json"),
        )
        self.safe_contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(os.getenv("SAFE_CONTRACT_ADDRESS")),
            abi=self._get_abi(name="safe.json"),
        )

    def _get_abi(self, name: str) -> Dict[str, Any]:
        """Get the abi of the contract."""
        path = Path(__file__).parent / "web3" / name
        with open(str(path)) as f:
            abi = json.load(f)
        return abi

    def check_rpc_connection(self) -> bool:
        try:
            if not self.web3.is_connected(show_traceback=True):
                log_message = "RPC node is not reachable."
                logging.error(log_message)
                return False
            return True
        except Exception as e:
            log_message = f"Error while checking RPC connection: {e}"
            logging.error(log_message)
            return False

    def get_deliver_events(self, from_block: BlockIdentifier) -> List[Dict[str, Any]]:
        """Get the deliver events."""
        try:
            return self.mech_contract.events.Deliver.create_filter(
                fromBlock=from_block
            ).get_all_entries()
        except Exception as e:
            logging.error(f"Error fetching Deliver events: {e}")
            return []

    def get_request_events(self, from_block: BlockIdentifier) -> List[Dict[str, Any]]:
        """Get the request events."""
        try:
            return self.mech_contract.events.Request.create_filter(
                fromBlock=from_block
            ).get_all_entries()
        except Exception as e:
            logging.error(f"Error fetching Request events: {e}")
            return []

    def get_relevant_events(self) -> List[Dict[str, Any]]:
        """Get the events events."""
        from_block = self.web3.eth.block_number - 500
        requests = self.get_request_events(from_block)
        delivers = self.get_deliver_events(from_block)
        logging.info(f"Request events: {len(requests)}")
        logging.info(f"Deliver events: {len(delivers)}")
        return requests + delivers

    def get_current_block(self) -> int:
        """Get the block timestamp."""
        try:
            block = self.web3.eth.get_block("latest")
            logging.info(f"Latest block: {block['number']}")
            return block["number"]
        except Exception as e:
            logging.error(f"Error fetching latest block: {e}")
            return -1

    def get_current_nonce(self) -> int:
        """Get the nonce from safe."""
        try:
            nonce = self.safe_contract.functions.nonce().call()
            logging.info(f"Latest nonce: {nonce}")
            return nonce
        except Exception as e:
            logging.error(f"Error fetching latest nonce: {e}")
            return -1


def run_rpc_checks() -> None:
    rpc_handler = RPCCheckHandler()

    while True:
        logging.info("Starting RPC checks...")
        if not rpc_handler.check_rpc_connection():
            logging.error("Failed to connect to RPC. Retrying...")
            time.sleep(10)
            continue

        current_block = rpc_handler.get_current_block()
        if current_block == -1:
            logging.error("Failed to fetch current block. Retrying...")
            time.sleep(10)
            continue

        current_nonce = rpc_handler.get_current_nonce()
        if current_nonce == -1:
            logging.error("Failed to fetch current nonce. Retrying...")
            time.sleep(10)
            continue

        events = rpc_handler.get_relevant_events()
        if not events:
            logging.info("No relevant events found.")
        else:
            logging.info(f"Found {len(events)} relevant events.")

        logging.info("Sleeping before the next check...")
        time.sleep(15)


if __name__ == "__main__":
    run_rpc_checks()
