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
"""Contains a small healthcheck checker"""

import json
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from web3.types import BlockIdentifier

from web3 import Web3

logging.basicConfig(
    filename="healthcheck_monitor.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class MechContract:
    def __init__(self, rpc_endpoint: str, contract_address: str) -> None:
        """Setup the base event tracker"""
        self.rpc_endpoint = rpc_endpoint
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_endpoint))
        self.contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=self._get_abi(),
        )

    def _get_abi(self) -> Dict[str, Any]:
        """Get the abi of the contract."""
        path = Path(__file__).parent / "web3" / "abi.json"
        with open(str(path)) as f:
            abi = json.load(f)
        return abi

    def get_deliver_events(self, from_block: BlockIdentifier) -> List[Dict[str, Any]]:
        """Get the deliver events."""
        return self.contract.events.Deliver.create_filter(
            fromBlock=from_block
        ).get_all_entries()

    def get_request_events(self, from_block: BlockIdentifier) -> List[Dict[str, Any]]:
        """Get the request events."""
        return self.contract.events.Request.create_filter(
            fromBlock=from_block
        ).get_all_entries()

    def get_unfulfilled_request(self) -> List[Dict[str, Any]]:
        """Get the unfulfilled events."""
        from_block = self.web3.eth.block_number - 5000
        delivers = self.get_deliver_events(from_block)
        requests = self.get_request_events(from_block)
        undeleted_requests = []
        deliver_req_ids = [deliver["args"]["requestId"] for deliver in delivers]

        for request in requests:
            if request["args"]["requestId"] not in deliver_req_ids:
                undeleted_requests.append(request)
        return undeleted_requests

    def get_block_timestamp(self, block_number: int) -> int:
        """Get the block timestamp."""
        return self.web3.eth.get_block(block_number)["timestamp"]

    def earliest_unfulfilled_request_timestamp(self) -> Optional[int]:
        """Get the earliest unfulfilled request."""
        unfulfilled_requests = self.get_unfulfilled_request()
        earliest_request = None
        for request in unfulfilled_requests:
            if (
                earliest_request is None
                or request["blockNumber"] < earliest_request["blockNumber"]
            ):
                earliest_request = request
        if earliest_request is not None:
            timestamp = self.get_block_timestamp(earliest_request["blockNumber"])
            return timestamp
        return None


class HealthCheckHandler:
    """Healthcheck handler."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the healthcheck handler."""
        self.mech_contract = MechContract(
            rpc_endpoint=os.getenv("RPC_ENDPOINT", "http://localhost:8545"),
            contract_address=os.getenv("MECH_CONTRACT_ADDRESS"),
        )
        self.grace_period = int(os.getenv("GRACE_PERIOD", 600))
        super().__init__(*args, **kwargs)

    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        req_timestamp = self.mech_contract.earliest_unfulfilled_request_timestamp()
        if req_timestamp is None:
            return True
        return req_timestamp + self.grace_period > time.time()


def run_healthcheck_server() -> None:
    """
    Run the health check.

    Returns:
        None
    """
    handler = HealthCheckHandler()
    while True:
        healthy = handler.is_healthy()
        status = "OK" if healthy else "NOT OK"
        message = f"Health check: {status}"
        (logging.info if healthy else logging.error)(message)
        time.sleep(15)


if __name__ == "__main__":
    run_healthcheck_server()
