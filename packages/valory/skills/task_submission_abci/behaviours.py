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

"""This package contains round behaviours of TaskExecutionAbciApp."""
import abc
import json
import threading
import time
from abc import ABC
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional, Set, Type, cast, Tuple

import openai  # noqa
from multibase import multibase
from multicodec import multicodec

from packages.valory.contracts.agent_mech.contract import (
    AgentMechContract,
    MechOperation,
)
from packages.valory.contracts.agent_registry.contract import AgentRegistryContract
from packages.valory.contracts.gnosis_safe.contract import (
    GnosisSafeContract,
    SafeOperation,
)
from packages.valory.contracts.hash_checkpoint.contract import HashCheckpointContract
from packages.valory.contracts.multisend.contract import (
    MultiSendContract,
    MultiSendOperation,
)
from packages.valory.contracts.service_registry.contract import ServiceRegistryContract
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.task_submission_abci.models import Params
from packages.valory.skills.task_submission_abci.payloads import TransactionPayload
from packages.valory.skills.task_submission_abci.rounds import (
    SynchronizedData,
    TaskPoolingPayload,
    TaskPoolingRound,
    TaskSubmissionAbciApp,
    TransactionPreparationRound,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)


ZERO_ETHER_VALUE = 0
AUTO_GAS = SAFE_GAS = 0
DONE_TASKS = "ready_tasks"
DONE_TASKS_LOCK = "lock"
NO_DATA = b""


class TaskExecutionBaseBehaviour(BaseBehaviour, ABC):
    """Base behaviour for the task_execution_abci skill."""

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def params(self) -> Params:
        """Return the params."""
        return cast(Params, super().params)

    @property
    def done_tasks(self) -> List[Dict[str, Any]]:
        """
        Return the done (ready) tasks from shared state.

        Use with care, the returned data here is NOT synchronized with the rest of the agents.

        :returns: the tasks
        """
        done_tasks = deepcopy(self.context.shared_state.get(DONE_TASKS, []))
        return cast(List[Dict[str, Any]], done_tasks)

    def done_tasks_lock(self) -> threading.Lock:
        """Get done_tasks_lock."""
        return self.context.shared_state[DONE_TASKS_LOCK]

    def remove_tasks(self, submitted_tasks: List[Dict[str, Any]]) -> None:
        """
        Pop the tasks from shared state.

        :param submitted_tasks: the done tasks that have already been submitted
        """
        # run this in a lock
        # the amount of done tasks will always be relatively low (<<20)
        # we can afford to do this in a lock
        with self.done_tasks_lock():
            done_tasks = self.done_tasks
            not_submitted = []
            for done_task in done_tasks:
                is_submitted = False
                for submitted_task in submitted_tasks:
                    if submitted_task["request_id"] == done_task["request_id"]:
                        is_submitted = True
                        break
                if not is_submitted:
                    not_submitted.append(done_task)
            self.context.shared_state[DONE_TASKS] = not_submitted

    @property
    def mech_addresses(self) -> List[str]:
        """Get the addresses of the MECHs."""
        return self.context.params.mech_addresses

    @staticmethod
    def to_multihash(hash_string: str) -> str:
        """To multihash string."""
        # Decode the Base32 CID to bytes
        cid_bytes = multibase.decode(hash_string)
        # Remove the multicodec prefix (0x01) from the bytes
        multihash_bytes = multicodec.remove_prefix(cid_bytes)
        # Convert the multihash bytes to a hexadecimal string
        hex_multihash = multihash_bytes.hex()
        return hex_multihash[6:]


class TaskPoolingBehaviour(TaskExecutionBaseBehaviour, ABC):
    """TaskPoolingBehaviour"""

    matching_round: Type[AbstractRound] = TaskPoolingRound

    def async_act(self) -> Generator:  # pylint: disable=R0914,R0915
        """Do the act, supporting asynchronous execution."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            # clean up the queue based on the outcome of the previous period
            self.handle_submitted_tasks()
            # sync new tasks
            payload_content = yield from self.get_payload_content()
            sender = self.context.agent_address
            payload = TaskPoolingPayload(sender=sender, content=payload_content)
        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
        self.set_done()

    def get_payload_content(self) -> Generator[None, None, str]:
        """Get the payload content."""
        done_tasks = yield from self.get_done_tasks(self.params.task_wait_timeout)
        return json.dumps(done_tasks)

    def get_done_tasks(self, timeout: float) -> Generator[None, None, List[Dict]]:
        """Wait for tasks to get done in the specified timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(self.done_tasks) == 0:
                one_second = 1.0
                yield from self.sleep(one_second)
                continue
            # there are done tasks, return all of them
            return self.done_tasks

        # no tasks are ready for this agent
        self.context.logger.info("No tasks were ready within the timeout")
        return []

    def handle_submitted_tasks(self) -> None:
        """Handle tasks that have been already submitted before (in a prev. period)."""
        submitted_tasks = cast(List[Dict[str, Any]], self.synchronized_data.done_tasks)
        self.context.logger.info(
            f"Tasks {submitted_tasks} has already been submitted. "
            f"Removing them from the list of tasks to be processed."
        )
        self.remove_tasks(submitted_tasks)


class FundsSplittingBehaviour(TaskExecutionBaseBehaviour, ABC):
    """FundsSplittingBehaviour"""

    def _get_num_requests_delivered(self) -> Generator[None, None, Optional[int]]:
        """Return the total number of requests delivered."""
        reqs_by_agent = yield from self._get_num_reqs_by_agent()
        if reqs_by_agent is None:
            self.context.logger.warning(
                "Could not get number of requests delivered. Don't split profits."
            )
            return None

        total_reqs = sum(reqs_by_agent.values())
        return total_reqs

    def _get_num_reqs_by_agent(self) -> Generator[None, None, Dict[str, int]]:
        """Return the total number of requests delivered."""
        # TODO: implement once the storage is ready
        return {
            "agent0": 0,
            "agent1": 1,
            "agent2": 2,
            "agent3": 3,
        }
        yield

    def _should_split_profits(self) -> Generator[None, None, bool]:
        """
        Returns true if profits from the mech should be split.

        Profits will be split based on the number of requests that have been delivered.
        I.e. We will be splitting every n-th request. Where, n- is configurable

        :returns: True if profits should be split, False otherwise.
        :yields: None
        """
        total_reqs = yield from self._get_num_requests_delivered()
        if total_reqs is None:
            self.context.logger.warning(
                "Could not get number of requests delivered. Don't split profits."
            )
            return False
        return total_reqs % self.params.profit_split_freq == 0

    def get_split_profit_txs(
        self,
    ) -> Generator[None, None, Optional[List[Dict[str, Any]]]]:
        """Get and split all the profits from all the mechs."""
        should_split_profits = yield from self._should_split_profits()
        if not should_split_profits:
            self.context.logger.info("Not splitting profits.")
            return []

        self.context.logger.info(f"Splitting profits {self.mech_addresses}.")
        txs = []
        for mech_address in self.mech_addresses:
            profits = yield from self._get_profits(mech_address)
            if profits is None:
                self.context.logger.error(
                    f"Could not get profits from mech {mech_address}. Don't split profits."
                )
                return None

            self.context.logger.info(f"Got {profits} profits from mech {mech_address}")
            split_funds = yield from self._split_funds(profits)
            if split_funds is None:
                self.context.logger.error(
                    f"Could not split profits from mech {mech_address}. Don't split profits."
                )
                return None

            self.context.logger.info(
                f"Split {profits} profits from mech {mech_address} into {split_funds}"
            )
            for receiver_address, amount in split_funds.items():
                tx = yield from self._get_transfer_tx(
                    mech_address, receiver_address, amount
                )
                if tx is None:
                    self.context.logger.error(
                        f"Could not get transfer tx from mech {mech_address} to {receiver_address}. "
                        f"Don't split profits."
                    )
                    return None
                txs.append(tx)

        return txs

    def _get_profits(self, address: str) -> Generator[None, None, int]:
        """Get the profits."""
        ledger_api_response = yield from self.get_ledger_api_response(
            performative=LedgerApiMessage.Performative.GET_STATE,
            ledger_callable="get_balance",
            account=address,
        )
        if ledger_api_response.performative != LedgerApiMessage.Performative.STATE:
            return False  # transition to await top-up round
        balance = cast(int, ledger_api_response.state.body.get("get_balance_result"))
        return balance

    def _split_funds(
        self, profits: int
    ) -> Generator[None, None, Optional[Dict[str, int]]]:
        """
        Split the funds among the operators based on the number of txs their agents have made.

        :param profits: the amount of funds to split.
        :returns: a dictionary mapping operator addresses to the amount of funds they should receive.
        :yields: None
        """
        service_owner = yield from self._get_service_owner(
            self.params.on_chain_service_id
        )
        if service_owner is None:
            self.context.logger.warning(
                "Could not get service owner. Don't split profits."
            )
            return None

        funds_by_address = {}
        service_owner_share = int(self.params.service_owner_share * profits)
        funds_by_address[service_owner] = service_owner_share

        operator_share = profits - service_owner_share
        funds_by_operator = yield from self._get_funds_by_operator(operator_share)
        if funds_by_operator is None:
            self.context.logger.warning(
                "Could not get funds by operator. Don't split profits."
            )
            return None

        # accumulate the funds
        funds_by_address.update(funds_by_operator)
        return funds_by_address

    def _get_transfer_tx(
        self, mech_address: str, receiver_address: str, amount: int
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get the transfer tx."""
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=mech_address,
            contract_id=str(AgentMechContract.contract_id),
            contract_callable="get_exec_tx_data",
            to=receiver_address,
            value=amount,
            data=NO_DATA,
            tx_gas=AUTO_GAS,
            operation=MechOperation.CALL,
        )
        if (
            contract_api_msg.performative != ContractApiMessage.Performative.STATE
        ):  # pragma: nocover
            self.context.logger.warning(
                f"get_exec_tx_data unsuccessful!: {contract_api_msg}"
            )
            return None

        data = cast(bytes, contract_api_msg.state.body["data"])
        return {
            "to": mech_address,
            # the safe is not moving any funds, the mech contract is
            "value": ZERO_ETHER_VALUE,
            "data": data,
        }

    def _get_service_owner(
        self, service_id: int
    ) -> Generator[None, None, Optional[str]]:
        """Get the service owner address."""
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=self.params.service_registry_address,
            contract_id=str(ServiceRegistryContract.contract_id),
            contract_callable="get_service_owner",
            service_id=service_id,
        )
        if contract_api_msg.performative != ContractApiMessage.Performative.STATE:
            self.context.logger.warning(
                f"get_service_owner unsuccessful!: {contract_api_msg}"
            )
            return None
        return cast(str, contract_api_msg.state.body["service_owner"])

    def _get_funds_by_operator(
        self, operator_share: int
    ) -> Generator[None, None, Optional[Dict[str, int]]]:
        """Split the funds among the operators based on the number of txs their agents have made."""
        reqs_by_agent = yield from self._get_num_reqs_by_agent()
        if reqs_by_agent is None:
            self.context.logger.warning(
                "Could not get number of requests delivered. Don't split profits."
            )
            return None

        total_reqs = sum(reqs_by_agent.values())
        if total_reqs == 0:
            # nothing to split
            return {agent: 0 for agent in reqs_by_agent.keys()}

        accumulated_reqs_by_operator = yield from self._accumulate_reqs_by_operator(
            reqs_by_agent
        )
        if accumulated_reqs_by_operator is None:
            self.context.logger.warning(
                "Could not get number of requests delivered. Don't split profits."
            )
            return None

        for agent, reqs in accumulated_reqs_by_operator.items():
            accumulated_reqs_by_operator[agent] = int(
                operator_share * (reqs / total_reqs)
            )

        return accumulated_reqs_by_operator

    def _accumulate_reqs_by_operator(
        self, reqs_by_agent: Dict[str, int]
    ) -> Generator[None, None, Optional[Dict[str, int]]]:
        """Accumulate requests by operator."""
        agent_instances = list(reqs_by_agent.keys())
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=self.params.agent_registry_address,
            contract_id=str(ServiceRegistryContract.contract_id),
            contract_callable="get_operators_mapping",
            agent_instances=agent_instances,
        )
        if (
            contract_api_msg.performative != ContractApiMessage.Performative.STATE
        ):  # pragma: nocover
            self.context.logger.warning(
                f"get_operators_mapping unsuccessful!: {contract_api_msg}"
            )
            return None
        agent_to_operator = cast(Dict[str, str], contract_api_msg.state.body)

        # accumulate reqs by operator
        reqs_by_operator = {}
        for agent, reqs in reqs_by_agent.items():
            operator = agent_to_operator[agent]
            if operator not in reqs_by_operator:
                reqs_by_operator[operator] = reqs
            else:
                reqs_by_operator[operator] += reqs
        return reqs_by_operator


class TrackingBehaviour(TaskExecutionBaseBehaviour, ABC):
    """Behaviour to track the execution of a task."""

    def _get_checkpoint_tx(
        self,
        hashcheckpoint_address: str,
        ipfs_hash: bytes,
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get the transfer tx."""
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=hashcheckpoint_address,
            contract_id=str(HashCheckpointContract.contract_id),
            contract_callable="get_checkpoint_data",
            ipfs_hash=ipfs_hash,
        )
        if (
            contract_api_msg.performative != ContractApiMessage.Performative.STATE
        ):  # pragma: nocover
            self.context.logger.warning(
                f"get_checkpoint_data unsuccessful!: {contract_api_msg}"
            )
            return None

        data = cast(bytes, contract_api_msg.state.body["data"])
        return {
            "to": hashcheckpoint_address,
            "value": ZERO_ETHER_VALUE,
            "data": data,
        }

    def _get_current_usage(
        self
    ) -> Generator[None, None, Optional[Dict[str, Any]]]:
        """Get the current ."""
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,
            contract_address=self.params.hash_checkpoint_address,
            contract_id=str(HashCheckpointContract.contract_id),
            contract_callable="get_latest_hash",
            sender_address=self.synchronized_data.safe_contract_address,
        )
        if contract_api_msg.performative != ContractApiMessage.Performative.STATE:
            self.context.logger.warning(
                f"get_latest_hash unsuccessful!: {contract_api_msg}"
            )
            return None
        latest_ipfs_hash = cast(str, contract_api_msg.state.body["data"])
        usage_data = yield from self.get_from_ipfs(latest_ipfs_hash)
        if usage_data is None:
            self.context.logger.warning(
                f"Could not get usage data from IPFS: {latest_ipfs_hash}"
            )
            return None
        return usage_data

    def _update_usage(
        self,
        current_usage: Dict[str, Any],
        done_tasks: List[Dict[str, Any]],
    ) -> Generator[None, None, Optional[Tuple[str, Dict[str, Any]]]]:
        """Update the usage of the tool on IPFS."""
        for task in done_tasks:
            agent, tool = task["executor_address"], task["tool"]
            if agent not in current_usage:
                current_usage[agent] = {}
            if tool not in current_usage[agent]:
                current_usage[agent][tool] = 0
            current_usage[agent][tool] += 1
        ipfs_hash = yield from self.send_to_ipfs("usage", current_usage)
        if ipfs_hash is None:
            self.context.logger.warning("Could not update usage.")
            return None
        return ipfs_hash, current_usage

    def get_update_usage_tx(self) -> Generator:
        """Get a tx to update the usage."""
        current_usage = yield from self._get_current_usage()
        if current_usage is None:
            # something went wrong
            self.context.logger.warning("Could not get current usage.")
            return None

        updated_usage, ipfs_hash = yield from self._update_usage(current_usage, self.synchronized_data.done_tasks)
        tx = yield from self._get_checkpoint_tx(self.params.hash_checkpoint_address, ipfs_hash)
        return tx


class HashUpdateBehaviour(TaskExecutionBaseBehaviour):
    """HashUpdateBehaviour"""

    def _get_latest_hash(self) -> Generator[None, None, Optional[bytes]]:
        """Get latest update hash."""
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=self.params.agent_registry_address,
            contract_id=str(AgentRegistryContract.contract_id),
            contract_callable="get_token_hash",
            token_id=self.params.agent_id,
        )
        if (
            contract_api_msg.performative != ContractApiMessage.Performative.STATE
        ):  # pragma: nocover
            self.context.logger.warning(
                f"get_token_hash unsuccessful!: {contract_api_msg}"
            )
            return None

        latest_hash = cast(bytes, contract_api_msg.state.body["data"])
        return latest_hash

    def _should_update_hash(self) -> Generator:
        """Check if the agent should update the hash."""
        if self.params.task_mutable_params.latest_metadata_hash is None:
            latest_hash = yield from self._get_latest_hash()
            if latest_hash is None:
                self.context.logger.warning(
                    "Could not get latest hash. Don't update the metadata."
                )
                return False
            self.params.task_mutable_params.latest_metadata_hash = latest_hash

        configured_hash = self.to_multihash(self.params.metadata_hash)
        latest_hash = self.params.task_mutable_params.latest_metadata_hash
        return configured_hash != latest_hash

    def get_mech_update_hash_tx(self) -> Generator:
        """Get the mech update hash tx."""
        should_update_hash = yield from self._should_update_hash()
        if not should_update_hash:
            return None

        metadata_str = self.to_multihash(self.params.metadata_hash)
        metadata = bytes.fromhex(metadata_str)
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=self.params.agent_registry_address,
            contract_id=str(AgentRegistryContract.contract_id),
            contract_callable="get_update_hash_tx_data",
            token_id=self.params.agent_id,
            metadata_hash=metadata,
        )
        if (
            contract_api_msg.performative != ContractApiMessage.Performative.STATE
        ):  # pragma: nocover
            self.context.logger.warning(
                f"get_mech_update_hash unsuccessful!: {contract_api_msg}"
            )
            return None

        data = cast(bytes, contract_api_msg.state.body["data"])
        return {
            "to": self.params.agent_registry_address,
            "value": ZERO_ETHER_VALUE,
            "data": data,
        }


class TransactionPreparationBehaviour(FundsSplittingBehaviour, HashUpdateBehaviour):
    """TransactionPreparationBehaviour"""

    matching_round: Type[AbstractRound] = TransactionPreparationRound

    def async_act(self) -> Generator:  # pylint: disable=R0914,R0915
        """Do the act, supporting asynchronous execution."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            payload_content = yield from self.get_payload_content()
            sender = self.context.agent_address
            payload = TransactionPayload(sender=sender, content=payload_content)
        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
        self.set_done()

    def get_payload_content(self) -> Generator[None, None, str]:
        """Prepare the transaction"""
        all_txs = []
        update_hash_tx = yield from self.get_mech_update_hash_tx()
        if update_hash_tx is not None:
            # in case of None, the agent should not update the hash
            # if this is caused by an error, the agent should still proceed with the rest
            # of the txs. The error will be logged.
            all_txs.append(update_hash_tx)

        split_profit_txs = yield from self.get_split_profit_txs()
        if split_profit_txs is not None:
            # in case of None, the agent should not update the hash
            # if this is caused by an error, the agent should still proceed with the rest
            # of the txs. The error will be logged.
            all_txs.extend(split_profit_txs)

        for task in self.synchronized_data.done_tasks:
            deliver_tx = yield from self._get_deliver_tx(task)
            if deliver_tx is None:
                # something went wrong, respond with ERROR payload for now
                # nothing should proceed if this happens
                return TransactionPreparationRound.ERROR_PAYLOAD
            all_txs.append(deliver_tx)
            response_tx = task.get("transaction", None)
            if response_tx is not None:
                all_txs.append(response_tx)

        multisend_tx_str = yield from self._to_multisend(all_txs)
        if multisend_tx_str is None:
            # something went wrong, respond with ERROR payload for now
            return TransactionPreparationRound.ERROR_PAYLOAD

        return multisend_tx_str

    def _to_multisend(
        self, transactions: List[Dict]
    ) -> Generator[None, None, Optional[str]]:
        """Transform payload to MultiSend."""
        multi_send_txs = []
        for transaction in transactions:
            transaction = {
                "operation": transaction.get("operation", MultiSendOperation.CALL),
                "to": transaction["to"],
                "value": transaction["value"],
                "data": transaction.get("data", b""),
            }
            multi_send_txs.append(transaction)

        response = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=self.params.multisend_address,
            contract_id=str(MultiSendContract.contract_id),
            contract_callable="get_tx_data",
            multi_send_txs=multi_send_txs,
        )
        if response.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.context.logger.error(
                f"Couldn't compile the multisend tx. "
                f"Expected performative {ContractApiMessage.Performative.RAW_TRANSACTION.value}, "  # type: ignore
                f"received {response.performative.value}."
            )
            return None

        # strip "0x" from the response
        multisend_data_str = cast(str, response.raw_transaction.body["data"])[2:]
        tx_data = bytes.fromhex(multisend_data_str)
        tx_hash = yield from self._get_safe_tx_hash(tx_data)
        if tx_hash is None:
            # something went wrong
            return None

        payload_data = hash_payload_to_hex(
            safe_tx_hash=tx_hash,
            ether_value=ZERO_ETHER_VALUE,
            safe_tx_gas=SAFE_GAS,
            operation=SafeOperation.DELEGATE_CALL.value,
            to_address=self.params.multisend_address,
            data=tx_data,
            gas_limit=self.params.manual_gas_limit,
        )
        return payload_data

    def _get_safe_tx_hash(self, data: bytes) -> Generator[None, None, Optional[str]]:
        """
        Prepares and returns the safe tx hash.

        This hash will be signed later by the agents, and submitted to the safe contract.
        Note that this is the transaction that the safe will execute, with the provided data.

        :param data: the safe tx data.
        :return: the tx hash
        """
        response = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=self.synchronized_data.safe_contract_address,
            contract_id=str(GnosisSafeContract.contract_id),
            contract_callable="get_raw_safe_transaction_hash",
            to_address=self.params.multisend_address,  # we send the tx to the multisend address
            value=ZERO_ETHER_VALUE,
            data=data,
            safe_tx_gas=SAFE_GAS,
            operation=SafeOperation.DELEGATE_CALL.value,
        )

        if response.performative != ContractApiMessage.Performative.STATE:
            self.context.logger.error(
                f"Couldn't get safe hash. "
                f"Expected response performative {ContractApiMessage.Performative.STATE.value}, "  # type: ignore
                f"received {response.performative.value}."
            )
            return None

        # strip "0x" from the response hash
        tx_hash = cast(str, response.state.body["tx_hash"])[2:]
        return tx_hash

    def _get_deliver_tx(
        self, task_data: Dict[str, Any]
    ) -> Generator[None, None, Optional[Dict]]:
        """Get the deliver tx."""
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=task_data["mech_address"],
            contract_id=str(AgentMechContract.contract_id),
            contract_callable="get_deliver_data",
            request_id=task_data["request_id"],
            data=task_data["task_result"],
        )
        if (
            contract_api_msg.performative != ContractApiMessage.Performative.STATE
        ):  # pragma: nocover
            self.context.logger.warning(
                f"get_deliver_data unsuccessful!: {contract_api_msg}"
            )
            return None

        data = cast(bytes, contract_api_msg.state.body["data"])
        return {
            "to": task_data["mech_address"],
            "value": ZERO_ETHER_VALUE,
            "data": data,
        }


class TaskSubmissionRoundBehaviour(AbstractRoundBehaviour):
    """TaskSubmissionRoundBehaviour"""

    initial_behaviour_cls = TaskPoolingBehaviour
    abci_app_cls = TaskSubmissionAbciApp
    behaviours: Set[Type[BaseBehaviour]] = {
        TaskPoolingBehaviour,  # type: ignore
        TransactionPreparationBehaviour,  # type: ignore
    }
