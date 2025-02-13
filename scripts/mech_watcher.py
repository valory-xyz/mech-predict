#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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
import logging
import os
import time

import requests

from propel_client.propel import PropelClient
from propel_client.propel import CredentialStorage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

key_to_agent_url_mech_predict = {
    169: "https://bbc9b272e8601990.agent.propel.autonolas.tech/healthcheck",
    170: "https://56523691161fd4ed.agent.propel.autonolas.tech/healthcheck",
    171: "https://683dcd5c4524ed21.agent.propel.autonolas.tech/healthcheck",
    172: "https://26c5a45d2f1d8b31.agent.propel.autonolas.tech/healthcheck",
}
key_to_agent_url_mech_marketplace = {
    305: "https://8aac3a49d4ee162a.agent.propel.autonolas.tech/healthcheck",
    319: "https://3f937d50881d3919.agent.propel.autonolas.tech/healthcheck",
    320: "https://0ad6d33fa06f4fcb.agent.propel.autonolas.tech/healthcheck",
    321: "https://bbbc5776f825edf3.agent.propel.autonolas.tech/healthcheck",
}

services = {
    "mech_predict": key_to_agent_url_mech_predict,
    "mech_marketplace": key_to_agent_url_mech_marketplace,
}

username = os.getenv("PROPEL_USERNAME")
password = os.getenv("PROPEL_PASSWORD")

if not username or not password:
    raise ValueError("PROPEL_USERNAME and PROPEL_PASSWORD must be set")

max_retries = 100

def get_propel(retries = 0):
    """Get the propel client"""
    try:
        credential_storage = CredentialStorage()
        propel_client: PropelClient = PropelClient("https://app.propel.valory.xyz", credential_storage, 3, 1, 30 )
        propel_client.login(username, password)
        return propel_client
    except Exception as e:
        print(e)
        if retries >= max_retries:
            raise e
        return get_propel(retries + 1)

def get_agents(service_id: str):
    """Get the agents from the key ids"""
    propel_client = get_propel()
    keys = list(services[service_id].keys())
    agent_ids = [agent["id"] for agent in propel_client.agents_list() if agent["key"] in keys]
    return agent_ids


def get_agent_id(key: int):
    """Get the agent id from the key"""
    propel_client = get_propel()
    agent_id = [agent["id"] for agent in propel_client.agents_list() if agent["key"] == key]
    return agent_id[0]


def restart_service(service_id: str):
    """Restarting the service"""
    print(f"Restarting {service_id=}")
    propel_client = get_propel()
    agent_ids = get_agents(service_id)
    for agent_id in agent_ids:
        print(f"Restating {agent_id=}")
        propel_client.agents_restart(agent_id)


while True:
    try:
        time.sleep(15)
        for service_id, key_to_agent_urls in services.items():
            agent_urls = key_to_agent_urls.values()
            agent_url_to_key = {url: key for key, url in key_to_agent_urls.items()}
            agents_to_data = {}
            for agent_url in agent_urls:
                res = requests.get(agent_url)
                if res.status_code != 200:
                    continue

                data = res.json()
                if data is None:
                    continue
                if (
                    data.get("last_successful_executed_task", {}).get("timestamp", float("inf")) < time.time() - 600
                    and data.get("queue_size", 0) > 25
                ):
                    agent_id = get_agent_id(agent_url_to_key[agent_url])
                    logger.warning(
                        f"Restarting service, agent={agent_id} with healthcheck={agent_url} is not executing task. "
                        f"This is likely related to issues with some external APIs the agent is using. "
                        f"Please check that the RPC is working as expected, that means that its returning a response in a timely manner. "
                        f"Make sure the IPFS is working as expected, that means that its returning a response in a timely manner. "
                        f"Make sure there are no issues with APIs used for tasks, example: OpenAI, Claude, Google Search, Serper, etc."
                    )
                    restart_service(service_id)
                    time.sleep(600)

                agents_to_data[agent_url] = data

            agents_in_registration = sum([
                data["current_round"] == "registration_startup_round"
                for data in agents_to_data.values()
            ])
            average_period_count = sum([data["period"] for data in agents_to_data.values()]) / len(agents_to_data)

            # check if any agent's current round is registration_startup_round
            if agents_in_registration == 0 or agents_in_registration == len(agents_to_data):
                # all agents are in registration_startup_round, skip
                logger.info("All agents operating normally.")
                continue

            # at least one of the agents is in registration
            if average_period_count > 5:
                # period count average is more than 5, meaning registration should've finished, hence one agent is stuck
                agent_id_to_data = {
                    get_agent_id(agent_url_to_key[agent_url]): data
                    for agent_url, data in agents_to_data.items()
                }
                logger.warning(
                    f"One of the agents has crashed, and is stuck in registration_startup_round. "
                    f"Restarting all agents s.t. the service can continue working normally. "
                    f"Agent data: {agent_id_to_data}"
                )
                restart_service(service_id)
                time.sleep(600)
                continue

    except Exception as e:
        logger.exception(e)