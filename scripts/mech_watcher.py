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
import os
import time

import requests

from propel_client.propel import PropelClient
from propel_client.propel import CredentialStorage

agent_urls = [
    "https://bbc9b272e8601990.agent.propel.autonolas.tech/healthcheck",
    "https://683dcd5c4524ed21.agent.propel.autonolas.tech/healthcheck",
    "https://56523691161fd4ed.agent.propel.autonolas.tech/healthcheck",
    "https://26c5a45d2f1d8b31.agent.propel.autonolas.tech/healthcheck",
]
keys = [
    169,
    170,
    171,
    172
]

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

def get_agents():
    """Get the agents from the key ids"""
    propel_client = get_propel()
    agent_ids = [agent["id"] for agent in propel_client.agents_list() if agent["key"] in keys]
    return agent_ids


def restart_service():
    """Restarting the service"""
    propel_client = get_propel()
    agent_ids = get_agents()
    for agent_id in agent_ids:
        print(f"Restating {agent_id=}")
        propel_client.agents_restart(agent_id)


while True:
    try:
        time.sleep(15)

        agents_to_data = {}
        for agent_url in agent_urls:
            res = requests.get(agent_url)
            if res.status_code != 200:
                continue

            data = res.json()
            if data["last_successful_executed_task"]["timestamp"] < time.time() - 600 and data["queue_size"] > 25:
                print(f"Restarting service, agent={agent_url} is not executing.")
                restart_service()
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
            continue

        # at least one of the agents is in registration
        if average_period_count > 5:
            # period count average is more than 5, meaning registration should've finished, hence one agent is stuck
            restart_service()
            time.sleep(600)
            continue

    except Exception as e:
        print(e)