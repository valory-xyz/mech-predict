# Install requirements:

```pip3 install -r requirements.txt```


# Run the RPC monitoring script. 

The script requires some details to be passed as env variables. 

Create a .env file and export the following variables.

```
RPC_ENDPOINT=""
MECH_CONTRACT_ADDRESS=""
SAFE_CONTRACT_ADDRESS=""
```

Once you have the vars exproted, you can then run : 

```python3 rpc.py```


# Run the healthcheck monitoring script.

The script requires some details to be passed as env variables.

Create a .env file and export the following variables.

```
RPC_ENDPOINT=""
MECH_CONTRACT_ADDRESS=""
SAFE_CONTRACT_ADDRESS=""
SLACK_WEBHOOK_URL=""
```

## How to setup the slack webhook url

1. Setup Slack Channel
    - Open Slack and navigate to your workspace.
    - Click on the `+` (Add a Channel) button in the sidebar.
    - Select `Create a New Channel` from the options.
    - Enter a name for your channel (e.g., #healthcheck-updates).
    - Click Create to finalize the channel setup.

2. Create Slack App
    - Navigate to [slack apps](https://api.slack.com/apps)
    - Create an app from scratch and name it (e.g., healtcheck-bot)
    - Choose a Slack workspace

3. Configure Slack App 
    - Inside app setting, navigate to `Incoming Webhooks`.
    - Toggle the switch to `Activate Incoming Webhooks`.
    - Click `Add New Webhook to Workspace`.
    - Select the create slack channel in the previous step

4. Generate Webhook URL
    - Click `Add New Webhook to Workspace`
    - Select the newly created slack channel 
    - Click on Allow
    - Copy the generated `Webhook URL`

Once you have the vars exproted, you can then run :

`python3 healthcheck.py`
