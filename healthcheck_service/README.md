# Install requirements:

```pip3 install -r requirements.txt```


# Run the RPC monitoring script. 

The script requires some details to be passed as env variables. 

```commandline
RPC_ENDPOINT=""
MECH_CONTRACT_ADDRESS=""
SAFE_CONTRACT_ADDRESS=""
```

You can export the following vars by adding the following in your terminal : 

```
export RPC_ENDPOINT="https://broken-clean-dream.xdai.quiknode.pro/86a2bc89e63e3f84bbf55db55e81222f7685fb99/"
export MECH_CONTRACT_ADDRESS="0x45b73d649c7B982548D5A6dd3D35E1C5C48997d0"
export SAFE_CONTRACT_ADDRESS="0xf9507B46c41F90D577a28EB6B66C7C3f49b1Bad9"
```

Once you have the vars exproted, you can then run : 

```python run.py```

