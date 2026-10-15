"""
Olas Mech Tool Adapter: Polymarket x402 Intelligence Service
Exposes our production x402 Polymarket Intelligence API to Olas Mech Worker Nodes and Pearl/Polystrat agents.

Architecture:
Pearl Agent ➔ Olas Mech Request ('polymarket_intelligence') ➔ Mech Worker Node ➔ HTTP 402 Challenge ➔ Base Mainnet $0.05 USDC Payment ➔ Production API Data ➔ Mech On-Chain Response
"""

import json
import os
import requests
from typing import Any, Dict, Optional

GATEWAY_URL = os.getenv("GATEWAY_BASE_URL", "https://agent-payment-gateway.vercel.app")
MERCHANT_WALLET = "0x43f9a721B59C247B6258e72A1Bf5A384b64F8A38"
PRICE_USDC = "0.05"
BASE_NETWORK = "eip155:8453"

def run(kwargs: Dict[str, Any]) -> str:
    """
    Main entrypoint for Olas Mech Worker execution.
    kwargs contains 'prompt', 'market_id', and optional 'payment_signature'.
    """
    market_id = kwargs.get("market_id") or "691547"
    payment_sig = kwargs.get("payment_signature") or os.getenv("X402_PAYMENT_SIGNATURE")

    endpoint = f"{GATEWAY_URL}/v1/prediction-market-analysis"
    payload = {
        "market_id": market_id,
        "analysis_type": kwargs.get("analysis_type", "market_intelligence")
    }

    headers = {
        "Content-Type": "application/json"
    }

    if payment_sig:
        headers["PAYMENT-SIGNATURE"] = payment_sig

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
        
        # HTTP 402 Payment Required Handling
        if response.status_code == 402:
            challenge_data = response.json()
            accepts = challenge_data.get("accepts", [{}])[0]
            
            return json.dumps({
                "status": "PAYMENT_REQUIRED",
                "http_status": 402,
                "message": "x402 Payment Authorization required to unlock live Polymarket analysis",
                "x402_challenge": {
                    "price_usdc": accepts.get("amount", PRICE_USDC),
                    "network": accepts.get("network", BASE_NETWORK),
                    "asset": accepts.get("asset", "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"),
                    "recipient": accepts.get("payTo", MERCHANT_WALLET)
                }
            })

        response.raise_for_status()
        api_data = response.json()

        return json.dumps({
            "status": "SUCCESS",
            "http_status": 200,
            "result": api_data.get("data", {}),
            "x402_settlement": api_data.get("payment", {
                "amount": PRICE_USDC,
                "currency": "USDC",
                "network": "BASE",
                "recipient": MERCHANT_WALLET
            })
        })

    except Exception as e:
        return json.dumps({
            "status": "ERROR",
            "error": str(e)
        })

if __name__ == "__main__":
    test_res = run({"market_id": "691547"})
    print(test_res)
