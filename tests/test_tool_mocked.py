"""
Unit & Integration Tests for Olas Mech Tool: polymarket_intelligence.py
ALL TESTS ARE 100% MOCKED (NO LIVE PAYMENTS OR REAL BLOCKCHAIN TRANSACTIONS EXECUTED).
"""

import json
import unittest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../tool')))

from polymarket_intelligence import run

class TestPolymarketIntelligenceOlasTool(unittest.TestCase):

    @patch('requests.post')
    def test_http_402_challenge_mocked(self, mock_post):
        """MOCKED TEST: Verify tool handles HTTP 402 challenge correctly."""
        mock_resp = MagicMock()
        mock_resp.status_code = 402
        mock_resp.json.return_value = {
            "error": "Payment Required",
            "status": 402,
            "accepts": [
                {
                  "scheme": "exact",
                  "network": "eip155:8453",
                  "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                  "amount": "0.05",
                  "payTo": "0x43f9a721B59C247B6258e72A1Bf5A384b64F8A38"
                }
            ]
        }
        mock_post.return_value = mock_resp

        output_str = run({"market_id": "691547"})
        data = json.loads(output_str)

        self.assertEqual(data["status"], "PAYMENT_REQUIRED")
        self.assertEqual(data["http_status"], 402)
        self.assertEqual(data["x402_challenge"]["price_usdc"], "0.05")
        self.assertEqual(data["x402_challenge"]["recipient"], "0x43f9a721B59C247B6258e72A1Bf5A384b64F8A38")
        print("[PASS] MOCKED TEST 1: HTTP 402 Challenge correctly parsed")

    @patch('requests.post')
    def test_http_200_success_mocked(self, mock_post):
        """MOCKED TEST: Verify tool returns live payload when payment signature attached."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "data": {
                "market": { "id": "691547", "question": "Kraken IPO by December 31, 2026?" },
                "outcomes": [{ "name": "Yes", "price_usd": 0.12 }, { "name": "No", "price_usd": 0.88 }]
            },
            "payment": {
                "amount": "0.05",
                "currency": "USDC",
                "network": "BASE",
                "tx_hash": "0x_mocked_tx_hash_for_unit_testing"
            }
        }
        mock_post.return_value = mock_resp

        output_str = run({
            "market_id": "691547",
            "payment_signature": "0x_mocked_tx_hash_for_unit_testing"
        })
        data = json.loads(output_str)

        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["http_status"], 200)
        self.assertEqual(data["result"]["market"]["id"], "691547")
        self.assertEqual(data["x402_settlement"]["tx_hash"], "0x_mocked_tx_hash_for_unit_testing")
        print("[PASS] MOCKED TEST 2: Paid HTTP 200 payload correctly returned")

if __name__ == '__main__':
    unittest.main()
