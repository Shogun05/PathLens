#!/bin/bash
# Quick API Test Script

API_BASE="http://localhost:8001"

echo "Testing PathLens Backend API"
echo "=============================="
echo ""

# Test root
echo "1. Testing root endpoint..."
curl -s "$API_BASE/" | python3 -m json.tool | head -5
echo ""

# Test health
echo "2. Testing health endpoint..."
curl -s "$API_BASE/health" | python3 -m json.tool
echo ""

# Test optimization status
echo "3. Testing optimization status..."
curl -s "$API_BASE/api/optimization/status" | python3 -m json.tool
echo ""

# Test nodes endpoint (will return empty if no data)
echo "4. Testing nodes endpoint (baseline)..."
curl -s "$API_BASE/api/nodes?type=baseline" 2>&1 | head -3
echo ""

# Test POIs endpoint
echo "5. Testing POIs endpoint..."
curl -s "$API_BASE/api/pois" | python3 -m json.tool | head -5
echo ""

echo "=============================="
echo "API Test Complete"
