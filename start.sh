#!/bin/bash
# ============================================================
# AABS Control Tower v8.3 - Startup Script
# ============================================================
# This script handles everything:
# - Creates virtual environment if needed
# - Installs dependencies
# - Launches the application
# ============================================================

echo "============================================================"
echo "AABS Control Tower v8.3"
echo "Enterprise Decision Intelligence Platform"
echo "============================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "${YELLOW}[1/4] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Create virtual environment if needed
echo ""
echo -e "${YELLOW}[2/4] Setting up virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null

# Install dependencies
echo ""
echo -e "${YELLOW}[3/4] Installing dependencies...${NC}"
python3 -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ Error installing dependencies${NC}"
    echo "Trying individual packages..."
    pip install streamlit pandas openpyxl plotly numpy scipy
fi

# Launch application
echo ""
echo -e "${YELLOW}[4/4] Launching AABS Control Tower...${NC}"
echo ""
echo "============================================================"
echo -e "${GREEN}Starting server at http://localhost:8501${NC}"
echo "Press Ctrl+C to stop"
echo "============================================================"
echo ""

streamlit run app.py --server.headless true --browser.gatherUsageStats false
