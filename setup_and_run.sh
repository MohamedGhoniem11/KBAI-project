#!/bin/bash
# =============================================================================
# Culinary RAG - Local Setup & Test Script
# Run this BEFORE deploying to ensure everything works
# =============================================================================

set -e

echo "=========================================="
echo " Culinary RAG - Local Setup & Test"
echo "=========================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check Python version
echo -e "\n${YELLOW}[1/6] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo -e "${RED}❌ Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}"

# Step 2: Check API key
echo -e "\n${YELLOW}[2/6] Checking API key...${NC}"
if [ -f .env ]; then
    source .env
    if [ -z "$XAI_API_KEY" ] || [ "$XAI_API_KEY" = "your_grok_api_key_here" ]; then
        echo -e "${RED}❌ API key not set in .env${NC}"
        echo "   Edit .env and replace 'your_grok_api_key_here' with your actual key"
        exit 1
    fi
    echo -e "${GREEN}✅ API key found in .env${NC}"
else
    echo -e "${RED}❌ .env file not found${NC}"
    echo "   Create .env with:"
    echo "   LLM_PROVIDER=xai"
    echo "   XAI_API_KEY=your_grok_api_key_here"
    echo "   LLM_MODEL=grok-2-1212"
    exit 1
fi

# Step 3: Check KB folder
echo -e "\n${YELLOW}[3/6] Checking KB folder...${NC}"
KB_COUNT=$(find KB/ -type f \( -name "*.pdf" -o -name "*.docx" \) 2>/dev/null | wc -l)
KB_SIZE=$(du -sh KB/ 2>/dev/null | awk '{print $1}')
if [ "$KB_COUNT" -lt 5 ]; then
    echo -e "${RED}❌ KB folder missing or empty${NC}"
    echo "   Download KB.zip from:"
    echo "   https://drive.google.com/uc?export=download&id=1RskXkZXqQiszdQ8QkEYySKlgToQPBZO4"
    echo "   Then unzip: unzip KB.zip -d KB/"
    exit 1
fi
echo -e "${GREEN}✅ KB folder: $KB_COUNT files ($KB_SIZE)${NC}"

# Step 4: Check/create venv
echo -e "\n${YELLOW}[4/6] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install dependencies
echo "Installing dependencies (this may take 5-10 minutes)..."
pip install -r requirements.txt -q

echo -e "${GREEN}✅ Dependencies installed${NC}"

# Step 5: Check vectorstore
echo -e "\n${YELLOW}[5/6] Checking vector store...${NC}"
if [ -d "data/vectorstore" ] && [ -f "data/vectorstore/index.faiss" ]; then
    VS_SIZE=$(du -sh data/vectorstore/ | awk '{print $1}')
    echo -e "${GREEN}✅ Vector store found ($VS_SIZE)${NC}"
    echo -e "${YELLOW}   Run 'python rebuild_and_test.py' to rebuild from scratch${NC}"
else
    echo -e "${YELLOW}⚠️ Vector store not found${NC}"
    echo "   Building vector store (this may take 10-20 minutes)..."
    python rebuild_and_test.py
fi

# Step 6: Test Streamlit
echo -e "\n${YELLOW}[6/6] Testing Streamlit app...${NC}"
echo -e "${GREEN}✅ Starting Streamlit...${NC}"
echo ""
echo "Open your browser to: http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""
streamlit run app.py --server.headless true