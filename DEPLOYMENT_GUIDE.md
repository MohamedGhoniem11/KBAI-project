# Culinary RAG Deployment Guide

> Complete step-by-step to deploy on Streamlit Community Cloud

---

## Option Comparison (CV Impact)

| Option | Pros | Cons | CV Value |
|--------|------|------|----------|
| **HuggingFace Spaces** | Professional ML platform, free, easy Gradio/Streamlit | 50GB disk limit | ⭐⭐⭐⭐⭐ |
| **Streamlit Cloud** | Official Streamlit hosting, simple | 1GB RAM limit, may OOM | ⭐⭐⭐⭐ |
| **GitHub + Render** | Full control, free tier | Complex setup, no auto-deploy | ⭐⭐⭐ |

**Recommendation: HuggingFace Spaces** — it's where AI/ML projects live. Recruiters recognize it.

---

## PRE-DEPLOYMENT CHECKLIST

Before deploying, verify these locally:

- [ ] `streamlit run app.py` works on your machine
- [ ] API key is in `.env` (not committed)
- [ ] KB folder has 9 PDFs (296MB)
- [ ] Vectorstore exists (`data/vectorstore/`)
- [ ] `.env` has placeholder key committed (not real key)

---

## DEPLOYMENT STEPS

### Step 1: Create HuggingFace Account

1. Go to [huggingface.co](https://huggingface.co)
2. Sign up with GitHub (or email)
3. Verify email

### Step 2: Create New Space

1. Click **"New Space"** (top right)
2. Fill in:
   - **Name**: `culinary-rag-assistant`
   - **License**: `apache-2.0`
   - **SDK**: **Streamlit** (or Gradio — your choice)
   - **Hardware**: `small` (free, 16GB RAM)

### Step 3: Push to HF Space

```bash
cd /home/dingo/Desktop/KBAI/KBAI-project

# Add HF remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/culinary-rag-assistant

# Push (this sets up the space)
git push hf main
```

### Step 4: Add Secrets (API Key)

1. Go to your HF Space → Settings → Repository secrets
2. Add:
   - **Name**: `XAI_API_KEY`
   - **Value**: your actual Grok API key

### Step 5: Host KB Files

The KB (296MB) is too large for the repo. Choose one:

**Option A — HuggingFace Dataset (Recommended):**
```bash
# Install HF CLI
pip install huggingface_hub

# Create dataset repo
huggingface-cli login
huggingface-cli repo create culinary-rag-kb --type dataset

# Upload KB folder
cd KB
huggingface-cli upload-repo --repo-type dataset --folder . culinary-rag-kb
```

Then modify `rebuild_and_test.py` to download from HF dataset at startup.

**Option B — Keep Google Drive (Simpler):**
The README already has a Google Drive link. The app can download from there on first run.

### Step 6: Wait for Deployment

- HF builds the Docker image
- Installs requirements.txt
- Runs app.py
- Takes 3-10 minutes
- Check Logs tab if errors

---

## POST-DEPLOYMENT (CV Updates)

Once live, get your URL:

```
https://huggingface.co/spaces/YOUR_USERNAME/culinary-rag-assistant
```

### Update These Files:

1. **LinkedIn** → Featured → Add link
2. **GitHub README.md** → Add demo badge
3. **CV Projects.md** → Add live URL
4. **CV Enhancement.md** → Check off deployment

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM (out of memory) | KB too large — host on HF dataset, load on demand |
| API key not found | Add as HF Space secret |
| Build fails | Check Logs tab for dependency errors |
| Slow first load | Normal — downloads embeddings + KB |

---

## Expected Timeline

- Account setup: 10 min
- HF Space creation: 5 min
- Git push: 5 min
- Build + deploy: 5-15 min
- Debug (if needed): 10-30 min

**Total: ~1 hour**

---

## Next Steps After Deploy

1. Test the live app with 3 queries
2. Screenshot the UI for GitHub README
3. Add badge: `[![Demo](https://img.shields.io/badge/Demo-Live-blue)]`
4. Apply to jobs mentioning "RAG" or "LLM"
5. Add to CV: *"Deployed RAG system with 295MB culinary corpus on HuggingFace"*

---

## Files Referenced

- `app.py` — Streamlit entry point
- `rebuild_and_test.py` — Vector store rebuild
- `requirements.txt` — Dependencies
- `.env` — API key (never commit real key)
- `src/config.py` — Paths and thresholds

---

## Links

- HF Spaces: https://huggingface.co/new-space
- HF Datasets: https://huggingface.co/new-dataset
- Streamlit Cloud: https://streamlit.io/cloud
- xAI Console: https://console.x.ai/