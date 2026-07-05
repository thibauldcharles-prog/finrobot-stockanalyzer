# Deploy FinRobot to a permanent public site (Streamlit Community Cloud)

This gives you a fixed URL like `https://finrobot-yourname.streamlit.app` that runs
24/7, independent of your PC. It's **free**. No git install needed — you can do it
entirely through the GitHub website.

> Heads-up: the app pulls live data from Yahoo Finance. On a shared cloud IP, Yahoo
> sometimes rate-limits bulk scans. The app already caches results and shows a friendly
> "rate limited, wait 10–15 min" message when that happens. Your **tunnel** link (the
> "FinRobot (Shareable Link)" desktop icon) uses your own IP and won't hit this, so keep
> that around for heavy scanning.

---

## Step 1 — Make a GitHub account (once)
Go to https://github.com and sign up (free). Verify your email.

## Step 2 — Create an empty repository
1. Click the **+** (top-right) → **New repository**.
2. Repository name: `finrobot-stockanalyzer`
3. Set it to **Public** (Streamlit's free tier needs public repos).
4. **Do not** add a README/.gitignore (we already have files).
5. Click **Create repository**.

## Step 3 — Upload the project files (no git required)
1. On the new repo page click **uploading an existing file** (the link in the
   "Quick setup" box), or go to **Add file → Upload files**.
2. Drag in these files/folders from
   `C:\Users\thiba\Documents\FinRobot_StockAnalyzer`:
   - `app.py`
   - `requirements.txt`
   - `.gitignore`
   - the `.streamlit` folder (contains `config.toml`)
   **Do NOT upload:** `cloudflared.exe`, any `_*.log`, `current_link.txt`,
   `launch*.ps1`, `__pycache__`. (The `.gitignore` lists these — they're local-only.)
3. Click **Commit changes**.

## Step 4 — Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and click **Sign in with GitHub** (authorize it).
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `yourname/finrobot-stockanalyzer`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. (Optional) Click **Advanced settings** and pick Python 3.11+.
5. Click **Deploy**. First build takes ~2–4 minutes while it installs the packages
   from `requirements.txt`.

## Step 5 — Done
You'll get a permanent URL like:

    https://finrobot-stockanalyzer-yourname.streamlit.app

Share it with anyone, open it on your phone — it stays up even when your PC is off.

### Updating the app later
Edit `app.py` (or upload a new version) in the GitHub repo → Streamlit Cloud
auto-redeploys within a minute.

---

## Alternative: deploy with the git command line
If you'd rather use git (you'd need to install it from https://git-scm.com/download/win):

```powershell
cd "C:\Users\thiba\Documents\FinRobot_StockAnalyzer"
git init
git add app.py requirements.txt .gitignore .streamlit
git commit -m "FinRobot Stock Analyzer"
git branch -M main
git remote add origin https://github.com/YOURNAME/finrobot-stockanalyzer.git
git push -u origin main
```

Then do Step 4 above.
