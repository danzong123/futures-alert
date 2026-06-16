# Futures Alert - Cloud Deployment

Deploy the monitoring backend to GitHub Actions so push notifications work
even when your computer is off.

## How it works

The system runs `python main.py --now` every 5 minutes during Chinese futures
trading hours via GitHub Actions scheduled workflows. When signals are detected,
they are pushed to your WeChat via PushPlus.

## Deployment Steps

### 1. Create a GitHub repository

Go to https://github.com/new and create a new repository (public or private).

### 2. Upload the code

Upload all files in this directory to your new repository using the
GitHub web interface (Add file -> Upload files), or use git:

```bash
# If you have git installed:
cd futures-alert-cloud
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

### 3. Add your PushPlus token as a secret

1. Go to your repo on GitHub: Settings -> Secrets and variables -> Actions
2. Click "New repository secret"
3. Name: `PUSHPLUS_TOKEN`
4. Value: your PushPlus token (`f7b736e4523149149562f943c261dbe8`)
5. Click "Add secret"

### 4. Enable GitHub Actions

The workflow file is already at `.github/workflows/monitor.yml`.
GitHub Actions will automatically start on the next scheduled run.

You can also manually trigger a test run:
1. Go to your repo -> Actions -> "Futures Alert Monitor"
2. Click "Run workflow" -> "Run workflow"

### 5. Verify

Check the workflow run logs for any errors. If successful, you will receive
push notifications on your WeChat when signals are detected.

## Trading Hours (handled by the workflow schedule)

- Day session: 9:00-15:00 CST, Monday-Friday (cron: */5 1-7 * * 1-5 UTC)
- Night session: 21:00-02:30 CST, Monday-Friday (cron: */5 13-18 * * 1-5 UTC)

## Local files (not uploaded)

- `config.yaml` - contains your PushPlus token (added to .gitignore)
- `outputs/` - local logs and database (added to .gitignore)
