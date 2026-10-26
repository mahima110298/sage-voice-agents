# SAGE human calibration

A page where two people check 120 benchmark turns:

- **Step A:** pick the gate the turn needs, then see the policy label and agree or disagree.
- **Step B:** say what the agent did and give a verdict (pass, fail, severe).

The page hides which setup produced each turn and what the automatic judge said.

Live page: https://mahima110298.github.io/sage-voice-agents/calibration/

## Connect the Google Sheet (one time)

1. Create a new Google Sheet.
2. Open **Extensions → Apps Script**. Replace the code with `apps_script/Code.gs` and save.
3. Click **Deploy → New deployment**. Pick **Web app** and set:
   - Execute as: **Me**
   - Who has access: **Anyone**
4. Click **Deploy** and allow access.
5. Copy the URL that ends in `/exec` and put it in `config.js`.

Until `config.js` has a URL, answers stay in each rater's browser. Raters can still save them from **Results → Download CSV**.

## Files

| File | What it is |
|---|---|
| `index.html` | The page |
| `config.js` | The Google Sheet web app URL |
| `data.json` | The 120 turns plus the gate rules |
| `audio/` | The user's audio for each turn |
| `apps_script/Code.gs` | The Google Sheet backend |
