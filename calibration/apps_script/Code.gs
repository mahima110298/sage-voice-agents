// Google Apps Script backend for the SAGE calibration page.
// Paste into the Sheet's Extensions > Apps Script, then Deploy > New deployment >
// Web app (Execute as: Me, Who has access: Anyone). Put the /exec URL in ../config.js.

const SHEET = 'ratings';
const COLS = ['key', 'rater', 'tid', 'gate_first', 'gate_note', 'revealed_at', 'agree', 'disagree_reason',
  'produced', 'readback', 'passcode', 'wrong_target', 'verdict', 'remarks', 'flag', 'updated_at'];

function sheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(SHEET);
  if (!sh) {
    sh = ss.insertSheet(SHEET);
    sh.appendRow(COLS);
    sh.setFrozenRows(1);
  }
  return sh;
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function doGet() {
  const values = sheet_().getDataRange().getValues();
  const head = values.shift();
  const rows = values.map(v => {
    const o = {};
    head.forEach((h, i) => { o[h] = v[i] === '' ? null : v[i]; });
    if (o.agree !== null) o.agree = o.agree === true || o.agree === 'true';
    o.flag = o.flag === true || o.flag === 'true';
    return o;
  });
  return json_({ ok: true, rows: rows });
}

function doPost(e) {
  let r;
  try { r = JSON.parse(e.postData.contents); } catch (err) { return json_({ ok: false, error: 'bad_json' }); }
  if (!/^T\d{3}$/.test(String(r.tid || '')) || !r.rater || String(r.rater).length > 60) {
    return json_({ ok: false, error: 'bad_row' });
  }
  const slug = String(r.rater).trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  r.key = slug + '__' + r.tid;
  const row = COLS.map(c => {
    const v = r[c];
    if (v === undefined || v === null) return '';
    if (typeof v !== 'string') return v;
    const t = v.slice(0, 2000);
    return /^[=+\-@]/.test(t) ? "'" + t : t;  // keep text from running as a formula
  });
  const lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    const sh = sheet_();
    const keys = sh.getRange(1, 1, sh.getLastRow(), 1).getValues().map(x => x[0]);
    const at = keys.indexOf(r.key);
    if (at > 0) sh.getRange(at + 1, 1, 1, COLS.length).setValues([row]);
    else sh.appendRow(row);
  } finally {
    lock.releaseLock();
  }
  return json_({ ok: true, key: r.key });
}
