// Uploads the approved clue sets in game_modes_data/clues/player_clues.json to Firestore
// player_clues/{playerId}, writing only the documents that changed.
//
//   node game_modes_data/clues/upload_clues.js --project dev            # dry run: show the diff
//   node game_modes_data/clues/upload_clues.js --project dev --apply
//   node game_modes_data/clues/upload_clues.js --project prod --apply
//
// Reads the collection once, then writes each new or changed document once
// (updated_at is set here so the onClueWritten trigger has nothing to add).
// The server republishes cache/clues_v1.json within 30 minutes of a change.
// Uses the service-account key for the project found in the repo root.

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "../..");
const admin = require(path.resolve(
  ROOT, "../footballquiz_firebase/functions/node_modules/firebase-admin"));

const PROJECTS = {dev: "football-quiz-32eb9", prod: "mercato-6e710"};
const FIELDS = ["clue1_en", "clue1_ar", "clue2_en", "clue2_ar", "clue3_en", "clue3_ar",
  "status", "notes"];

const args = process.argv.slice(2);
const apply = args.includes("--apply");
const projectArg = args[args.indexOf("--project") + 1];
const projectId = PROJECTS[projectArg];
if (!projectId) {
  console.error("Usage: node game_modes_data/clues/upload_clues.js --project dev|prod [--apply]");
  process.exit(1);
}

const keyFile = fs.readdirSync(ROOT)
  .find((f) => f.startsWith(`${projectId}-firebase-adminsdk`) && f.endsWith(".json"));
if (!keyFile) {
  console.error(`No service-account key for ${projectId} in ${ROOT}`);
  process.exit(1);
}
const key = JSON.parse(fs.readFileSync(path.join(ROOT, keyFile), "utf8"));
if (key.project_id !== projectId) {
  console.error(`Key ${keyFile} is for ${key.project_id}, not ${projectId}`);
  process.exit(1);
}
admin.initializeApp({credential: admin.credential.cert(key), projectId});
const db = admin.firestore();

/** Sets LAST_UPDATED.json[section] (see sources/last_updated.py). */
function recordLastUpdated(section, facts) {
  const file = path.join(__dirname, "LAST_UPDATED.json");
  const data = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : {};
  const keys = section.split(".");
  let target = data;
  for (const key of keys.slice(0, -1)) target = target[key] ??= {};
  target[keys.at(-1)] = {date: new Date().toLocaleDateString("en-CA"), ...facts};
  fs.writeFileSync(file, JSON.stringify(data, null, 2) + "\n");
}

/** The fields a clue document holds in Firestore (empty notes left out). */
function toDoc(source) {
  const doc = {};
  for (const field of FIELDS) {
    if (source[field] != null && source[field] !== "") doc[field] = source[field];
  }
  return doc;
}

(async () => {
  const local = JSON.parse(fs.readFileSync(path.join(ROOT, "game_modes_data/clues/player_clues.json"), "utf8"));
  const wanted = Object.entries(local).filter(([, doc]) => doc.status === "approved");

  const snap = await db.collection("player_clues").get();
  const existing = new Map(snap.docs.map((d) => [d.id, d.data()]));

  const changes = [];
  let replacedMock = 0;
  for (const [id, source] of wanted) {
    const doc = toDoc(source);
    const current = existing.get(id);
    const same = current && current.mock !== true &&
      FIELDS.every((f) => (current[f] ?? "") === (doc[f] ?? ""));
    if (same) continue;
    if (current?.mock === true) replacedMock++;
    changes.push([id, doc]);
  }
  const wantedIds = new Set(wanted.map(([id]) => id));
  const untouched = snap.docs.filter((d) => !wantedIds.has(d.id));

  console.log(`${projectId}: ${snap.size} docs in player_clues, ${wanted.length} approved locally`);
  console.log(`  to write: ${changes.length} (${replacedMock} replace a mock doc)`);
  console.log(`  already up to date: ${wanted.length - changes.length}`);
  console.log(`  left alone (not in the local set): ${untouched.length}` +
    (untouched.length ? ` — ${untouched.filter((d) => d.get("mock") === true).length} mock` : ""));

  if (!apply) {
    console.log("Dry run. Add --apply to write.");
    return;
  }
  const writer = db.bulkWriter();
  let failed = 0;
  writer.onWriteError((error) => {
    if (error.failedAttempts < 5) return true;
    failed++;
    console.error(`  failed ${error.documentRef.id}: ${error.message}`);
    return false;
  });
  for (const [id, doc] of changes) {
    writer.set(db.collection("player_clues").doc(id), {
      ...doc,
      updated_at: admin.firestore.FieldValue.serverTimestamp(),
    });
  }
  await writer.close();
  console.log(`Wrote ${changes.length - failed} documents` + (failed ? `, ${failed} failed` : ""));
  if (failed) process.exit(1);
  recordLastUpdated(`synced.${projectId}`, {approved_clue_sets: wanted.length, written: changes.length});
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
