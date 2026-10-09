// Runs the server's clue validator (functions/src/clueValidation.ts) over every
// game_modes_data/clues/batches/*.json file and merges the results into game_modes_data/clues/player_clues.json
// and game_modes_data/clues/review.csv (ranked by market value, with each player's difficulty).
//
//   node game_modes_data/clues/check_clues.js
//
// Build the functions first (cd ../footballquiz_firebase/functions && npm run build).
// Errors keep a player out of player_clues.json; warnings are listed only.

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "../..");
const CLUES = __dirname;
const {checkClueDoc} = require(path.resolve(
  ROOT, "../footballquiz_firebase/functions/lib/clueValidation.js"));

const facts = new Map(fs.readFileSync(path.join(CLUES, "facts.jsonl"), "utf8")
  .split("\n").filter(Boolean).map((l) => JSON.parse(l)).map((p) => [p.player_id, p]));

const batchDir = path.join(CLUES, "batches");
const docs = {};
const notInPool = [];
const report = [];
for (const file of fs.readdirSync(batchDir).filter((f) => f.endsWith(".json")).sort()) {
  for (const doc of JSON.parse(fs.readFileSync(path.join(batchDir, file), "utf8"))) {
    const p = facts.get(doc.player_id);
    if (!p) {  // no longer in the player pool (see transfer_history/build_dataset.py)
      notInPool.push(doc.name);
      continue;
    }
    const check = checkClueDoc(doc, p && {
      name: p.name, nativeName: p.native_name, currentClub: p.current_club,
    });
    if (check.errors.length || check.warnings.length) {
      report.push({file, player_id: doc.player_id, name: doc.name, ...check, clues: undefined});
    }
    if (!check.ok) continue;
    const {player_id: id, ...fields} = doc;
    docs[id] = {...fields, status: fields.status ?? "draft"};
  }
}

fs.writeFileSync(path.join(CLUES, "player_clues.json"), JSON.stringify(docs, null, 2) + "\n");
fs.writeFileSync(path.join(CLUES, "clue_report.json"), JSON.stringify(report, null, 2) + "\n");

const csvCell = (v) => /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v);
const columns = ["clue1_en", "clue2_en", "clue3_en", "clue1_ar", "clue2_ar", "clue3_ar", "notes", "status"];
const rows = Object.entries(docs)
  .map(([id, d]) => ({id, d, p: facts.get(id)}))
  .sort((a, b) => a.p.rank - b.p.rank)
  .map(({id, d, p}) => [p.rank, id, d.name, p.difficulty, ...columns.map((c) => d[c] ?? "")]);
fs.writeFileSync(path.join(CLUES, "review.csv"),
  [["rank", "player_id", "name", "difficulty", ...columns], ...rows]
    .map((r) => r.map(csvCell).join(",")).join("\n") + "\n");

const tiers = ["beginner", "intermediate", "expert"];
const written = Object.keys(docs).map((id) => facts.get(id).difficulty);
const total = [...facts.values()].map((p) => p.difficulty);
console.log(`${Object.keys(docs).length} valid, ${report.filter((r) => !r.ok).length} with errors, ` +
  `${report.filter((r) => r.ok).length} with warnings only`);
if (notInPool.length) {
  console.log(`  ${notInPool.length} clue sets left out, player not in the pool: ${notInPool.slice(0, 8).join(", ")}…`);
}
for (const t of tiers) {
  console.log(`  ${t}: ${written.filter((x) => x === t).length} / ${total.filter((x) => x === t).length}`);
}
for (const r of report) {
  console.log(`  ${r.ok ? "warn " : "ERROR"} ${r.player_id} ${r.name}: ${[...r.errors, ...r.warnings].join(", ")}`);
}
