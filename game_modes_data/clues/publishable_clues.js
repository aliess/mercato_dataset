// Picks the clue sets that go into cache/clues_v1.json and prints them as JSON.
// Called by publish_files.py; touches no network and writes no file.
//
//   node game_modes_data/clues/publishable_clues.js < players.json
//
// stdin:  [{id, player_name, name_in_home_country, current_club_name, market_value}, …]
//         (the players of the game data file being published)
// stdout: {clues: {playerId: [[en, ar], [en, ar], [en, ar]]}, by_difficulty, problems, skipped}
//
// Same rules the server used when it published clues: only status "approved", the player must
// be in the game data, and the set must pass checkClueDoc. Uses the functions' compiled
// validator (cd ../footballquiz_firebase/functions && npm run build).

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "../..");
const lib = path.resolve(ROOT, "../footballquiz_firebase/functions/lib/clueValidation.js");
if (!fs.existsSync(lib)) {
  console.error(`Missing ${lib}\nBuild the functions: cd ../footballquiz_firebase/functions && npm install && npm run build`);
  process.exit(1);
}
const {checkClueDoc, difficultyForValue} = require(lib);

const players = new Map(JSON.parse(fs.readFileSync(0, "utf8")).map((p) => [String(p.id), p]));
const local = JSON.parse(fs.readFileSync(path.join(__dirname, "player_clues.json"), "utf8"));

const clues = {};
const byDifficulty = {beginner: 0, intermediate: 0, expert: 0};
const problems = [];
const skipped = {not_approved: 0, not_in_game_data: 0, failed_validation: 0};

for (const [id, doc] of Object.entries(local)) {
  if (doc.status !== "approved") {
    skipped.not_approved++;
    continue;
  }
  const player = players.get(id);
  const check = checkClueDoc(doc, player ? {
    name: player.player_name ?? "",
    nativeName: player.name_in_home_country,
    currentClub: player.current_club_name,
  } : undefined);
  if (check.errors.length || check.warnings.length) {
    problems.push({player_id: id, name: doc.name, published: check.ok,
      errors: check.errors, warnings: check.warnings});
  }
  if (!player) {
    skipped.not_in_game_data++;
    continue;
  }
  if (!check.ok) {
    skipped.failed_validation++;
    continue;
  }
  clues[id] = check.clues;
  byDifficulty[difficultyForValue(player.market_value)]++;
}

process.stdout.write(JSON.stringify({clues, by_difficulty: byDifficulty, problems, skipped}));
