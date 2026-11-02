# NFL Betting Board 2026

Runs the NFL Betting Checklist (Checks 1-5, pass rules, totals, Check 8 angles) on every remaining 2026 game.
Data: nflverse schedule, lines and play-by-play. Refreshes automatically every day at 7am Central via GitHub Actions
(Actions tab > Refresh NFL data > Run workflow to refresh by hand).

## Sportsbook lines (The Odds API)
1. Get a key at https://the-odds-api.com.
2. Repo Settings > Secrets and variables > Actions > New repository secret: name `ODDS_API_KEY`, value = your key.
3. The daily 7am CT run then logs Caesars, DraftKings, FanDuel and BetMGM lines (opening vs current, total moves)
   and Caesars 1Q/1H lines for games in the next 30 hours.
4. Paid plan only: Actions > Refresh NFL data > Run workflow > tick "backfill" to pull opening/closing lines and
   Caesars 1Q/1H closing lines for every played game. That fills the Check 6 and 1Q/1H tests on Season results.
