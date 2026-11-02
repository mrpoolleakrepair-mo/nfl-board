"""NFL Betting Checklist auto-scorer. Usage: python3 -I model.py <datadir> <outdir>"""
import sys, json, numpy as np, pandas as pd

D, OUT = sys.argv[1], sys.argv[2]
SEASON = 2026
games_all = pd.read_csv(f"{D}/games.csv")
G = games_all[(games_all.season == SEASON) & (games_all.game_type == "REG")].copy()
P = pd.read_parquet(f"{D}/pbp_2026.parquet")
P = P[P.season_type == "REG"]


def team_stats(pbp, sched):
    """Season metrics per team from plays/games completed in pbp/sched."""
    done = sched[sched.result.notna()]
    teams = sorted(set(sched.home_team) | set(sched.away_team))
    gp = pd.Series({t: ((done.home_team == t) | (done.away_team == t)).sum() for t in teams})
    db = pbp[pbp.qb_dropback == 1]
    ru = pbp[(pbp.rush_attempt == 1) & (pbp.qb_kneel != 1)]
    s = pd.DataFrame(index=teams)
    s["gp"] = gp
    s["def_sacks_pg"] = db.groupby("defteam").sack.sum() / gp
    s["def_sack_rate"] = db.groupby("defteam").sack.mean()
    s["def_pressure_rate"] = db.groupby("defteam").apply(lambda x: ((x.sack == 1) | (x.qb_hit == 1)).mean(), include_groups=False)
    s["off_sacks_allowed_pg"] = db.groupby("posteam").sack.sum() / gp
    s["off_sack_rate"] = db.groupby("posteam").sack.mean()
    s["off_ypc"] = ru.groupby("posteam").yards_gained.mean()
    s["def_ypc"] = ru.groupby("defteam").yards_gained.mean()
    s["off_rush_ypg"] = ru.groupby("posteam").yards_gained.sum() / gp
    # last 3 games rushing
    l3 = {}
    for t in teams:
        gids = sorted(ru[ru.posteam == t].game_id.unique())[-3:]
        x = ru[(ru.posteam == t) & ru.game_id.isin(gids)]
        l3[t] = x.yards_gained.mean() if len(x) else np.nan
    s["off_ypc_l3"] = pd.Series(l3)
    to = pbp[(pbp.interception == 1) | (pbp.fumble_lost == 1)]
    s["giveaways_pg"] = (to.groupby("posteam").size() / gp).fillna(0)
    s["takeaways_pg"] = (to.groupby("defteam").size() / gp).fillna(0)
    s["int_rate"] = db.groupby("posteam").interception.mean()
    # QB mobility: rush yds/game by players who also threw passes for that team
    passers = pbp[pbp.passer_player_id.notna()].groupby("posteam").passer_player_id.unique()
    qbr = {}
    for t in teams:
        ids = set(passers.get(t, []))
        x = ru[(ru.posteam == t) & ru.rusher_player_id.isin(ids)]
        qbr[t] = x.yards_gained.sum() / gp[t] if gp[t] else 0
    s["qb_rush_ypg"] = pd.Series(qbr)
    # red zone via drives
    dr = pbp[pbp.posteam.notna() & pbp.fixed_drive.notna()].groupby(["game_id", "posteam", "fixed_drive"]).agg(
        defteam=("defteam", "first"), minyd=("yardline_100", "min"), res=("fixed_drive_result", "first")).reset_index()
    rz = dr[dr.minyd <= 20]
    s["off_rz_trips_pg"] = rz.groupby("posteam").size() / gp
    s["off_rz_td_pct"] = rz.groupby("posteam").res.apply(lambda r: (r == "Touchdown").mean())
    s["def_rz_trips_pg"] = rz.groupby("defteam").size() / gp
    s["def_rz_td_pct"] = rz.groupby("defteam").res.apply(lambda r: (r == "Touchdown").mean())
    td3 = pbp[(pbp.third_down_converted == 1) | (pbp.third_down_failed == 1)]
    s["def_3rd_pct"] = td3.groupby("defteam").third_down_converted.mean()
    s["off_3rd_pct"] = td3.groupby("posteam").third_down_converted.mean()
    # points & records from schedule
    rec = {}
    for t in teams:
        h = done[done.home_team == t]; a = done[done.away_team == t]
        pf = h.home_score.sum() + a.away_score.sum(); pa = h.away_score.sum() + a.home_score.sum()
        hw = (h.result > 0).sum(); hl = (h.result < 0).sum(); aw = (a.result < 0).sum(); al = (a.result > 0).sum()
        # ATS: result (home margin) vs spread_line (home favored by)
        h_ats = h.result - h.spread_line; a_ats = -(a.result - a.spread_line)
        ou = pd.concat([h.total - h.total_line, a.total - a.total_line])
        rec[t] = dict(pf_pg=pf / max(gp[t], 1), pa_pg=pa / max(gp[t], 1),
                      wins=hw + aw, losses=hl + al, ties=len(h) + len(a) - hw - hl - aw - al,
                      home_w=hw, home_l=hl, road_w=aw, road_l=al,
                      home_ats_w=(h_ats > 0).sum(), home_ats_l=(h_ats < 0).sum(),
                      road_ats_w=(a_ats > 0).sum(), road_ats_l=(a_ats < 0).sum(),
                      unders=(ou < 0).sum(), overs=(ou > 0).sum(),
                      tds_pg=None)
    s = s.join(pd.DataFrame(rec).T.drop(columns=["tds_pg"]).astype(float))
    tdg = pbp[(pbp.touchdown == 1) & pbp.td_team.notna()].groupby("td_team").size()
    s["tds_pg"] = (tdg / gp).reindex(teams).fillna(0)
    s = s.astype(float)
    # ranks (1 = best for that team)
    hi_good = ["def_sacks_pg", "def_sack_rate", "def_pressure_rate", "off_ypc", "takeaways_pg", "off_rz_td_pct", "pf_pg", "tds_pg"]
    lo_good = ["off_sacks_allowed_pg", "def_ypc", "giveaways_pg", "def_rz_trips_pg", "def_rz_td_pct", "def_3rd_pct", "pa_pg"]
    for c in hi_good: s[c + "_rk"] = s[c].rank(ascending=False, method="min")
    for c in lo_good: s[c + "_rk"] = s[c].rank(ascending=True, method="min")
    return s


def z(series):
    return (series - series.mean()) / series.std()


def score_game(a, h, S, g):
    """Auto-score checks 1-5 for away a / home h. Returns dict."""
    za = {c: z(S[c]) for c in ["def_sack_rate", "def_pressure_rate", "off_sack_rate", "def_rz_trips_pg", "def_rz_td_pct"]}
    lg = {c: S[c].mean() for c in ["off_ypc", "def_ypc"]}
    mobile_cut = 20.0  # QB rush yds/game
    r = {}

    # Check 1: pass rush vs protection
    def rush_edge(d, o):
        e = (za["def_sack_rate"][d] + za["def_pressure_rate"][d]) / 2 + za["off_sack_rate"][o]
        if S.qb_rush_ypg[o] >= mobile_cut: e *= 0.5
        return e
    e1a, e1h = rush_edge(a, h), rush_edge(h, a)
    r["c1_edge_away"], r["c1_edge_home"] = e1a, e1h
    r["c1"] = side(e1a, e1h, 0.35)

    # Check 2: run game (season 2/3, last 3 1/3)
    def run_edge(o, d):
        ypc = (2 * S.off_ypc[o] + S.off_ypc_l3[o]) / 3
        return (ypc - lg["off_ypc"]) + (S.def_ypc[d] - lg["def_ypc"])
    e2a, e2h = run_edge(a, h), run_edge(h, a)
    r["c2_edge_away"], r["c2_edge_home"] = e2a, e2h
    r["c2"] = side(e2a, e2h, 0.25)

    # Check 3: turnovers
    def to_edge(t, o):
        return (S.takeaways_pg[t] + S.giveaways_pg[o]) - (S.giveaways_pg[t] + S.takeaways_pg[o])
    e3a = to_edge(a, h)
    r["c3_edge_away"], r["c3_edge_home"] = e3a, -e3a
    r["c3"] = side(e3a, -e3a, 0.4)

    # Check 4: red zone defense (trips allowed + TD% allowed; lower is better)
    def rz(t): return -(za["def_rz_trips_pg"][t] + za["def_rz_td_pct"][t])
    e4a, e4h = rz(a), rz(h)
    r["c4_edge_away"], r["c4_edge_home"] = e4a, e4h
    r["c4"] = side(e4a, e4h, 0.4)

    # Check 5: venue split + rest
    hv = S.home_w[h] / max(S.home_w[h] + S.home_l[h], 1)
    av = S.road_w[a] / max(S.road_w[a] + S.road_l[a], 1)
    venue = "home" if hv - av > 0.15 else ("away" if av - hv > 0.15 else None)
    rest_d = (g.home_rest or 7) - (g.away_rest or 7)
    rest = "home" if rest_d >= 2 else ("away" if rest_d <= -2 else None)
    if venue and rest and venue != rest: c5 = rest
    else: c5 = venue or rest
    r["c5"] = c5
    r["venue_note"] = f"{h} home {int(S.home_w[h])}-{int(S.home_l[h])}, {a} road {int(S.road_w[a])}-{int(S.road_l[a])}; rest {int(g.away_rest)}d vs {int(g.home_rest)}d"

    W = {"c1": 2, "c2": 2, "c3": 2, "c4": 2, "c5": 1}
    pa = sum(W[c] for c in W if r[c] == "away"); ph = sum(W[c] for c in W if r[c] == "home")
    r["pts_away"], r["pts_home"] = pa, ph
    return r


def side(ea, eh, thr):
    if ea - eh > thr: return "away"
    if eh - ea > thr: return "home"
    return None


def verdict(lead_pts, passes):
    # 9 auto pts available; Check 0 (3) + Check 6 (1) are manual -> max 13
    need = max(0, 10 - lead_pts)
    if passes: return "Pass rule", need
    if lead_pts >= 9: return "Bet if money agrees", need
    if lead_pts >= 7: return "Needs the handle", need
    if lead_pts >= 6: return "Needs handle + line", need
    if lead_pts >= 4: return "Lean at best", need
    return "Pass", need


def build(G, P, weeks, stats_through):
    S = team_stats(P[P.week <= stats_through], G[G.week <= stats_through])
    rows = []
    for _, g in G[G.week.isin(weeks)].iterrows():
        a, h = g.away_team, g.home_team
        r = score_game(a, h, S, g)
        lead = "away" if r["pts_away"] > r["pts_home"] else ("home" if r["pts_home"] > r["pts_away"] else None)
        lt = {"away": a, "home": h}.get(lead)
        opp = {"away": h, "home": a}.get(lead)
        lp = max(r["pts_away"], r["pts_home"])
        has_line = pd.notna(g.spread_line)
        fav = None
        if has_line and g.spread_line != 0:
            fav = h if g.spread_line > 0 else a
        dog = ({a, h} - {fav}).pop() if fav else None
        # pass rules (auto-checkable)
        passes = []
        if lead:
            if r["c1"] not in (None, lead) and r["c2"] not in (None, lead):
                passes.append(f"{lt} loses both trench checks")
            if lt == dog and (S.tds_pg_rk[lt] >= 28 or S.off_rz_td_pct_rk[lt] >= 28):
                passes.append(f"{lt} is a dog that can't score (bottom 5)")
        v, need = verdict(lp, passes)
        if not lead: v, need = "Pass", 10
        # Check 7 totals
        under, over = [], []
        fav_side = "home" if fav == h else ("away" if fav == a else None)
        if fav_side and r["c2"] == fav_side: under.append("favorite wins the run game")
        if max(S.off_rz_td_pct_rk[a], S.off_rz_td_pct_rk[h]) >= 23:
            under.append("an offense is bottom-10 in RZ TD%")
        if (S.qb_rush_ypg[h] < 20 and S.def_pressure_rate_rk[a] <= 10) or (S.qb_rush_ypg[a] < 20 and S.def_pressure_rate_rk[h] <= 10):
            under.append("pocket QB vs top-10 pass rush")
        if S.unders[a] > S.overs[a] and S.unders[h] > S.overs[h]: under.append("both teams mostly unders")
        if S.pa_pg_rk[a] >= 23 and S.pa_pg_rk[h] >= 23: over.append("both defenses bottom-10 in points allowed")
        if S.off_rz_td_pct_rk[a] <= 12 and S.off_rz_td_pct_rk[h] <= 12: over.append("both offenses top-12 RZ TD%")
        if S.giveaways_pg_rk[a] >= 23 or S.giveaways_pg_rk[h] >= 23: over.append("turnover-prone team involved")
        if g.roof in ("dome", "closed"): over.append("dome")
        tl = "Under" if len(under) >= 2 and len(under) > len(over) else ("Over" if len(over) >= 2 and len(over) > len(under) else "")
        # Check 8 underdog angles
        c8 = ""
        if fav and dog:
            fs, ds = ("home", "away") if fav == h else ("away", "home")
            dog_win = S.wins[dog] > S.losses[dog]
            run_gap = r[f"c2_edge_{fs}"] - r[f"c2_edge_{ds}"]
            if run_gap > 0.6:
                c8 = f"{dog} team total under" + (" (winning-record dog: strongest)" if dog_win else "")
            elif r["c2"] == ds and S.wins[dog] < S.losses[dog]:
                c8 = f"{dog} is a bad team that can run: dog/ML look"
        row = dict(game_id=g.game_id, week=int(g.week), date=g.gameday, day=g.weekday, kickoff=g.gametime,
                   away=a, home=h, matchup=f"{a} @ {h}",
                   spread=fmt_spread(g, a, h), total=g.total_line if has_line else None,
                   away_ml=g.away_moneyline, home_ml=g.home_moneyline, roof=g.roof, div_game=int(g.div_game),
                   away_qb=g.away_qb_name if isinstance(g.away_qb_name, str) else "", home_qb=g.home_qb_name if isinstance(g.home_qb_name, str) else "",
                   pick=lt or "", pick_pts=lp, opp_pts=min(r["pts_away"], r["pts_home"]) if lead else lp,
                   verdict=v, manual_pts_needed=need,
                   c1=team(r["c1"], a, h), c2=team(r["c2"], a, h), c3=team(r["c3"], a, h), c4=team(r["c4"], a, h), c5=team(r["c5"], a, h),
                   pass_rules="; ".join(passes), total_lean=tl, under_reasons="; ".join(under), over_reasons="; ".join(over),
                   check8=c8, venue_rest=r["venue_note"],
                   away_score=g.away_score, home_score=g.home_score, home_margin=g.result, spread_line=g.spread_line, total_line=g.total_line, final_total=g.total)
        rows.append(row)
    return pd.DataFrame(rows), S


def team(sd, a, h): return {"away": a, "home": h}.get(sd, "Even")


def fmt_spread(g, a, h):
    if pd.isna(g.spread_line): return ""
    if g.spread_line == 0: return "PK"
    return f"{h} -{g.spread_line:g}" if g.spread_line > 0 else f"{a} -{-g.spread_line:g}"


last_done = int(G[G.result.notna()].week.max())
# Stats through the last week with results (partial week counts what's played)
upcoming, S = build(G, P, list(range(last_done + 1, 19)), 18)
# include unplayed games of the current week too
cur_unplayed = G[(G.week == last_done) & G.result.isna()]
if len(cur_unplayed):
    u2, _ = build(G, P, [last_done], 18)
    upcoming = pd.concat([u2[u2.game_id.isin(cur_unplayed.game_id)], upcoming])

# Backtest: weeks 4..last_done, stats from prior weeks only
bt = []
for w in range(4, last_done + 1):
    df, _ = build(G, P, [w], w - 1)
    bt.append(df)
bt = pd.concat(bt)
bt = bt[bt.home_margin.notna() & (bt.pick != "")]
def ats(r):
    m = r.home_margin - r.spread_line  # >0 home covered
    if m == 0: return "Push"
    home_cov = m > 0
    return "Win" if (r.pick == r.home) == home_cov else "Loss"
def su(r):
    if r.home_margin == 0: return "Tie"
    return "Win" if (r.pick == r.home) == (r.home_margin > 0) else "Loss"
bt["ats"] = bt.apply(ats, axis=1); bt["su"] = bt.apply(su, axis=1)
def tot(r):
    if not r.total_lean: return ""
    d = r.final_total - r.total_line
    if d == 0: return "Push"
    return "Win" if (r.total_lean == "Under") == (d < 0) else "Loss"
bt["total_result"] = bt.apply(tot, axis=1)

# summary by tier
def tier(p):
    return "9 of 9 auto" if p >= 9 else ("7-8 auto" if p >= 7 else ("5-6 auto" if p >= 5 else "0-4 auto"))
bt["tier"] = bt.pick_pts.apply(tier)
summ = []
for t in ["9 of 9 auto", "7-8 auto", "5-6 auto", "0-4 auto"]:
    x = bt[bt.tier == t]
    w, l = (x.ats == "Win").sum(), (x.ats == "Loss").sum()
    sw, sl = (x.su == "Win").sum(), (x.su == "Loss").sum()
    summ.append(dict(tier=t, games=len(x), ats_w=int(w), ats_l=int(l), ats_pct=round(w / max(w + l, 1), 3),
                     su_w=int(sw), su_l=int(sl), su_pct=round(sw / max(sw + sl, 1), 3)))
x = bt[bt.pass_rules != ""]
summ.append(dict(tier="Pass rule fired", games=len(x), ats_w=int((x.ats == "Win").sum()), ats_l=int((x.ats == "Loss").sum()),
                 ats_pct=round((x.ats == "Win").sum() / max(((x.ats == "Win") | (x.ats == "Loss")).sum(), 1), 3),
                 su_w=int((x.su == "Win").sum()), su_l=int((x.su == "Loss").sum()),
                 su_pct=round((x.su == "Win").sum() / max(((x.su == "Win") | (x.su == "Loss")).sum(), 1), 3)))
tt = bt[bt.total_result.isin(["Win", "Loss"])]
summ.append(dict(tier="Total leans (Check 7)", games=len(tt), ats_w=int((tt.total_result == "Win").sum()), ats_l=int((tt.total_result == "Loss").sum()),
                 ats_pct=round((tt.total_result == "Win").mean(), 3) if len(tt) else 0, su_w=None, su_l=None, su_pct=None))
summary = pd.DataFrame(summ)

# Team table
T = S.reset_index().rename(columns={"index": "team"})
T["record"] = T.apply(lambda r: f"{int(r.wins)}-{int(r.losses)}" + (f"-{int(r.ties)}" if r.ties else ""), axis=1)
T["home_rec"] = T.apply(lambda r: f"{int(r.home_w)}-{int(r.home_l)}", axis=1)
T["road_rec"] = T.apply(lambda r: f"{int(r.road_w)}-{int(r.road_l)}", axis=1)
for c in T.columns:
    if T[c].dtype.kind == "f": T[c] = T[c].round(3)

upcoming = upcoming.drop(columns=["away_score", "home_score", "home_margin", "final_total"])
upcoming.to_csv(f"{OUT}/upcoming_games.csv", index=False)
T.to_csv(f"{OUT}/team_stats.csv", index=False)
bt_out = bt[["game_id", "week", "matchup", "spread", "pick", "pick_pts", "verdict", "pass_rules", "away_score", "home_score", "su", "ats", "total_lean", "total_result"]].copy()
bt_out["away_score"] = bt_out.away_score.astype(int); bt_out["home_score"] = bt_out.home_score.astype(int)
bt_out.to_csv(f"{OUT}/backtest_games.csv", index=False)
summary.to_csv(f"{OUT}/backtest_summary.csv", index=False)
print("stats through week", last_done, "| upcoming", len(upcoming), "| backtest", len(bt))
print(summary.to_string())

# Check 8 backtest: dog team-total under (implied = (total - |spread|)/2), bad-dog-that-can-run ATS
c8 = bt[bt.check8.fillna("") != ""].copy()
res = []
for _, r in c8.iterrows():
    dog = r.check8.split(" ")[0]
    dog_score = r.home_score if dog == r.home else r.away_score
    if "team total under" in r.check8:
        implied = (r.total_line - abs(r.spread_line)) / 2
        res.append(("Dog team total under", "Win" if dog_score < implied else ("Push" if dog_score == implied else "Loss")))
    else:
        m = r.home_margin - r.spread_line
        dog_cov = (m > 0) if dog == r.home else (m < 0)
        res.append(("Bad dog that can run (ATS)", "Push" if m == 0 else ("Win" if dog_cov else "Loss")))
c8r = pd.DataFrame(res, columns=["tier", "r"])
add = []
for t, x in c8r.groupby("tier"):
    w, l = int((x.r == "Win").sum()), int((x.r == "Loss").sum())
    add.append(dict(tier=t, games=len(x), ats_w=w, ats_l=l, ats_pct=round(w / max(w + l, 1), 3), su_w=None, su_l=None, su_pct=None))
summary = pd.concat([summary, pd.DataFrame(add)], ignore_index=True)
summary.to_csv(f"{OUT}/backtest_summary.csv", index=False)
print(summary.tail(3).to_string())

import datetime
json.dump({"updated": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "through_week": last_done},
          open(f"{OUT}/meta.json", "w"))
