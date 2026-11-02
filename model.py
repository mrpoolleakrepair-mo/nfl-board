"""NFL Betting Checklist auto-scorer. Usage: python3 -I model.py <datadir> <outdir>"""
import sys, json, numpy as np, pandas as pd

D, OUT = sys.argv[1], sys.argv[2]
SEASON = 2026
games_all = pd.read_csv(f"{D}/games.csv")
G = games_all[(games_all.season == SEASON) & (games_all.game_type == "REG")].copy()
P = pd.read_parquet(f"{D}/pbp_2026.parquet")
P = P[P.season_type == "REG"]
import os
PRIOR_WEEKS = 4  # weeks 1-4 lean on last season's final 8 weeks until 2026 data builds up
if os.path.exists(f"{D}/pbp_2025.parquet"):
    P25 = pd.read_parquet(f"{D}/pbp_2025.parquet")
    P25 = P25[(P25.season_type == "REG") & (P25.week >= 11)]
    G25 = games_all[(games_all.season == SEASON - 1) & (games_all.game_type == "REG") & (games_all.week >= 11)].copy()
else:
    P25, G25 = P.iloc[0:0], G.iloc[0:0]


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


def qb_profiles(pbp):
    """Per-QB rushing (designed runs + scrambles) per game he played, from the same plays the team stats use."""
    ru = pbp[(pbp.rush_attempt == 1) & (pbp.qb_kneel != 1) & pbp.rusher_player_id.notna()]
    db = pbp[pbp.passer_player_id.notna()]
    gp = db.groupby("passer_player_id").game_id.nunique()
    gp = gp[db.groupby("passer_player_id").size() >= 5]
    yds = ru.groupby("rusher_player_id").yards_gained.sum()
    scr = pbp[pbp.qb_scramble == 1].groupby("rusher_player_id").size()
    dbk = pbp[pbp.qb_dropback == 1].groupby("passer_player_id").size()
    out = {}
    for pid, n in gp.items():
        out[pid] = dict(games=int(n), rush_ypg=float(yds.get(pid, 0)) / n, scramble_rate=float(scr.get(pid, 0)) / max(float(dbk.get(pid, 1)), 1))
    return out


def qb_info(team, qid, qname, S, QB, starters):
    """Who starts, is he the usual starter, and is he mobile (20+ rush yds a game)."""
    prof = QB.get(qid) if isinstance(qid, str) else None
    ypg = prof["rush_ypg"] if prof else float(S.qb_rush_ypg[team])
    usual = starters.get(team)
    # Backup = not the opening-day starter and hasn't started 3 straight (an injury fill-in, not a new No. 1)
    backup = bool(usual and isinstance(qid, str) and qid != usual[0] and usual[1] != {qid})
    mobile = ypg >= 20.0
    label = ("Backup · " if backup else "") + ("Mobile" if mobile else "Pocket")
    return dict(name=qname if isinstance(qname, str) else "", ypg=ypg, mobile=mobile, backup=backup, label=label,
                games=prof["games"] if prof else 0)


def score_game(a, h, S, g, qa=None, qh=None):
    """Auto-score checks 1-5 for away a / home h. Returns dict."""
    za = {c: z(S[c]) for c in ["def_sack_rate", "def_pressure_rate", "off_sack_rate", "def_rz_trips_pg", "def_rz_td_pct"]}
    lg = {c: S[c].mean() for c in ["off_ypc", "def_ypc"]}
    mobile_cut = 20.0  # QB rush yds/game
    r = {}

    # Check 1: pass rush vs protection
    qbs = {a: qa, h: qh}
    def rush_edge(d, o):
        e = (za["def_sack_rate"][d] + za["def_pressure_rate"][d]) / 2 + za["off_sack_rate"][o]
        q = qbs.get(o)
        if q and q["backup"]: e += 0.5  # backup starter: checklist's biggest pass-rush edge
        if (q["mobile"] if q else S.qb_rush_ypg[o] >= mobile_cut): e *= 0.5
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
    p, g = P[P.week <= stats_through], G[G.week <= stats_through]
    if min(weeks) <= PRIOR_WEEKS and len(P25):
        p, g = pd.concat([P25, p], ignore_index=True), pd.concat([G25, g], ignore_index=True)
    S = team_stats(p, g)
    QB = qb_profiles(p)
    played = g[g.result.notna() & g.away_qb_id.notna()]
    st = pd.concat([played[["away_team", "away_qb_id"]].set_axis(["t", "q"], axis=1), played[["home_team", "home_qb_id"]].set_axis(["t", "q"], axis=1)])
    st = st.assign(w=list(played.week) * 2)
    starters = {}
    for t, x in st.sort_values("w").groupby("t"):
        starters[t] = (x.q.iloc[0], set(x.q.iloc[-3:]) if len(x) >= 3 else set())  # (opening-day starter, QB if he started the last 3)
    known = G[G.away_qb_id.notna()].sort_values("week")
    last_qb = {}
    for _, k in known.iterrows():
        last_qb[k.away_team] = (k.away_qb_id, k.away_qb_name); last_qb[k.home_team] = (k.home_qb_id, k.home_qb_name)
    rows = []
    for _, g in G[G.week.isin(weeks)].iterrows():
        a, h = g.away_team, g.home_team
        qa_id, qa_nm = (g.away_qb_id, g.away_qb_name) if isinstance(g.away_qb_id, str) else last_qb.get(a, (None, None))
        qh_id, qh_nm = (g.home_qb_id, g.home_qb_name) if isinstance(g.home_qb_id, str) else last_qb.get(h, (None, None))
        qa, qh = qb_info(a, qa_id, qa_nm, S, QB, starters), qb_info(h, qh_id, qh_nm, S, QB, starters)
        r = score_game(a, h, S, g, qa, qh)
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
        if (not qh["mobile"] and S.def_pressure_rate_rk[a] <= 10) or (not qa["mobile"] and S.def_pressure_rate_rk[h] <= 10):
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
                   away_rest=g.away_rest, home_rest=g.home_rest, away_qb=qa["name"], home_qb=qh["name"], away_qb_type=qa["label"], home_qb_type=qh["label"],
                   away_qb_ypg=round(qa["ypg"], 1), home_qb_ypg=round(qh["ypg"], 1), away_qb_backup=int(qa["backup"]), home_qb_backup=int(qh["backup"]),
                   away_qb_listed=int(isinstance(g.away_qb_id, str)), home_qb_listed=int(isinstance(g.home_qb_id, str)),
                   away_rush_top10=int(S.def_pressure_rate_rk[a] <= 10 or S.def_sacks_pg_rk[a] <= 10), home_rush_top10=int(S.def_pressure_rate_rk[h] <= 10 or S.def_sacks_pg_rk[h] <= 10),
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
    upcoming = pd.concat([u2[u2.game_id.isin(cur_unplayed.game_id)], upcoming], ignore_index=True)

# Backtest: weeks 4..last_done, stats from prior weeks only
bt = []
for w in range(1, last_done + 1):
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
w_, l_ = (bt.ats == "Win").sum(), (bt.ats == "Loss").sum(); sw_, sl_ = (bt.su == "Win").sum(), (bt.su == "Loss").sum()
summ.append(dict(tier="All picks", games=len(bt), ats_w=int(w_), ats_l=int(l_), ats_pct=round(w_ / (w_ + l_), 3),
                 su_w=int(sw_), su_l=int(sl_), su_pct=round(sw_ / (sw_ + sl_), 3)))
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


# ---------- Odds feed (The Odds API), when present ----------
def med_point(df, market, outcome):
    x = df[(df.market == market) & (df.outcome == outcome)].point.dropna()
    return float(x.median()) if len(x) else None

def fmt_line(home_pt, a, h):
    if home_pt is None: return ""
    if home_pt == 0: return "PK"
    return f"{h} {home_pt:+g}".replace("+", "+") if home_pt < 0 else f"{a} {-home_pt:+g}"

line_cols = ["line_open", "line_now", "move_toward", "move_pts", "total_open", "total_now", "total_move", "cz_spread", "cz_total", "cz_q1", "cz_h1", "lines_seen"]
for c in line_cols: upcoming[c] = pd.Series([""] * len(upcoming), index=upcoming.index, dtype=object)
snap_path, per_path = f"{OUT}/odds_snapshots.csv", f"{OUT}/caesars_periods.csv"
if os.path.exists(snap_path):
    sn = pd.read_csv(snap_path)
    pr = pd.read_csv(per_path) if os.path.exists(per_path) else sn.iloc[0:0]
    for i, r in upcoming.iterrows():
        x = sn[sn.game_id == r.game_id]
        if not len(x): continue
        first, last = x[x.ts == x.ts.min()], x[x.ts == x.ts.max()]
        so, sc = med_point(first, "spreads", "home"), med_point(last, "spreads", "home")
        to, tc = med_point(first, "totals", "over"), med_point(last, "totals", "over")
        cz = last[last.book == "williamhill_us"]
        upcoming.at[i, "line_open"], upcoming.at[i, "line_now"] = fmt_line(so, r.away, r.home), fmt_line(sc, r.away, r.home)
        if so is not None and sc is not None and abs(sc - so) >= 0.5:
            upcoming.at[i, "move_toward"] = r.home if sc < so else r.away
            upcoming.at[i, "move_pts"] = abs(sc - so)
        upcoming.at[i, "total_open"], upcoming.at[i, "total_now"] = to if to is not None else "", tc if tc is not None else ""
        if to is not None and tc is not None:
            upcoming.at[i, "total_move"] = tc - to
            if abs(tc - to) >= 3 and r.total_lean:
                upcoming.at[i, "total_lean"] = ""
                upcoming.at[i, "under_reasons"] = upcoming.at[i, "over_reasons"] = f"Total moved {tc - to:+g} since first seen: skip it (pass rule)"
        upcoming.at[i, "cz_spread"] = fmt_line(med_point(cz, "spreads", "home"), r.away, r.home)
        czt = med_point(cz, "totals", "over"); upcoming.at[i, "cz_total"] = czt if czt is not None else ""
        p = pr[pr.game_id == r.game_id]
        if len(p):
            p = p[p.ts == p.ts.max()]
            q1s, q1t, h1s, h1t = med_point(p, "spreads_q1", "home"), med_point(p, "totals_q1", "over"), med_point(p, "spreads_h1", "home"), med_point(p, "totals_h1", "over")
            upcoming.at[i, "cz_q1"] = " · ".join(v for v in [fmt_line(q1s, r.away, r.home), f"O/U {q1t:g}" if q1t is not None else ""] if v)
            upcoming.at[i, "cz_h1"] = " · ".join(v for v in [fmt_line(h1s, r.away, r.home), f"O/U {h1t:g}" if h1t is not None else ""] if v)
        upcoming.at[i, "lines_seen"] = f"{x.ts.min()[:10]} to {x.ts.max()[:10]}"
    print("odds merged for", (upcoming.line_now != "").sum(), "games")

# Played games with stored open/close lines: test Check 6 and the 1Q/1H theories
cl_path = f"{OUT}/closing_lines.csv"
per_rows = []
if os.path.exists(cl_path):
    cl = pd.read_csv(cl_path)
    q = P.groupby(["game_id", "qtr"])[["total_home_score", "total_away_score"]].max().reset_index()
    def score_at(gid, qtr):
        x = q[(q.game_id == gid) & (q.qtr <= qtr)]
        return (x.total_away_score.max(), x.total_home_score.max()) if len(x) else (None, None)
    def tally(name, res):
        w, l = res.count("Win"), res.count("Loss")
        per_rows.append(dict(tier=name, games=w + l, ats_w=w, ats_l=l, ats_pct=round(w / max(w + l, 1), 3)))
    c6, q1won, q1cov, h1cov, q1big = [], [], [], [], []
    for _, g in G[G.result.notna() & G.game_id.isin(cl.game_id)].iterrows():
        x = cl[cl.game_id == g.game_id]
        so, sc = med_point(x[x.ts == "open"], "spreads", "home"), med_point(x[x.ts == "close"], "spreads", "home")
        m = g.result - g.spread_line
        if so is not None and sc is not None and abs(sc - so) >= 1 and m != 0:
            toward_home = sc < so
            c6.append("Win" if toward_home == (m > 0) else "Loss")
        cz = x[(x.ts == "close") & (x.book == "williamhill_us")]
        q1s, h1s = med_point(cz, "spreads_q1", "home"), med_point(cz, "spreads_h1", "home")
        if q1s is not None and q1s != 0:
            fav_home = q1s < 0
            if g.result != 0: q1won.append("Win" if fav_home == (g.result > 0) else "Loss")
            if abs(q1s) >= 2.5 and g.result != 0: q1big.append("Win" if fav_home == (g.result > 0) else "Loss")
            a1, h1 = score_at(g.game_id, 1)
            if a1 is not None:
                mm = (h1 - a1) + q1s
                if mm != 0: q1cov.append("Win" if (mm > 0) == fav_home else "Loss")
        if h1s is not None and h1s != 0:
            a2, h2 = score_at(g.game_id, 2)
            if a2 is not None:
                mm = (h2 - a2) + h1s
                if mm != 0: h1cov.append("Win" if (mm > 0) == (h1s < 0) else "Loss")
    tally("Check 6: side the line moved toward (ATS)", c6)
    tally("1Q favorite won the game", q1won)
    tally("1Q favorite of 2.5+ won the game", q1big)
    tally("1Q favorite covered the 1Q spread", q1cov)
    tally("1H favorite covered the 1H spread", h1cov)
pd.DataFrame(per_rows, columns=["tier", "games", "ats_w", "ats_l", "ats_pct"]).to_csv(f"{OUT}/lines_results.csv", index=False)

upcoming = upcoming.drop(columns=["away_score", "home_score", "home_margin", "final_total"])
upcoming.to_csv(f"{OUT}/upcoming_games.csv", index=False)
T.to_csv(f"{OUT}/team_stats.csv", index=False)
bt["basis"] = bt.week.apply(lambda w: "2026 + late 2025" if w <= PRIOR_WEEKS else "2026 only")
bt_out = bt[["game_id", "week", "date", "matchup", "away", "home", "spread", "pick", "pick_pts", "verdict", "pass_rules", "c1", "c2", "c3", "c4", "c5", "away_score", "home_score", "su", "ats", "total_lean", "total_result", "check8", "basis"]].copy()
bt_out["away_score"] = bt_out.away_score.astype(int); bt_out["home_score"] = bt_out.home_score.astype(int)
bt_out.to_csv(f"{OUT}/backtest_games.csv", index=False)

# Weekly accuracy
wk = []
for w, x in bt.groupby("week"):
    s7 = x[x.pick_pts >= 7]
    wk.append(dict(week=int(w), games=len(x), su_w=int((x.su == "Win").sum()), su_l=int((x.su == "Loss").sum()),
                   ats_w=int((x.ats == "Win").sum()), ats_l=int((x.ats == "Loss").sum()),
                   strong_ats_w=int((s7.ats == "Win").sum()), strong_ats_l=int((s7.ats == "Loss").sum())))
pd.DataFrame(wk).to_csv(f"{OUT}/backtest_weekly.csv", index=False)

# Each check on its own: when a team wins the check, how often does it win the game / cover?
names = {"c1": "C1 Pass rush", "c2": "C2 Run game", "c3": "C3 Turnovers", "c4": "C4 Red zone D", "c5": "C5 Situational spot"}
ck = []
for c, nm in names.items():
    x = bt[bt[c] != "Even"]
    if not len(x): continue
    won = x.apply(lambda r: (r[c] == r.home) == (r.home_margin > 0) if r.home_margin != 0 else None, axis=1).dropna()
    m = x.home_margin - x.spread_line
    cov = x[m != 0].apply(lambda r: (r[c] == r.home) == ((r.home_margin - r.spread_line) > 0), axis=1)
    ck.append(dict(check=nm, games=len(x), su_w=int(won.sum()), su_l=int((~won.astype(bool)).sum()), su_pct=round(won.mean(), 3),
                   ats_w=int(cov.sum()), ats_l=int((~cov).sum()), ats_pct=round(cov.mean(), 3)))
pd.DataFrame(ck).to_csv(f"{OUT}/backtest_checks.csv", index=False)
print(pd.DataFrame(ck).to_string()); print(pd.DataFrame(wk).to_string())
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
