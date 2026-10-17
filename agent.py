"""LLM-commanded, model-driven lane-defence agent for GDG NSUT Autopilot.

Every call (opening, then every 3 s of game time):
  1. assess(): a per-row combat model from the live board. For each row: enemy HP, regen and
     the damage they put on our front; our DPS there (soldiers in the row + all archers, which
     sweep rows lowest-first); time to clear the row vs. how long the front and the whole line
     last. Status: breach / critical / weak / ok / open (no defence, no enemy).
  2. Fixes per row, by counter: Penguin vs stacks & Wizards (500 to every enemy in the row),
     Soldiers where a row needs DPS (40 dps per 200), Shields where it needs time (700 hp per
     150), a far Shield to keep every row obstructed (a Pirate crosses faster than one pause).
  3. Options: hold (critical fixes only) / needed (+ weak rows) / speed (+ archers from coins
     above the reserve). Coins below the reserve go only to breach/critical rows.
  4. The LLM gets the rules (system prompt), the assessment and options, can call tools
     (assess_board, row_detail, check_plan) and picks an option or its own cards. Its plan is
     held to the reserve, and critical fixes are added for rows it left untouched. Any LLM
     failure -> the rules' default option.

Layout: Soldier @1, Archers @0 (then @2) behind, line Shield @6. At x=6 a Clown stops ~3.3 tiles
ahead (x~9.3, still on the board and in our range); a Wizard stops at the spawn (x=10) and its
5.3-tile beam can't reach anything at x<=4; a Pirate blast at x=6 (2 u = 1.3 tiles) can't reach
x<=4. Soldiers and archers have unlimited range.

SDK facts: each call runs in a FRESH subprocess (no memory between calls) and a PUBLISH run may
open no other file (requests' trust_env would read ~/.netrc, so it is switched off).
"""
import json
import math
import sys
import time

try:
    from autopilot_sdk.llm import LLMClient
    from autopilot_sdk.tools import run_tool_loop, tool
except Exception:  # offline tests without the SDK
    LLMClient = run_tool_loop = None

    def tool(f):
        return f

# --- game facts (player guide) ----------------------------------------------
PENGUIN, SOLDIER, ARCHER, SHIELD = 0, 1, 2, 3
CLOWN, MONK, WIZARD, PIRATE = 0, 1, 2, 3
ENEMY_IDS = {"EnemClown": CLOWN, "EnemyMonk": MONK, "EnemyWizard": WIZARD, "EnemyPirate": PIRATE}
DEF_IDS = {"defencePenguin": PENGUIN, "defenceSolider": SOLDIER, "defenceArcher": ARCHER, "defenceShield": SHIELD}
DEF_NAME = {PENGUIN: "penguin", SOLDIER: "soldier", ARCHER: "archer", SHIELD: "shield"}
FOE_NAME = {CLOWN: "clown", MONK: "monk", WIZARD: "wizard", PIRATE: "pirate"}
DEFAULT_COSTS = {PENGUIN: 250, SOLDIER: 200, ARCHER: 300, SHIELD: 150}
FOE_HP = {CLOWN: 200, MONK: 280, WIZARD: 500, PIRATE: 30}
REGEN = {CLOWN: 15, WIZARD: 15}                    # hp/s while hurt
FOE_DPS = {CLOWN: 17, MONK: 36, WIZARD: 50, PIRATE: 0}  # on the first defence in their row
PIRATE_BLAST = 200
DPS = {SOLDIER: 40, ARCHER: 26}
PENGUIN_HIT = 500
ROWS = range(6)
MAX_X = 10

# --- tuning -----------------------------------------------------------------
USE_LLM = True
T = {
    "soldier_x": 1,
    "shield_x": 6,            # furthest useful blocker (see docstring); 5, 4 if 6 is taken
    "penguin_x": 7,           # opening penguin in wizard rows: in front of the shield, beamed first
    "archer_cols": (0, 2),    # archers behind the line: x=0 first, then x=2 (out of beam/blast reach)
    "max_archers": 9,
    "reserve_frac": 0.25,     # coin reserve: 25% of the starting coins, clamped to [min, max]...
    "reserve_min": 600,
    "reserve_max": 1200,
    "small_start": 2000,      # ...or 20% of them when the level starts with less than this
    "small_frac": 0.2,
    "wave_hp_per_archer": 1300,  # opening: one archer per this much wave HP (2..6 of them)
    "weak_margin": 1.0,       # row is weak if time-to-clear > front time * this
    "speed_clear_s": 15,      # late game (nothing queued): still buy speed if a row needs longer
    "llm_deadline": 11.0,     # s, hard cap (guide: 15 s per call; keep ~3 s spare)
    "llm_rounds_start": 3,
    "llm_rounds_pause": 3,    # 60 calls/min per team; pauses come every ~4 s real time
    "llm_max_tokens": 500,
}
INF = float("inf")


def log(msg):
    try:
        print(f"[agent] {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _xy(u):
    return int(round(u["position"]["x"])), int(round(u["position"]["y"]))


def card(kind, x, y):
    return {"defenceId": kind, "defenceGridPosition": {"x": x, "y": y}}


def _kxy(c):
    p = c["defenceGridPosition"]
    return c["defenceId"], p["x"], p["y"]


def check(cards, taken, coins, costs):
    """Keep cards that are on the board, on a free tile, not duplicated and affordable in order."""
    out = []
    for c in cards:
        try:
            kind, x, y = (int(v) for v in _kxy(c))
        except (KeyError, TypeError, ValueError):
            continue
        if kind in costs and 0 <= x <= MAX_X and 0 <= y <= 5 and (x, y) not in taken and costs[kind] <= coins:
            taken.add((x, y))
            coins -= costs[kind]
            out.append(card(kind, x, y))
    return out


def cost_of(cards, costs):
    return sum(costs.get(c["defenceId"], 999) for c in cards)


def _r(v, n=1):
    return None if v == INF else round(v, n)


# --- board ------------------------------------------------------------------
def analyse(state):
    costs = {int(k): int(v) for k, v in (state.get("unitCosts") or {}).items()} or dict(DEFAULT_COSTS)
    defs, foes = [], []
    for d in state.get("defences") or []:
        if d.get("health", 0) > 0 and "position" in d:
            x, y = _xy(d)
            defs.append({"k": DEF_IDS.get(d.get("type"), SHIELD), "x": x, "y": y, "hp": float(d["health"]),
                         "max": float(d.get("maxHealth") or d["health"])})
    for e in state.get("enemies") or []:
        if not e.get("isDead") and e.get("health", 0) > 0 and "position" in e:
            x, y = _xy(e)
            foes.append({"k": ENEMY_IDS.get(e.get("type"), CLOWN), "x": x, "y": y, "hp": float(e["health"]),
                         "atk": bool(e.get("isAttacking"))})
    counts, budget = state.get("counts") or {}, state.get("budget") or {}
    gs = state.get("gameSettings") or {}
    queued = int(counts.get("enemiesQueued", 0) or 0)
    total = int(counts.get("totalEnemiesInWave", 0) or 0)
    kills = max(0, total - queued - int(counts.get("activeEnemyCount", 0) or 0))
    coins = float(budget.get("remaining", 0) or 0)
    start = float(budget.get("unlocked", coins) or coins) - 50 * kills
    B = {
        "phase": state.get("phase", "pause"), "costs": costs, "coins": coins, "start": start,
        "defs": defs, "foes": foes, "occ": {(d["x"], d["y"]) for d in defs},
        "queued": queued, "total": total, "reinf": gs.get("reinforcementsEnabled", True),
        "threats": row_threats(state) if state.get("phase") == "start" else {},
        "budget_s": float(state.get("timeBudgetSeconds") or 15), "t": state.get("elapsedSeconds"),
    }
    B["reserve"] = reserve(B)
    return B


def reserve(B):
    """Coins banked for last-moment fixes; only breach/critical rows may spend below it."""
    if B["phase"] == "start" and not B["reinf"]:
        return 0.0  # no later decisions to save for
    s = B["start"]
    if s < T["small_start"]:
        return T["small_frac"] * max(0.0, s)
    return min(T["reserve_max"], max(T["reserve_min"], T["reserve_frac"] * s))


def row_threats(state):
    """{row: {enemyId: expected count}}. Missing allowedRows means any row."""
    threats = {y: {} for y in ROWS}
    for g in state.get("upcomingEnemies") or []:
        try:
            eid, n = int(g.get("enemyId", 0)), int(g.get("count", 0))
        except (TypeError, ValueError):
            continue
        rows = [r for r in (g.get("allowedRows") or ROWS) if r in threats] or list(ROWS)
        for r in rows:
            threats[r][eid] = threats[r].get(eid, 0) + n / len(rows)
    for e in state.get("enemies") or []:
        eid = ENEMY_IDS.get(e.get("type"))
        if eid is not None and "position" in e:
            r = _xy(e)[1]
            if r in threats:
                threats[r][eid] = max(threats[r].get(eid, 0), 1)
    return {y: t for y, t in threats.items() if t}


# --- combat model -----------------------------------------------------------
def assess(B, extra=()):
    """Per-row model of the live board (plus hypothetical `extra` cards [(kind, x, y)])."""
    defs = B["defs"] + [{"k": k, "x": x, "y": y, "hp": 700.0 if k == SHIELD else 150.0, "max": 1.0}
                        for k, x, y in extra]
    archer_dps = DPS[ARCHER] * sum(1 for d in defs if d["k"] == ARCHER)
    rows, t_prev = {}, 0.0
    for y in ROWS:
        foes = sorted((f for f in B["foes"] if f["y"] == y), key=lambda f: f["x"])
        mine = sorted((d for d in defs if d["y"] == y), key=lambda d: d["x"])
        blockers = [d for d in mine if d["k"] != ARCHER]
        walkers = [f for f in foes if f["k"] != PIRATE]
        hp = sum(f["hp"] for f in foes)
        regen = sum(REGEN.get(f["k"], 0) for f in foes)
        sold = sum(DPS[SOLDIER] for d in mine if d["k"] == SOLDIER and any(f["x"] >= d["x"] for f in foes))
        edps = sum(FOE_DPS[f["k"]] for f in walkers)
        front = max(blockers, key=lambda d: d["x"]) if blockers else None
        r = {"row": y, "enemies": [[FOE_NAME[f["k"]], round(f["hp"]), f["x"]] for f in foes],
             "defences": [[DEF_NAME[d["k"]], round(d["hp"]), d["x"]] for d in mine],
             "enemy_hp": round(hp), "regen": regen, "soldier_dps": sold, "enemy_dps_on_front": edps,
             "front": [DEF_NAME[front["k"]], round(front["hp"]), front["x"]] if front else None}
        if foes:
            # Archers sweep rows lowest-first: this row has only its soldiers until lower rows clear.
            hp_left = max(0.0, hp - max(0.0, sold - regen) * t_prev)
            net = sold + archer_dps - regen
            t_clear = t_prev + hp_left / net if net > 0 else INF
            t_prev = t_clear if t_clear < INF else t_prev
            t_front = front["hp"] / edps if front and edps else INF
            t_line = sum(d["hp"] for d in blockers) / edps if edps else INF
            pir = [f for f in foes if f["k"] == PIRATE]
            pirate_hits_line = bool(pir) and front is not None and front["x"] <= T["soldier_x"] + 1
            if not blockers:
                status = "BREACH"
            elif t_clear > t_line or pirate_hits_line:
                status = "critical"
            elif t_clear > t_front * T["weak_margin"]:
                status = "weak"
            else:
                status = "ok"
            first = y == min(f["y"] for f in B["foes"])  # archers are on this row right now
            r.update(dps_here=sold + (archer_dps if first else 0),
                     t_clear=_r(t_clear), t_front=_r(t_front), t_line=_r(t_line), status=status,
                     nearest_enemy_x=foes[0]["x"])
        else:
            r.update(status="open" if not blockers else "quiet")
        rows[y] = r
    return {"rows": rows, "archer_dps_total": archer_dps,
            "max_t_clear": max([r.get("t_clear") or 0 for r in rows.values()] + [0])}


def _pick(xs, y, occ, before=MAX_X + 1):
    return next((x for x in xs if 0 <= x < before and x <= MAX_X and (x, y) not in occ), None)


def row_fixes(B, y, r, occ):
    """(critical, normal) cards for one row, by counter. Mutates occ."""
    crit, norm, why = [], [], []
    foes = [f for f in B["foes"] if f["y"] == y]
    nearest = min((f["x"] for f in foes), default=MAX_X + 1)
    walkers = [f for f in foes if f["k"] != PIRATE]
    types = {f["k"] for f in foes}
    mine = [d for d in B["defs"] if d["y"] == y]
    kinds = {d["k"] for d in mine}
    blockers = [d for d in mine if d["k"] != ARCHER]
    front_x = max((d["x"] for d in blockers), default=-1)
    sx, shx = T["soldier_x"], T["shield_x"]

    def add(lst, kind, x, note):
        if x is not None:
            lst.append(card(kind, x, y)); occ.add((x, y)); why.append(f"r{y} {note}")

    status = r["status"]
    if status == "open":  # obstruct: the next pause will see whatever stops here
        add(crit, SHIELD, _pick((shx, shx - 1, shx - 2), y, occ), "open->shield")
        return crit, norm, why
    if status == "quiet":
        if front_x <= sx and kinds & {SOLDIER}:  # soldier with nothing in front: one pirate kills it
            add(norm, SHIELD, _pick((shx, shx - 1, shx - 2), y, occ), "shield the soldier")
        return crit, norm, why
    if status == "BREACH":
        if walkers:  # penguin: any hit makes it slide through the row for 500 each
            add(crit, PENGUIN, _pick((min(nearest - 1, shx), 3, 2, 1, 0), y, occ, nearest), "BREACH->penguin")
        else:
            add(crit, SHIELD, _pick((min(nearest - 1, shx), 4, 3, 2), y, occ, nearest), "BREACH->shield")
        if nearest > sx + 1 and SOLDIER not in kinds:
            add(crit, SOLDIER, _pick((sx, 0, 2), y, occ, nearest), "BREACH->soldier")
        return crit, norm, why

    lst = crit if status == "critical" else norm
    if PIRATE in types and front_x <= sx + 1:  # pirate would blow up on the soldier
        add(crit, SHIELD, _pick(range(min(shx, nearest - 1), sx + 1, -1), y, occ, nearest), "pirate->bait")
    if status in ("critical", "weak"):
        stack_ok = walkers and all(f["hp"] <= PENGUIN_HIT for f in walkers)
        penguin = False
        if PENGUIN not in kinds and stack_ok and (len(walkers) >= 2 or WIZARD in types):
            # one penguin kills the whole stack. Furthest forward free tile in front of the line, so it is
            # the first thing they hit; else behind the line as a backstop.
            n0 = len(lst)
            add(lst, PENGUIN, _pick([*range(min(shx, nearest - 1), front_x, -1), *range(front_x - 1, sx, -1)],
                                    y, occ, nearest), f"penguin vs {len(walkers)} ({status})")
            penguin = len(lst) > n0
        else:
            t_line = r["t_line"] if r["t_line"] is not None else INF
            need = (r["enemy_hp"] / max(t_line, 1.0)) + r["regen"] - r["dps_here"]
            for _ in range(min(2, max(1, math.ceil(need / DPS[SOLDIER])))):
                add(lst, SOLDIER, _pick((sx, 2, 0, 3, 4), y, occ, min(nearest, front_x if front_x > sx else nearest)),
                    f"soldier (+40dps, need {need:.0f})")
        if not penguin and r["t_front"] is not None and r["t_front"] < 6:  # more wall: in front, else behind
            add(lst, SHIELD, _pick([*range(min(shx, nearest - 1), front_x, -1), *range(front_x - 1, 2, -1)],
                                   y, occ, nearest), "shield (front dying)")
    if not kinds & {SOLDIER, PENGUIN} and not any(c["defenceId"] in (SOLDIER, PENGUIN) for c in crit + norm):
        add(crit if status == "critical" else norm, SOLDIER, _pick((sx, 0, 2), y, occ, nearest), "no killer")
    return crit, norm, why


def archer_slots(B, occ, threats=None):
    """Archer tiles: quiet rows first (unreachable), then x=0/x=2 behind a built line."""
    out = []
    lined = [y for y in ROWS if any((x, y) in occ for x in range(3, MAX_X + 1))]
    quiet = [y for y in ROWS if threats is not None and y not in threats]
    for col in T["archer_cols"]:
        for y in quiet + lined:
            if (col, y) not in occ and (col, y) not in out and not (col > 0 and y in quiet):
                out.append((col, y))
    return out


def keep_reserve(cards, coins, floor, costs):
    out, left = [], coins
    for c in cards:
        p = costs.get(c["defenceId"], 999)
        if left - p >= floor:
            out.append(c); left -= p
    return out


# --- options: opening -------------------------------------------------------
def opening_options(B):
    threats, costs, coins, res = B["threats"], B["costs"], B["coins"], B["reserve"]
    sx, shx = T["soldier_x"], T["shield_x"]
    order = sorted(threats, key=lambda y: (PIRATE not in threats[y], MONK not in threats[y],
                                           WIZARD not in threats[y], y))
    occ = set(B["occ"])
    blocks, killers, extra = [], [], []
    for y in order:
        t = threats[y]
        x = _pick((shx, shx - 1, shx - 2), y, occ)
        if x is not None:
            blocks.append(card(SHIELD, x, y)); occ.add((x, y))
        if (sx, y) not in occ:
            killers.append(card(SOLDIER, sx, y)); occ.add((sx, y))
        others = t.get(CLOWN, 0) + t.get(MONK, 0) + t.get(PIRATE, 0)
        if t.get(WIZARD, 0) >= max(1, others) and (T["penguin_x"], y) not in occ:
            extra.append(card(PENGUIN, T["penguin_x"], y)); occ.add((T["penguin_x"], y))
        if not B["reinf"] and t.get(WIZARD, 0) >= 1.5 and (T["penguin_x"] + 1, y) not in occ:
            extra.append(card(PENGUIN, T["penguin_x"] + 1, y)); occ.add((T["penguin_x"] + 1, y))
        if not B["reinf"] and t.get(CLOWN, 0) + t.get(MONK, 0) >= 2.5 and (2, y) not in occ:
            extra.append(card(SOLDIER, 2, y)); occ.add((2, y))  # no pauses to add dps later
    core = blocks + killers + extra  # every row obstructed first: pirates outrun a pause
    wave_hp = sum(FOE_HP.get(k, 200) * n for t in threats.values() for k, n in t.items())
    n_base = max(2, min(6, round(wave_hp / T["wave_hp_per_archer"])))
    slots = archer_slots(B, occ, threats)
    arch = [card(ARCHER, x, y) for x, y in slots][: T["max_archers"]]
    left = coins - cost_of(core, costs)
    opts = {
        "fortress": (core + keep_reserve(arch, left, res, costs),
                     f"every row blocked (shield@6) + soldier@1, then every archer the coins allow above the "
                     f"{res:.0f} reserve (speed)"),
        "balanced": (core + keep_reserve(arch[:n_base], left, res, costs),
                     f"line + {n_base} archers (one per ~{T['wave_hp_per_archer']} wave hp), keeps the reserve"),
        "lean": (core, "only shields@6 + soldiers@1"),
    }
    return opts, "fortress", blocks + killers, {"wave_hp": wave_hp, "archers_suggested": n_base,
                                                "threats": {y: {FOE_NAME.get(k, k): round(v, 1) for k, v in t.items()}
                                                            for y, t in threats.items()}}


# --- options: pauses --------------------------------------------------------
def pause_options(B):
    costs, res = B["costs"], B["reserve"]
    A = assess(B)
    occ = set(B["occ"])
    order = sorted(ROWS, key=lambda y: ["BREACH", "critical", "open", "weak", "ok", "quiet"].index(A["rows"][y]["status"]))
    critical, normal, why = [], [], []
    for y in order:
        c, n, w = row_fixes(B, y, A["rows"][y], occ)
        critical += c; normal += n; why += w
    needed = critical + keep_reserve(normal, B["coins"] - cost_of(critical, costs), res, costs)
    n_arch = sum(1 for d in B["defs"] if d["k"] == ARCHER)
    arch = [card(ARCHER, x, y) for x, y in archer_slots(B, occ)][: max(0, T["max_archers"] - n_arch)]
    speed = needed + keep_reserve(arch, B["coins"] - cost_of(needed, costs), res, costs)
    want_speed = len(speed) > len(needed) and (B["queued"] > 0 or A["max_t_clear"] > T["speed_clear_s"])
    opts = {
        "speed": (speed, f"needed + archers with the coins above the {res:.0f} reserve (faster kills, all rows)"),
        "needed": (needed, "fixes: " + (", ".join(why) or "nothing needed")),
        "hold": (critical, "only critical fixes (breach / line about to fall / open rows), keep coins"),
    }
    default = "speed" if want_speed else ("needed" if needed else "hold")
    return opts, default, critical, A


# --- LLM decision -----------------------------------------------------------
PROMPT = """You command a lane-defence game. Grid x 0-10 (base x=0; enemies enter at x=10 and walk left), rows y 0-5, 1 tile = 1.5 units. Enemies never change rows. ONE enemy crossing x=0 = LOSS.
DEFENCES (id name cost hp):
0 penguin 250 200hp: idle until hit, then slides forward through its row dealing 500 to EVERY enemy it touches, then gone. Best vs stacks (2+ enemies) and wizards. Wasted on a lone clown.
1 soldier 200 150hp: 40 dps at the first enemy anywhere ahead in its own row (unlimited range). 0.20 dps/coin, one row.
2 archer 300 100hp: 26 dps into ANY row; all archers focus the lowest row with enemies, nearest to base. 0.087 dps/coin but covers every row and speeds up the whole game. Keep at x=0/x=2 behind the line.
3 shield 150 700hp: no attack; enemies stop at the first defence in their row and their shots/beams hit it first. 4.7 hp/coin.
ENEMIES: clown 200hp +15/s regen, stops 3.3 tiles before the first defence, throws 17 dps. monk 280hp, 2.3 tiles/s, melee 36 dps. wizard 500hp +15/s regen, stops 5.3 tiles before the first defence, beam 50 dps (switches to the next defence in range). pirate 30hp, 5.3 tiles/s (crosses in 1.9 s, faster than one pause), explodes for 200 on the first defence it reaches (or where it dies), radius 1.3 tiles: same tile, x+-1, and the tiles above/below.
LAYOUT: soldier@1, archers@0/@2, shield@6 in every row. Shield@6 keeps clowns on the board (x~9.3, in range), wizards' beams and pirate blasts can't reach x<=4. Never leave a row unobstructed. Never put a shield past x=6 (clowns would stop off the board).
COUNTERS: regen enemies (clown, wizard) vs a lone soldier: 2+ clowns out-heal it -> penguin or archers. monks: shield in front + soldier. wizards: penguin in front of the shield, or lots of dps. pirates: a shield far ahead of the soldier.
SCORE of a WON game /100: win 25, kills 25, defences still standing 25, speed 10, coins left 15. A LOST game scores only kills. Priorities: 1) never lose, 2) keep defences alive, 3) speed, 4) coins left.
COINS: keep `reserve` coins banked for surprises; spend below it ONLY for rows with status BREACH or critical. Coins above the reserve should buy speed (archers) while enemies are still coming.
The user message has assess_board() output: per row enemy_hp, regen, our dps_here, enemy_dps_on_front, t_clear (s to kill the row), t_front / t_line (s our front / whole line survives), status (BREACH: nothing stops them; critical: line falls before they die; weak: front falls; ok; open: no defence and no enemy; quiet). Rule-based options are precomputed. Tools: row_detail(row) re-runs the model for one row with optional extra cards, check_plan(placements) validates cards and shows the new model.
Answer with JSON only: {"option": "<name>", "reason": "<=12 words"} or {"placements": [[defenceId, x, y], ...], "reason": "..."}"""


def _from_triples(raw):
    cards = []
    for p in raw or []:
        try:
            if isinstance(p, dict):
                g = p.get("defenceGridPosition") or {}
                k, x, y = p.get("defenceId"), p.get("x", g.get("x")), p.get("y", g.get("y"))
            else:
                k, x, y = p
            cards.append(card(int(k), int(x), int(y)))
        except (TypeError, ValueError):
            continue
    return cards


def llm_decide(B, opts, default, briefing):
    if not USE_LLM or LLMClient is None:
        return None, "llm off"
    t0 = time.monotonic()
    deadline = min(T["llm_deadline"], B["budget_s"] - 3)
    if deadline < 1:
        return None, "no time"

    @tool
    def row_detail(row: int, extra: list = None) -> dict:
        """Combat model for one row, optionally with hypothetical extra cards [[defenceId, x, y], ...].

        Args:
            row: Board row, 0-5.
            extra: optional cards to add before modelling, as [defenceId, x, y].
        """
        ex = [tuple(_kxy(c)) for c in _from_triples(extra or [])]
        return assess(B, ex)["rows"][int(row)]

    @tool
    def check_plan(placements: list) -> dict:
        """Validate placements [[defenceId, x, y], ...]: accepted cards, cost, coins left vs reserve, new row statuses.

        Args:
            placements: list of [defenceId, x, y].
        """
        ok = check(_from_triples(placements), set(B["occ"]), B["coins"], B["costs"])
        left = B["coins"] - cost_of(ok, B["costs"])
        A = assess(B, [tuple(_kxy(c)) for c in ok]) if B["phase"] != "start" else None
        return {"accepted": [list(_kxy(c)) for c in ok], "cost": cost_of(ok, B["costs"]), "coins_left": left,
                "below_reserve": left < B["reserve"],
                "row_status": {y: r["status"] for y, r in A["rows"].items()} if A else None}

    msg = {
        "phase": B["phase"], "game_time": B["t"], "coins": B["coins"], "reserve": round(B["reserve"]),
        "enemies_still_to_spawn": B["queued"], "wave_size": B["total"], "reinforcements_later": bool(B["reinf"]),
        "assess_board": briefing,
        "options": {n: {"cost": cost_of(c, B["costs"]), "cards": [list(_kxy(x)) for x in c], "what": w}
                    for n, (c, w) in opts.items()},
        "rules_recommend": default,
    }
    client = LLMClient()
    # PUBLISH sandbox: any file open outside agent.py kills the call. requests' trust_env reads
    # ~/.netrc (and proxy/CA env files) before each request; the gateway is a local http relay.
    session = getattr(client, "_session", None)
    if session is not None:
        session.trust_env = False
    run = run_tool_loop(
        client,
        [{"role": "system", "content": PROMPT}, {"role": "user", "content": json.dumps(msg, separators=(",", ":"), default=str)}],
        [row_detail, check_plan],
        deadline_s=deadline,
        max_rounds=T["llm_rounds_start"] if B["phase"] == "start" else T["llm_rounds_pause"],
        max_tokens=T["llm_max_tokens"], temperature=0,
    )
    text = (run.content or "").strip()
    if "{" in text:
        text = text[text.index("{"): text.rindex("}") + 1]
    answer = json.loads(text)
    reason = str(answer.get("reason", ""))[:80]
    tools_used = [c.name for c in run.calls]
    if answer.get("option") in opts:
        cards, src = opts[answer["option"]][0], f"llm option={answer['option']}"
    elif isinstance(answer.get("placements"), list):
        cards, src = _from_triples(answer["placements"]), "llm custom"
    else:
        return None, f"llm unusable answer: {text[:80]}"
    return cards, f"{src} ({reason}) tools={tools_used} {time.monotonic() - t0:.1f}s"


def enforce_reserve(B, cards, critical_rows):
    """LLM plans keep the reserve too: below it, only cards for breach/critical rows survive."""
    out, left = [], B["coins"]
    for c in cards:
        p = B["costs"].get(c["defenceId"], 999)
        if left - p >= B["reserve"] or _kxy(c)[2] in critical_rows:
            out.append(c); left -= p
    return out


# --- entry point ------------------------------------------------------------
def get_reinforcements(state):
    try:
        B = analyse(state)
        if B["phase"] == "start":
            try:
                log("STATE " + json.dumps(state, separators=(",", ":")))
            except Exception:
                pass
            opts, default, critical, briefing = opening_options(B)
            critical_rows = set(ROWS)
        else:
            opts, default, critical, A = pause_options(B)
            briefing = A
            critical_rows = {y for y, r in A["rows"].items() if r["status"] in ("BREACH", "critical", "open")}
            log("rows " + " | ".join(
                f"r{y} {r['status']} hp={r['enemy_hp']} dps={r.get('dps_here', 0)} clear={r.get('t_clear')}s "
                f"line={r.get('t_line')}s" for y, r in A["rows"].items() if r["status"] not in ("quiet",)))
        log(f"{B['phase']} t={B['t']} coins={B['coins']:.0f} reserve={B['reserve']:.0f} queued={B['queued']}/{B['total']} "
            + " ".join(f"{n}={cost_of(c, B['costs'])}" for n, (c, _) in opts.items()) + f" default={default}")
        if B["phase"] != "start" and not B["foes"] and not opts[default][0]:
            chosen, src = [], "nothing to do"
        else:
            try:
                chosen, src = llm_decide(B, opts, default, briefing)
            except Exception as exc:
                chosen, src = None, f"llm failed: {type(exc).__name__}: {str(exc)[:80]}"
            if chosen is None:
                chosen, src = opts[default][0], f"rules {default} ({src})"
            else:
                chosen = enforce_reserve(B, chosen, critical_rows)
            touched = {_kxy(c)[2] for c in chosen}
            chosen = chosen + [c for c in critical if _kxy(c)[2] not in touched]  # safety net
        out = check(chosen, set(B["occ"]), B["coins"], B["costs"])
        log(f"{src} -> placing {[_kxy(c) for c in out]}")
        return out
    except Exception as exc:
        log(f"ERROR {type(exc).__name__}: {exc}")
        return []
