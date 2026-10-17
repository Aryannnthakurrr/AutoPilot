"""LLM-commanded lane-defence agent for GDG NSUT Autopilot.

Every call (opening and each 3 s pause):
  1. Rules turn the live board into 2-3 named, priced OPTIONS (e.g. fortress / balanced /
     lean at the opening; reinforce / patch / hold at pauses), built on these principles:
       - the line in each row: Soldier @1, a Shield @~4 in front (or a Penguin @3 where
         Wizards dominate), Archers @0 behind it or in quiet rows; soldiers have unlimited range;
       - pirate blasts (2 u) at the x=4 shield miss the soldier and archer; neighbouring rows'
         shields are staggered (4/6) so one blast doesn't hit another row's shield;
       - every row stays obstructed at every pause (a Pirate crosses faster than one pause);
         shields stop at x=6 so stopped Clowns stay on the board, visible and in range;
       - Penguins only where a row can't win alone (regen >= dps, Wizard, 2+ Monks, worn front);
       - spend early on lasting defences; keep a coin floor (500, or 20% of starting coins
         under 2000) that only critical blockers may dip into.
  2. The LLM sees the board + options, may call tools (lane_report, check_plan), and picks an
     option or proposes its own placements.
  3. Safety net: critical cards (empty rows under attack, pirate baits) are added for rows the
     LLM left untouched; any LLM error/timeout/garbage -> the rules' default option.

Facts read from the SDK source: each call runs in a FRESH subprocess (no memory between calls)
and a PUBLISH run may open no other file. So everything is derived from the live state.
"""
import json
import sys
import time

try:
    from autopilot_sdk.llm import LLMClient
    from autopilot_sdk.tools import run_tool_loop, tool
except Exception:  # offline tests without the SDK
    LLMClient = run_tool_loop = None

    def tool(f):
        return f

# --- game facts -------------------------------------------------------------
PENGUIN, SOLDIER, ARCHER, SHIELD = 0, 1, 2, 3
CLOWN, MONK, WIZARD, PIRATE = 0, 1, 2, 3
ENEMY_IDS = {"EnemClown": CLOWN, "EnemyMonk": MONK, "EnemyWizard": WIZARD, "EnemyPirate": PIRATE}
DEF_IDS = {"defencePenguin": PENGUIN, "defenceSolider": SOLDIER, "defenceArcher": ARCHER, "defenceShield": SHIELD}
DEF_NAME = {PENGUIN: "penguin", SOLDIER: "soldier", ARCHER: "archer", SHIELD: "shield"}
FOE_NAME = {CLOWN: "clown", MONK: "monk", WIZARD: "wizard", PIRATE: "pirate"}
DEFAULT_COSTS = {PENGUIN: 250, SOLDIER: 200, ARCHER: 300, SHIELD: 150}
DPS = {SOLDIER: 40, ARCHER: 26}
REGEN = {CLOWN: 15, WIZARD: 15}
ROWS = range(6)
MAX_X = 10

# --- tuning -----------------------------------------------------------------
USE_LLM = True
T = {
    "soldier_x": 1,
    "shield_x": 4,            # line shield / row blocker: any pirate blast here misses soldier@1, archer@0
    "shield_x_max": 6,        # clowns stop ~3.3 tiles in front of it: x<=6 keeps them on the board
    "penguin_x": 3,           # opening penguin vs wizards: outside a neighbouring x=4 shield's blast
    "max_archers": 6,
    "balanced_archers": 3,
    "balanced_extra": 600,    # balanced option keeps this much above the coin floor
    "floor_abs": 500,         # coin floor kept for last-minute fixes...
    "floor_frac": 0.2,        # ...or this share of the starting coins when they are under:
    "floor_small_start": 2000,
    "early_frac": 0.4,        # pause: >= this share of the wave still queued counts as "early"
    "worn_frac": 0.35,        # front defence below this (with 2+ attackers) -> penguin
    "llm_deadline": 11.0,     # s, hard cap (guide: 15 s per call; keep ~3 s spare)
    "llm_rounds_start": 3,
    "llm_rounds_pause": 2,    # 60 calls/min per team; pauses come every ~4 s real time
    "llm_max_tokens": 400,
}

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


# --- board ------------------------------------------------------------------
def analyse(state):
    costs = {int(k): int(v) for k, v in (state.get("unitCosts") or {}).items()} or dict(DEFAULT_COSTS)
    defs, foes = [], []
    for d in state.get("defences") or []:
        if d.get("health", 0) > 0 and "position" in d:
            x, y = _xy(d)
            defs.append({"k": DEF_IDS.get(d.get("type"), SHIELD), "x": x, "y": y, "hp": d["health"],
                         "max": d.get("maxHealth") or d["health"]})
    for e in state.get("enemies") or []:
        if not e.get("isDead") and e.get("health", 0) > 0 and "position" in e:
            x, y = _xy(e)
            foes.append({"k": ENEMY_IDS.get(e.get("type"), CLOWN), "x": x, "y": y, "hp": e["health"]})
    counts = state.get("counts") or {}
    gs = state.get("gameSettings") or {}
    return {
        "phase": state.get("phase", "pause"), "costs": costs,
        "coins": float((state.get("budget") or {}).get("remaining", 0)),
        "defs": defs, "foes": foes, "occ": {(d["x"], d["y"]) for d in defs},
        "queued": int(counts.get("enemiesQueued", 0) or 0),
        "total": int(counts.get("totalEnemiesInWave", 0) or 0),
        "reinf": gs.get("reinforcementsEnabled", True),
        "threats": row_threats(state) if state.get("phase") == "start" else {},
        "budget_s": float(state.get("timeBudgetSeconds") or 15),
        "floor": coin_floor(state),
        "t": state.get("elapsedSeconds"),
    }


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


def blast_zone(x, y):
    """Tiles a pirate exploding at defence (x, y) damages: radius 2 u, the pirate stands just ahead."""
    return {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x + 1, y - 1), (x + 1, y + 1)}


def choose_shield(y, occ, shields, before=MAX_X + 1):
    """Blocker tile for row y: x=4 by default, so a pirate blast there misses soldier@1 and archer@0;
    never past x=6 (clowns would stop off the board where nothing sees or hits them). Avoids putting
    our units inside another shield's blast; accepts a shared blast rather than an open row."""
    danger = set().union(*(blast_zone(*s) for s in shields)) if shields else set()
    sx, top = T["shield_x"], min(T["shield_x_max"], before - 1)
    prefs = [x for x in (sx, sx + 2, sx + 1, sx - 1, top, 2) if 2 <= x <= top]
    for x in prefs:
        if (x, y) not in occ and (x, y) not in danger and not (blast_zone(x, y) & occ):
            return x
    return next((x for x in prefs + [1, 0] if x < before and (x, y) not in occ), None)


def coin_floor(state):
    """Coins we lean towards keeping: 500, or 20% of the starting coins when those are under 2000."""
    budget, counts = state.get("budget") or {}, state.get("counts") or {}
    kills = max(0, int(counts.get("totalEnemiesInWave", 0) or 0) - int(counts.get("enemiesQueued", 0) or 0)
                - int(counts.get("activeEnemyCount", 0) or 0))
    start = float(budget.get("unlocked", budget.get("remaining", 0)) or 0) - 50 * kills
    return T["floor_abs"] if start >= T["floor_small_start"] else T["floor_frac"] * max(0.0, start)


def keep_floor(cards, coins, floor, costs):
    """Non-critical cards, in order, as long as the balance stays at or above the floor."""
    out, left = [], coins
    for c in cards:
        price = costs.get(c["defenceId"], 999)
        if left - price >= floor:
            out.append(c); left -= price
    return out


# --- options: opening -------------------------------------------------------
def opening_options(B):
    threats, costs, coins, floor = B["threats"], B["costs"], B["coins"], B["floor"]
    sx = T["soldier_x"]
    order = sorted(threats, key=lambda y: (PIRATE not in threats[y], MONK not in threats[y],
                                           WIZARD not in threats[y], y))
    occ, shields = set(B["occ"]), []
    first, second = [], []
    for y in order:
        t = threats[y]
        others = t.get(CLOWN, 0) + t.get(MONK, 0) + t.get(PIRATE, 0)
        if t.get(WIZARD, 0) > others:  # wizards dominate: their beam triggers the penguin's slide
            row = [card(PENGUIN, T["penguin_x"], y), card(SOLDIER, sx, y)]
            occ |= {(T["penguin_x"], y), (sx, y)}
        else:
            bx = choose_shield(y, occ | {(sx, y)}, shields)
            block = [card(SHIELD, bx, y)] if bx is not None else []
            # pirates and monks reach the line first: the block comes before the soldier
            row = block + [card(SOLDIER, sx, y)] if (PIRATE in t or MONK in t) else [card(SOLDIER, sx, y)] + block
            occ.add((sx, y))
            if bx is not None:
                occ.add((bx, y)); shields.append((bx, y))
        first.append(row[0]); second += row[1:]
    extra = []
    if not B["reinf"]:  # no second chance: a second penguin where more wizards come
        for y in order:
            if threats[y].get(WIZARD, 0) >= 1.5 and (2, y) not in occ:
                extra.append(card(PENGUIN, 2, y)); occ.add((2, y))
    core = first + second + extra
    critical = first + second

    # Archers @0: quiet rows first (unreachable), then rows whose line is built in front.
    quiet = [y for y in ROWS if y not in threats]
    lined = [y for y in order if (sx, y) in occ and any((x, y) in occ for x in range(2, MAX_X + 1))]
    arch = [card(ARCHER, 0, y) for y in (quiet + lined) if (0, y) not in occ][: T["max_archers"]]
    left = coins - cost_of(core, costs)
    fl = floor if B["reinf"] else 0
    opts = {
        "fortress": (core + keep_floor(arch, left, fl, costs),
                     f"line in every threatened row + every protected archer the coins allow above the {fl:.0f} floor"),
        "balanced": (core + keep_floor(arch[: T["balanced_archers"]], left, fl + T["balanced_extra"], costs),
                     f"line + up to {T['balanced_archers']} archers, keeps ~{fl + T['balanced_extra']:.0f} coins"),
        "lean": (core, "only the line (shield/penguin + soldier) per threatened row"),
    }
    return opts, "fortress", critical


# --- options: pauses --------------------------------------------------------
def pause_options(B):
    costs, sx, floor = B["costs"], T["soldier_x"], B["floor"]
    occ = set(B["occ"])
    shields = [(d["x"], d["y"]) for d in B["defs"] if d["k"] == SHIELD and d["x"] >= 2]
    critical, sentinels, extra, why = [], [], [], []
    for y in ROWS:
        foes = [f for f in B["foes"] if f["y"] == y]
        mine = sorted((d for d in B["defs"] if d["y"] == y), key=lambda d: d["x"])
        blockers = [d for d in mine if d["k"] != ARCHER]
        kinds = {d["k"] for d in mine}
        nearest = min((f["x"] for f in foes), default=MAX_X + 1)
        types = {f["k"] for f in foes}
        walkers = [f for f in foes if f["k"] != PIRATE]

        def free(x):
            return 0 <= x < nearest and (x, y) not in occ

        def put(lst, kind, x, note):
            lst.append(card(kind, x, y)); occ.add((x, y)); why.append(f"r{y} {note}")
            if kind == SHIELD:
                shields.append((x, y))

        if not foes:
            if not mine:  # open row: a pirate could spawn and cross before the next pause
                bx = choose_shield(y, occ, shields)
                if bx is not None:
                    put(sentinels, SHIELD, bx, "open row->blocker")
            elif blockers and max(d["x"] for d in blockers) <= sx:  # soldier alone: one pirate kills it
                bx = choose_shield(y, occ, shields)
                if bx is not None:
                    put(extra, SHIELD, bx, "shield the soldier")
            continue
        if not blockers:  # breach: nothing stops this row (an archer alone is no wall)
            px = next((x for x in (2, 3, 1, 0) if free(x)), None)
            if px is not None:
                put(critical, PENGUIN, px, "BREACH->penguin")
            if nearest > sx + 1 and free(sx):
                put(critical, SOLDIER, sx, "BREACH->soldier")
            continue
        front = max(d["x"] for d in blockers)
        if front <= sx:  # nothing in front of the soldier: block it so pirates/monks stop out there
            bx = choose_shield(y, occ, shields, before=nearest)
            if bx is not None and bx > sx and free(bx):
                put(critical, SHIELD, bx, "block in front")
                front = bx
        if not kinds & {SOLDIER, ARCHER, PENGUIN} and free(sx):
            put(critical, SOLDIER, sx, "no killer")
        # Penguins only where the row can't win on its own: stalled by regen, a wizard,
        # a monk pair, or a worn front under 2+ attackers. A lone clown is the soldier's job.
        info = lane(B, y)
        fd = max(blockers, key=lambda d: d["x"])
        worn = fd["hp"] < T["worn_frac"] * fd["max"]
        monks = sum(1 for f in foes if f["k"] == MONK)
        if PENGUIN not in kinds and walkers and (
                info["dps"] <= info["regen"] or WIZARD in types or monks >= 2 or (worn and len(walkers) >= 2)):
            px = next((x for x in range(front + 1, min(front + 3, T["shield_x_max"] + 1)) if free(x)), None)
            if px is not None:
                put(extra, PENGUIN, px, f"penguin ({info['verdict'][:20]})")
    critical += sentinels  # rows under attack first, then blockers for open rows
    coins_left = B["coins"] - cost_of(critical, costs)
    patch = critical + keep_floor(extra, coins_left, floor, costs)

    # Lasting defences: archers @0 behind a built line (soldier @1 + something in front).
    n_arch = sum(1 for d in B["defs"] if d["k"] == ARCHER)
    lined = [y for y in ROWS if (sx, y) in occ and any((x, y) in occ for x in range(2, MAX_X + 1))
             and (0, y) not in occ]
    arch = [card(ARCHER, 0, y) for y in lined][: max(0, T["max_archers"] - n_arch)]
    reinforce = patch + keep_floor(arch, B["coins"] - cost_of(patch, costs), floor, costs)

    early = B["total"] and B["queued"] >= T["early_frac"] * B["total"]
    opts = {
        "reinforce": (reinforce, "patch + archers behind built lines (lasting DPS for the rest of the wave)"),
        "patch": (patch, "only what the board needs now: " + (", ".join(why) or "nothing")),
        "hold": (critical, "only critical blockers, keep coins"),
    }
    default = "reinforce" if early and len(reinforce) > len(patch) else ("patch" if patch else "hold")
    return opts, default, critical


# --- lane analysis (tool + prompt) ------------------------------------------
def lane(B, y, extra=()):
    foes = sorted((f for f in B["foes"] if f["y"] == y), key=lambda f: f["x"])
    mine = sorted([d for d in B["defs"] if d["y"] == y]
                  + [{"k": k, "x": x, "y": yy, "hp": 1, "max": 1} for k, x, yy in extra if yy == y],
                  key=lambda d: d["x"])
    rows_hit = [f["y"] for f in B["foes"]]
    archers = sum(1 for d in B["defs"] if d["k"] == ARCHER) + sum(1 for k, *_ in extra if k == ARCHER)
    dps = sum(DPS[SOLDIER] for d in mine if d["k"] == SOLDIER
              and any(f["x"] >= d["x"] for f in foes))  # unlimited range ahead
    if rows_hit and min(rows_hit) == y:
        dps += archers * DPS[ARCHER]  # all archers focus the lowest row with enemies
    regen = sum(REGEN.get(f["k"], 0) for f in foes)
    hp = sum(f["hp"] for f in foes)
    if not foes:
        verdict = "quiet"
    elif not mine:
        verdict = "BREACH: no defence in this row"
    elif dps <= regen:
        verdict = f"STALLED: regen {regen}/s >= our dps {dps}"
    else:
        verdict = f"holding: ~{hp / (dps - regen):.0f}s to clear"
    out = {"row": y, "enemies": [[FOE_NAME[f["k"]], round(f["hp"]), f["x"]] for f in foes],
           "defences": [[DEF_NAME[d["k"]], round(d["hp"]), d["x"]] for d in mine],
           "dps": dps, "regen": regen, "verdict": verdict}
    if B["threats"].get(y):
        out["expected"] = {FOE_NAME[k]: round(v, 1) for k, v in B["threats"][y].items()}
    return out


def plan_warnings(B, cards):
    occ = set(B["occ"]) | {(c["defenceGridPosition"]["x"], c["defenceGridPosition"]["y"]) for c in cards}
    warn = []
    for c in cards:
        k, x, y = _kxy(c)
        row = [xx for (xx, yy) in occ if yy == y and xx != x]
        foes = [f["x"] for f in B["foes"] if f["y"] == y]
        if foes and x >= min(foes):
            warn.append(f"{DEF_NAME[k]}@({x},{y}) is at/behind an enemy")
        if k == ARCHER and (B["threats"].get(y) or foes) and not any(xx > x for xx in row):
            warn.append(f"archer@({x},{y}) has nothing in front of it")
        if k == SHIELD and x > T["shield_x_max"]:
            warn.append(f"shield@({x},{y}) past x={T['shield_x_max']}: clowns would stop off the board")
    return warn


# --- LLM decision -----------------------------------------------------------
PROMPT = """You command a lane-defence game. Grid x 0-10 (base at x=0; enemies enter at x=10 and walk left), rows y 0-5. Enemies never change rows. ONE enemy crossing x=0 = LOSS.
Defences (id: name cost hp):
0 penguin 250 200hp: idle until hit, then slides forward through its row dealing 500 to EVERY enemy there, then is gone. Best vs stacks and wizards.
1 soldier 200 150hp: ~40 dps (10 per shot every 0.25 s), hits the first enemy anywhere ahead of it in its own row; ignores enemies behind it.
2 archer 300 100hp: ~26 dps into ANY row (all archers focus the lowest row with enemies, nearest to base). Fragile: keep at x=0 behind a soldier+shield, or in a quiet row. Lasting value all game.
3 shield 150 700hp: no attack; enemies stop at it and their shots hit it first.
Enemies: clown 200hp +15/s regen, stops ~3 tiles from the first defence and throws (~17 dps). monk 280hp, 3.5 u/s (crosses in ~4.3 s), melee ~36 dps. wizard 500hp +15/s regen, stops ~5 tiles away, beam 50 dps. pirate 30hp, 8 u/s (crosses in ~1.9 s), explodes for 200 on the first defence it reaches (or where it dies), radius 2 units = 1.3 tiles: same tile, x+-1 in its row, and the tiles above/below.
1 tile = 1.5 units.
Tactics: the line in a row is soldier@1 + shield@~4 (pirates blow up on the shield, far from the soldier) with archers@0 behind. Keep EVERY row obstructed: a pirate crosses the board faster than one pause. Shields up to x=6 only: further out, clowns stop off the board where nothing sees or hits them. Penguins are single-use: only where a row cannot win alone (enemy regen >= our dps there, a wizard, 2+ monks, a worn front under 2+ attackers); a lone clown is the soldier's job. Opening and pauses are frozen time; pauses come every reinforcementIntervalSeconds of game time.
Score of a WON game /100: win 25, kills 25, defences still standing 25, speed 10, coins left 15. A LOST game scores only kills. So never risk a loss; early in the wave build lasting defences; late, when the line holds, keep coins. Keep at least coin_floor coins unless a breach forces you to spend them (they pay for last-minute fixes).
You get rule-based options with costs. Tools: lane_report(row) for one row's detail, check_plan(placements) to test your own placements.
Answer with JSON only: {"option": "<name>", "reason": "<=12 words"} or {"placements": [[defenceId, x, y], ...], "reason": "..."}"""


def llm_decide(B, opts, default):
    if not USE_LLM or LLMClient is None:
        return None, "llm off"
    t0 = time.monotonic()
    deadline = min(T["llm_deadline"], B["budget_s"] - 3)
    if deadline < 1:
        return None, "no time"

    @tool
    def lane_report(row: int) -> dict:
        """One row in detail: enemies and defences as [name, hp, x], our dps vs their regen, a verdict.

        Args:
            row: Board row, 0-5.
        """
        return lane(B, int(row))

    @tool
    def check_plan(placements: list) -> dict:
        """Validate placements [[defenceId, x, y], ...]: accepted cards, cost, coins left, warnings.

        Args:
            placements: list of [defenceId, x, y].
        """
        cards = _from_triples(placements)
        ok = check(cards, set(B["occ"]), B["coins"], B["costs"])
        return {"accepted": [list(_kxy(c)) for c in ok], "rejected": len(cards) - len(ok),
                "cost": cost_of(ok, B["costs"]), "coins_left": B["coins"] - cost_of(ok, B["costs"]),
                "warnings": plan_warnings(B, ok)}

    board = {
        "phase": B["phase"], "game_time": B["t"], "coins": B["coins"], "coin_floor": B["floor"], "costs": B["costs"],
        "enemies_still_to_spawn": B["queued"], "wave_size": B["total"],
        "reinforcements_later": bool(B["reinf"]),
        "rows": [lane(B, y) for y in ROWS],
        "options": {n: {"cost": cost_of(c, B["costs"]), "cards": [list(_kxy(x)) for x in c], "what": w}
                    for n, (c, w) in opts.items()},
        "rules_recommend": default,
    }
    run = run_tool_loop(
        LLMClient(),
        [{"role": "system", "content": PROMPT}, {"role": "user", "content": json.dumps(board, separators=(",", ":"))}],
        [lane_report, check_plan],
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
        cards = opts[answer["option"]][0]
        src = f"llm option={answer['option']}"
    elif isinstance(answer.get("placements"), list):
        cards = _from_triples(answer["placements"])
        src = "llm custom"
    else:
        return None, f"llm unusable answer: {text[:80]}"
    return cards, f"{src} ({reason}) tools={tools_used} {time.monotonic() - t0:.1f}s"


def _from_triples(raw):
    cards = []
    for p in raw or []:
        try:
            if isinstance(p, dict):
                k, x, y = p.get("defenceId"), p.get("x", (p.get("defenceGridPosition") or {}).get("x")), \
                    p.get("y", (p.get("defenceGridPosition") or {}).get("y"))
            else:
                k, x, y = p
            cards.append(card(int(k), int(x), int(y)))
        except (TypeError, ValueError):
            continue
    return cards


# --- entry point ------------------------------------------------------------
def get_reinforcements(state):
    try:
        B = analyse(state)
        if B["phase"] == "start":
            try:
                log("STATE " + json.dumps(state, separators=(",", ":")))
            except Exception:
                pass
            opts, default, critical = opening_options(B)
        else:
            opts, default, critical = pause_options(B)
        log(f"{B['phase']} t={B['t']} coins={B['coins']:.0f} queued={B['queued']}/{B['total']} "
            + " ".join(f"{n}={cost_of(c, B['costs'])}" for n, (c, _) in opts.items()) + f" default={default}")
        if B["phase"] != "start" and not B["foes"] and not opts[default][0]:
            chosen, src = [], "nothing on the board"
        else:
            try:
                chosen, src = llm_decide(B, opts, default)
            except Exception as exc:
                chosen, src = None, f"llm failed: {type(exc).__name__}: {str(exc)[:80]}"
            if chosen is None:
                chosen, src = opts[default][0], f"rules {default} ({src})"
            touched = {_kxy(c)[2] for c in chosen}
            chosen = chosen + [c for c in critical if _kxy(c)[2] not in touched]  # safety net
        out = check(chosen, set(B["occ"]), B["coins"], B["costs"])
        log(f"{src} -> placing {[_kxy(c) for c in out]}")
        return out
    except Exception as exc:
        log(f"ERROR {type(exc).__name__}: {exc}")
        return []
