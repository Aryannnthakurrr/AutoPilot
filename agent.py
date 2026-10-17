"""agent.py that lets an LLM plan the defence, with one small tool.

The LLM sees a short board summary and can call one tool defined below (it runs
here, inside your agent): the full picture of one row. Then it answers with JSON.
Its answer is checked again (on the board, free cell, affordable); if the LLM is
slow, fails or proposes nothing usable, a simple rule takes over.
Run it with the play command in README_PARTICIPANT.md (step 3).
"""
import json
import sys

from autopilot_sdk.llm import LLMClient
from autopilot_sdk.tools import run_tool_loop, tool

PROMPT = (
    "You defend a base in a lane game. Board: x 0-10, y 0-5. Enemies walk from x=10 "
    "toward x=0 along their row y; one reaching x=0 loses the game. Defences fight "
    "enemies in their own row (archers hit any row). Units (defenceId: cost):\n"
    "0 penguin 250: rams enemies for 500 damage, can ram again. Good against wizards.\n"
    "1 soldier 200: steady damage. Put one at x=1 in every row with enemies.\n"
    "2 archer 300: highest damage, fragile. Put it at x=0.\n"
    "3 shield 150: no damage. Bait at x=5 in rows with pirates (they explode).\n"
    "Never leave a row with enemies undefended. Placements together must cost at most "
    "coins. Answer with JSON only, no other text: "
    '{"placements": [{"defenceId": 2, "x": 0, "y": 3}], "reason": "<at most 8 words>"}'
)


def log(message):
    print(f"[agent] {message}", file=sys.stderr, flush=True)  # shows in the play window


def _xy(unit):
    return round(unit["position"]["x"]), round(unit["position"]["y"])


def get_reinforcements(state):
    coins = state["budget"]["remaining"]
    costs = {int(k): v for k, v in state["unitCosts"].items()}
    defences = [d for d in state["defences"] if d["health"] > 0]
    enemies = [e for e in state["enemies"] if not e["isDead"] and e["health"] > 0]
    taken = {_xy(d) for d in defences}
    if state["phase"] == "start":
        rows = sorted({r for e in state["upcomingEnemies"] for r in e.get("allowedRows") or range(6)})
    else:
        rows = sorted({_xy(e)[1] for e in enemies})
    if coins < min(costs.values()) or not rows:
        return []

    @tool
    def lane_report(row: int) -> dict:
        """Everything in one row: enemies and defences as [type, health, x], and free columns.

        Args:
            row: Board row, 0-5.
        """
        return {
            "enemies": [[e["type"], round(e["health"]), _xy(e)[0]] for e in enemies if _xy(e)[1] == row],
            "defences": [[d["type"], round(d["health"]), _xy(d)[0]] for d in defences if _xy(d)[1] == row],
            "free_columns": [x for x in range(7) if (x, row) not in taken],
        }

    board = {
        "phase": state["phase"], "coins": coins, "unitCosts": costs,
        "enemies": [[e["type"], round(e["health"]), *_xy(e)] for e in enemies],
        "upcomingEnemies": state.get("upcomingEnemies", []),
        "enemiesStillToCome": state["counts"]["enemiesQueued"],
        "defences": [[d["type"], round(d["health"]), *_xy(d)] for d in defences],
        "rowsToDefend": rows,
    }
    try:
        run = run_tool_loop(
            LLMClient(),
            [{"role": "system", "content": PROMPT}, {"role": "user", "content": json.dumps(board)}],
            [lane_report],
            deadline_s=state["timeBudgetSeconds"] - 1,   # keep a second to return the answer
            max_rounds=1 if state["phase"] == "start" else 2,  # at the start, answer at once
            max_tokens=250,
        )
        answer = json.loads((run.content or "").strip().strip("`").removeprefix("json"))
        placements = check(answer.get("placements", []), taken, coins, costs)
        if placements:
            log(f"LLM: {placements} ({str(answer.get('reason', ''))[:80]})")
            return placements
    except Exception as exc:  # slow, unreachable or unreadable: use the rule below
        log(f"LLM not used: {type(exc).__name__}: {str(exc)[:100]}")
    return fallback(rows, taken, coins, costs)


def check(raw, taken, coins, costs):
    """Keep the placements that are on the board, on a free cell and affordable."""
    chosen = []
    for p in raw[:8]:
        try:
            kind, x, y = int(p["defenceId"]), int(p["x"]), int(p["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if kind in costs and 0 <= x <= 10 and 0 <= y <= 5 and (x, y) not in taken and costs[kind] <= coins:
            taken.add((x, y))
            coins -= costs[kind]
            chosen.append({"defenceId": kind, "defenceGridPosition": {"x": x, "y": y}})
    return chosen


def fallback(rows, taken, coins, costs):
    """A soldier at x=1 in each threatened row, as far as the coins go."""
    chosen = []
    for y in rows:
        if costs[1] <= coins and (1, y) not in taken:
            taken.add((1, y))
            chosen.append({"defenceId": 1, "defenceGridPosition": {"x": 1, "y": y}})
            coins -= costs[1]
    return chosen
