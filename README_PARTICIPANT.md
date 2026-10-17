# Autopilot: participant guide

You write one Python file, `agent.py`, that decides where to place defences.
The game runs in your browser; the SDK runs your agent on your laptop.

| | |
|---|---|
| Downloads | https://autopilot.gdgnsut.com/download/ |
| Player guide | https://autopilot.gdgnsut.com/guide/event |
| Game | https://autopilot.gdgnsut.com/ or https://13-200-183-134.sslip.io/ (Chrome or Firefox, not Safari) |
| Leaderboard | https://autopilot.gdgnsut.com/leaderboard/ |

## 1. Set up (5 minutes)

1. **Your login:** the organisers send your team a 4-digit **team ID** and a **password**.
2. **Install and start with one line** (it puts everything in a folder
   `autopilot` in your home folder and starts the SDK):

   macOS or Linux, in Terminal:
   ```bash
   curl -fsSL https://autopilot.gdgnsut.com/install.sh | sh
   ```
   Windows, in PowerShell (Start menu → type "PowerShell"):
   ```powershell
   irm https://autopilot.gdgnsut.com/install.ps1 | iex
   ```

   It downloads the SDK program for your computer and a starter `agent.py` (it
   never overwrites your existing agent files). On a **Mac**, and on any computer no
   program fits, it installs the SDK with Python 3.10+ in a virtual environment
   (a Mac's built-in `python3` is too old: install Python from python.org first).
   Run the same line again to reinstall.

   **The SDK updates itself.** When the organisers release a new SDK, yours
   downloads it, restarts and carries on (still logged in, the same PLAY or
   PUBLISH run). An SDK installed before the automatic updates (before 17 Oct)
   is refused by the server: run the install line above once more.
   **Next time:** double-click the program in `autopilot` (on Windows without the
   program: `autopilot.cmd`), or run the line again.

   <details><summary>Or download it by hand</summary>

   From https://autopilot.gdgnsut.com/download/ download the program for your
   computer (`autopilot-windows-x86_64.exe` or `autopilot-linux-x86_64`). Put it
   in a folder with `agent.py` (download that from the same page). Double-click
   the program on Windows. On Linux, run `chmod +x autopilot-linux-x86_64`
   once, then `./autopilot-linux-x86_64`. On a Mac, use the one-line installer.
   The first time only:
   - Windows: if SmartScreen warns, click **More info → Run anyway**.
   - Mac: macOS blocks it ("not verified"). **System Settings → Privacy &
     Security** → scroll down → **Open Anyway**, then double-click it again.
     (The one-line install avoids this.)
   </details>

3. **With Python instead** (any computer with Python 3.10+; this also makes
   `autopilot_sdk` imports work in VS Code):
   ```bash
   python3 -m venv .venv                  # Windows: py -3 -m venv .venv
   source .venv/bin/activate              # Windows: .venv\Scripts\activate
   pip install https://autopilot.gdgnsut.com/download/autopilot_sdk-0.3.0-py3-none-any.whl
   autopilot                              # starts the SDK, same as the program
   ```
   In VS Code pick that interpreter: Ctrl+Shift+P → **Python: Select
   Interpreter** → `.venv`. (Using only the program? Run just the `pip install`
   line, in the Python VS Code uses, so the editor knows `autopilot_sdk`.)

## 2. Play a game

1. **Start the SDK** (the one-line install starts it; later, double-click). It asks two things:
   - **Path to your agent.py:** choose `agent.py` (filled in for a new install).
     If you already have your own agent, keep selecting that file.
   - **Your team ID and password.**
   If `agent.py` has a mistake, the SDK says which line and what is wrong, and
   waits: fix the file and press Enter. It never closes on a mistake.
2. **Pick from the menu:**
   ```
   1  PLAY             practice one public test case (not on the leaderboard)
   2  PUBLISH PUBLIC   every public test case once: your total goes on the live leaderboard
   3  PUBLISH PRIVATE  every hidden test case once: the final leaderboard (opens later)
   4  VERIFY           check agent.py and try it on a sample board
   A  agent.py  ·  L  switch team  ·  Q  quit
   ```
   Start with **4 VERIFY**, then **1 PLAY** with `TC01`. The SDK says **READY**.
3. **Open the game** (https://13-200-183-134.sslip.io/), type your team ID and
   press ▶. (You are already logged in through the SDK.)
4. **Chrome or Edge asks** whether the site may "access other apps and services
   on this device" (top left, after ▶): click **Allow**. That is the game
   reaching the SDK. Clicked Block? Click the icon left of the address, set
   that permission to Allow, reload the page.

Every pause shows in the SDK window: what your agent proposed, and what the
referee accepted or refused, and why. At the end you see your score.

- **Press ▶ again** to play the next game.
- **Ctrl+C** in the SDK returns to the menu, to switch test case or mode. You
  never need to restart it.
- **Edits to `agent.py` apply from the next pause.** No restart either: the
  SDK prints `↻ agent.py changed` when the new version plays. It runs the
  `agent.py` whose path the SDK shows at the top, so edit that one. (PUBLISH is
  the exception: it keeps the copy it started with.)

## 3. Write `agent.py`

Your file needs one function, `get_reinforcements(state)`: it gets the board
and returns the defences to place. The starter `agent.py` (in Downloads) lets
an **LLM** plan, with one **tool** it can call, and falls back to a simple rule
if the LLM is slow or wrong:

```python
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

```

**The LLM:** `LLMClient()` is the organisers' LLM, called the standard
(OpenAI chat completions) way. No key or URL to set: it comes with your login.
60 calls a minute per team.

```python
from autopilot_sdk.llm import LLMClient

reply = LLMClient().chat.completions.create(messages=[{"role": "user", "content": "..."}], timeout=5)
text = reply.choices[0].message.content
```

**Tools:** mark a function with `@tool` (type hints and an `Args:` docstring
describe it to the LLM) and pass it to `run_tool_loop`. The function runs on
your laptop when the LLM asks for it. `agent.py` in Downloads has one tool, `lane_report`.

**The board:** 11 columns (`x` 0–10) by 6 rows (`y` 0–5). Enemies walk from the
right toward `x = 0`; if one gets there, you lose. Defences fight enemies in
their own row.

| `defenceId` | Unit | Cost | Health | Role |
|---|---|---|---|---|
| 0 | Penguin | 250 | 200 | rams the first enemy in its row (about 500 damage), then is gone |
| 1 | Soldier | 200 | 150 | steady damage, cheap all-rounder |
| 2 | Archer | 300 | 100 | highest damage, fragile: put it behind a shield |
| 3 | Shield | 150 | 700 | no attack, blocks |

Enemies: Clown (weak), Monk (tough), Wizard (500 HP), Pirate (fast, hits hardest).

**Coins:** each test case sets its starting coins (`state["budget"]`; 1300 to
3900 on the public ones), plus 50 per kill. The referee accepts your
placements in the order you return them until the coins run out, and drops
cells that are taken or off the board (it never moves them).

**When it is called:** the updated Unity player spawns the first enemy and
freezes before it moves. The opening (`phase == "start"`) includes that enemy’s
actual position and health, plus the whole wave in `upcomingEnemies`. The game
places approved defences before resuming. This opening also runs on levels with
reinforcements disabled; those levels get no later calls. Other levels call the
agent at every pause (`phase == "pause"`), every
`reinforcementIntervalSeconds` of game time (each test case sets it; the SDK
prints it when a game starts). You have **15 seconds** to answer
(`timeBudgetSeconds`); the game waits.

**Random on public, fixed on hidden.** On the public test cases (PLAY and
PUBLISH PUBLIC) enemies spawn in random rows, so the same `agent.py` can score
differently from game to game: make it handle any row. On the hidden test cases
(PUBLISH PRIVATE) every game spawns the same enemies in the same rows and order,
so your final score depends on your `agent.py`, not luck. The game runs at a
fixed step (no fast-forward) and the LLM answers at temperature 0.

**What `state` holds:**

| Key | Contents |
|---|---|
| `defences`, `enemies` | `type`, `health`, `maxHealth`, `position` {`x`, `y`}; enemies also `isDead`, `isAttacking` |
| `upcomingEnemies` | at the start: `enemyId`, `count`, `allowedRows` |
| `counts` | `activeEnemyCount`, `enemiesQueued` (still to come), `totalEnemiesInWave` |
| `budget` | `remaining`, `spent`, `unlocked`, `total` |
| `unitCosts` | cost of each `defenceId` |
| `phase`, `timeBudgetSeconds` | `"start"` or `"pause"`; 15 |

Return `[]` to place nothing. If your code crashes or is too slow, that pause
places nothing and the game goes on.

**Rules:** the Python standard library and `autopilot_sdk` only. No network
except the LLM, and no starting other programs. **4 VERIFY** checks this.

## 4. Score and the leaderboards

**A won game scores out of 100:** win 25 · kills up to 25 · defences still
standing up to 25 · speed up to 10 · coins left up to 15.
**A lost game scores only its kills** (up to 25); everything else is 0.

**PLAY is practice.** Its games are saved in your run log but never reach a
leaderboard.

**To get on a leaderboard, publish a run.** One frozen `agent.py` plays every
test case of the board once; your score is the **sum** of those games. **Only
complete runs count:** each test case starts once per run, so if you reload the
page or quit a game midway, that run can't count and you publish again.
- **2 PUBLISH PUBLIC:** all public test cases → the live leaderboard
  (https://13-200-183-134.sslip.io/leaderboard/). Publish as often as you like:
  your best run counts.
- **3 PUBLISH PRIVATE:** all hidden test cases → the final leaderboard. It opens
  when the organisers say so, and each team plays it **once**, with one
  `agent.py`: if the SDK stops mid-run, start it again with the same `agent.py`
  and the run continues after the cases already played (a game closed midway
  can't be replayed: ask the organisers). The results are revealed at the end.
  Single hidden test cases can't be played.
- Log in first. The SDK freezes `agent.py` when the run starts; edits during it don't count.
- Press ▶ for each test case and let every game finish. Your score, your
  `agent.py` and the run log go to the organisers.
- Hidden cases differ from the public ones, so don't tune only to the public ones.
- The backend checks every game's report against what it saw (time played,
  pauses, the referee's placements); a report that can't be true scores 0. The
  organisers re-run the top teams' `agent.py` before announcing winners.

**Run logs:** every run is saved in `autopilot-runs/` (each pause, your agent's
answer, the referee's decision, every version of `agent.py`). Keep them: the
organisers may ask the winners for them.

## Troubleshooting

| You see | Do this |
|---|---|
| Nothing happens after ▶ | Does the SDK say **READY**? Did you click **Allow** in the browser's popup (top left)? |
| `waiting for the organisers' approval` | Your team is not approved yet. |
| `Wrong team ID or password` | Check both; the organisers can reset the password. |
| `The backend refused this game` | Read the reason after it. To publish, log in first. |
| `LLM call failed: HTTP 401` | Log in: the LLM token comes with the login. |
| VS Code: `Import "autopilot_sdk…" could not be resolved` | Step 1.3: install the SDK with Python and select that interpreter. |
| `LLM call failed: HTTP 429` | Over 60 LLM calls a minute. Make fewer. |
| `No placements this pause: …` | Your agent crashed or was too slow; the reason is printed. |
| `Port 5000 is in use` | Another copy of the SDK is running. Close it. |
| Mac: "cannot be opened" | Use the one-line install (step 1.2), or System Settings → Privacy & Security → **Open Anyway**. |

**Without the menu** (for scripts):

```bash
./autopilot-linux-x86_64 verify --agent agent.py
./autopilot-linux-x86_64 play --agent agent.py --test-case TC03     # or --all
./autopilot-linux-x86_64 eval --board public --agent agent.py --team 0042   # PUBLISH PUBLIC; asks for the password
./autopilot-linux-x86_64 eval --board private --agent agent.py --team 0042  # PUBLISH PRIVATE (once it is open)
```
Add `--backend https://13-200-183-134.sslip.io/backend` if the SDK uses another server.
