"""
scheduler.py — Automated Draw-Day Simulation
Runs every Monday, Wednesday, Saturday at 8:00 AM.
Simulates 500 draws, picks 3 best combinations, saves to generated_powerball_numbers.
"""

import os
import random
import requests
import json
from datetime import datetime
from collections import defaultdict
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

# ── Timezone: US Eastern (Powerball draw timezone) ──────────
DRAW_TZ = pytz.timezone('US/Eastern')

# ── How many draws to simulate each run ─────────────────────
SIM_DRAW_COUNT = 500

# ── How many combinations to save per run ───────────────────
COMBOS_TO_SAVE = 3


def run_draw_day_simulation():
    """
    Core job: simulate draws, build smart combos, save to Supabase.
    Imported lazily to avoid circular import with index.py.
    """
    # Lazy imports from the main app module
    try:
        from api.index import (
            df,
            simulate_multiple_draws,
            save_generated_numbers_to_db,
            group_a,
            GLOBAL_WHITE_BALL_RANGE,
            GLOBAL_POWERBALL_RANGE,
        )
    except ImportError:
        # Fallback path when running directly
        from index import (
            df,
            simulate_multiple_draws,
            save_generated_numbers_to_db,
            group_a,
            GLOBAL_WHITE_BALL_RANGE,
            GLOBAL_POWERBALL_RANGE,
        )

    now = datetime.now(DRAW_TZ)
    day = now.strftime('%A')
    log(f"🎰 Draw-day simulation starting ({day} {now.strftime('%Y-%m-%d %H:%M %Z')})")

    if df.empty:
        log("⚠️  Historical data not loaded — aborting simulation.", error=True)
        return

    # ── Step 1: Run batch simulation ────────────────────────
    sim_results = simulate_multiple_draws(
        df_source=df,
        group_a_list=group_a,
        odd_even_choice='Any',
        white_ball_range=GLOBAL_WHITE_BALL_RANGE,
        powerball_range=GLOBAL_POWERBALL_RANGE,
        excluded_numbers=[],
        num_draws=SIM_DRAW_COUNT
    )

    wb_freq = sim_results.get('white_ball_freq', [])
    pb_freq = sim_results.get('powerball_freq', [])

    if not wb_freq or not pb_freq:
        log("⚠️  Simulation returned no data — aborting.", error=True)
        return

    log(f"✅ Simulation complete: {SIM_DRAW_COUNT} draws processed.")

    # ── Step 2: Build ranked pools ───────────────────────────
    # Sort white balls by simulated frequency (desc), then number (asc) for ties
    wb_ranked = sorted(wb_freq, key=lambda x: (-x['Frequency'], x['Number']))
    pb_ranked = sorted(pb_freq, key=lambda x: (-x['Frequency'], x['Number']))

    # Build decade-balanced pool from top 20 white balls
    top_wb_pool = [d['Number'] for d in wb_ranked[:20]]
    top_pb      = pb_ranked[0]['Number']  # single best powerball

    # ── Step 3: Generate 3 smart combinations ────────────────
    combos = _build_balanced_combos(top_wb_pool, top_pb, wb_ranked, count=COMBOS_TO_SAVE)

    # ── Step 4: Save each combo to Supabase ─────────────────
    saved = 0
    for wb, pb in combos:
        success, msg = save_generated_numbers_to_db(wb, pb, source='scheduler')
        status = "✅ Saved" if success else f"⚠️  Skip ({msg})"
        log(f"  {status}: {wb} + PB {pb}")
        if success:
            saved += 1

    log(f"🏁 Done — {saved}/{len(combos)} combos saved for {day} draw.")


def _build_balanced_combos(pool, top_pb, wb_ranked, count=3):
    """
    Build `count` combinations from the high-frequency pool.
    Each combo:
      - 5 unique white balls spread across at least 3 decades
      - Uses weighted random sampling (higher freq = more likely)
      - Powerball = top simulated PB (varied slightly for combo 2 & 3)
    """
    pb_top3 = []
    seen_pbs = set()

    # We need the full pb_ranked list; pass as parameter or reuse arg
    # top_pb is just the #1; for variety use top 3 distinct PBs from caller
    # We'll use top_pb for combo 1, and vary +1/-1 range for 2 & 3
    pbs = [top_pb, (top_pb % 26) + 1, max(1, top_pb - 1)]

    combos = []
    used_sets = []

    weights = list(range(len(pool), 0, -1))  # highest rank = highest weight

    max_attempts = 200
    for i in range(count):
        pb = pbs[i % len(pbs)]
        for _ in range(max_attempts):
            try:
                sample = random.choices(pool, weights=weights[:len(pool)], k=15)
                # Deduplicate while preserving order (weighted preference)
                seen = set()
                unique = []
                for n in sample:
                    if n not in seen:
                        seen.add(n)
                        unique.append(n)
                    if len(unique) == 5:
                        break

                if len(unique) < 5:
                    continue

                wb = sorted(unique)

                # Decade spread check: need numbers from at least 3 different decades
                decades = set(n // 10 for n in wb)
                if len(decades) < 3:
                    continue

                # No duplicate combo
                if frozenset(wb) in used_sets:
                    continue

                used_sets.append(frozenset(wb))
                combos.append((wb, pb))
                break

            except Exception:
                continue

    # Fallback: if we couldn't build enough, add simple top-5 slice
    while len(combos) < count:
        fallback_wb = sorted([d['Number'] for d in wb_ranked[:5]])
        fallback_pb = pbs[len(combos) % len(pbs)]
        if frozenset(fallback_wb) not in used_sets:
            combos.append((fallback_wb, fallback_pb))
            used_sets.append(frozenset(fallback_wb))
        else:
            break  # avoid infinite loop

    return combos[:count]


def log(msg, error=False):
    tag = "ERROR" if error else "INFO"
    print(f"[SCHEDULER {tag}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — {msg}", flush=True)


def init_scheduler(app):
    """
    Call this once from index.py after app is created.
    Schedules the simulation job for Mon/Wed/Sat at 08:00 Eastern.
    """
    scheduler = BackgroundScheduler(timezone=DRAW_TZ)

    scheduler.add_job(
        func=run_draw_day_simulation,
        trigger=CronTrigger(
            day_of_week='mon,wed,sat',
            hour=8,
            minute=0,
            timezone=DRAW_TZ
        ),
        id='draw_day_simulation',
        name='Draw-Day Auto Simulation',
        replace_existing=True,
        misfire_grace_time=3600  # If server was down, allow up to 1hr late
    )

    scheduler.start()
    log(f"⏰ Scheduler started — jobs: {[j.id for j in scheduler.get_jobs()]}")
    return scheduler
