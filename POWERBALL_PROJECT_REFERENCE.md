# POWERBALL ANALYTICS HUB — PROJECT REFERENCE
# ═══════════════════════════════════════════════════════════════
# Upload this file at the start of any new Claude chat to restore
# full project context instantly.
# Last updated: March 2026 (Sessions 1–8 + PWA)
# ═══════════════════════════════════════════════════════════════


## 1. PROJECT OVERVIEW
─────────────────────
Name:        Powerball Analytics Hub
Live URL:    https://powerball-play.onrender.com
GitHub:      https://github.com/ukdeepak-33/Powerball_Play
Stack:       Flask + Supabase + Jinja2 + Tailwind + Groq AI
Hosting:     Render.com (auto-deploys from GitHub main branch)
PWA:         Installed — works as mobile app on iPhone/Android


## 2. FOLDER STRUCTURE
──────────────────────
Powerball_Play/
├── api/
│   ├── index.py              ← Main Flask app (~7000 lines)
│   └── scheduler.py          ← APScheduler draw-day automation
├── static/
│   ├── manifest.json         ← PWA manifest
│   ├── service-worker.js     ← PWA service worker
│   ├── favicon.ico
│   └── icons/
│       ├── icon-192.png      ← PWA home screen icon
│       └── icon-512.png      ← PWA splash icon
├── templates/
│   ├── base.html             ← Master layout (dark terminal design)
│   ├── index.html            ← Home page (tabbed generators)
│   └── [30+ page templates]
├── requirements.txt
└── render.yaml (or Procfile)


## 3. KEY CONFIGURATION (index.py top)
───────────────────────────────────────
SUPABASE_PROJECT_URL      = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY         = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY      = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_TABLE_NAME       = 'powerball_draws'
GENERATED_NUMBERS_TABLE_NAME = 'generated_powerball_numbers'
GROQ_API_KEY              = os.environ.get("GROQ_API_KEY")
Groq model used:          llama-3.1-8b-instant

Flask app init:
  TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
  STATIC_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
  app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


## 4. SUPABASE DATABASE TABLES
───────────────────────────────
Table 1: powerball_draws
  Columns: Draw Date, Number 1, Number 2, Number 3, Number 4, Number 5, Powerball
  draw_date field used in some helpers (lowercase, from get_all_official_draws())
  ~1239 records (2016 to present), 3 draws/week (Mon/Wed/Sat)

Table 2: generated_powerball_numbers
  Columns: id, generated_date, number_1, number_2, number_3, number_4,
           number_5, powerball, source
  source values: 'SCHEDULER', 'manual', 'smart_pick', 'custom'

IMPORTANT — Supabase pagination:
  Default cap = 1000 rows silently. Always paginate large fetches:
  while True:
      rows = fetch(limit=1000, offset=offset)
      if not rows or len(rows) < 1000: break
      offset += 1000


## 5. DESIGN SYSTEM
────────────────────
Theme: Dark Terminal / Cyber Analytics

CSS Variables (in base.html <style>):
  --bg:     #040608      (page background)
  --s1:     #080c10      (card background)
  --s2:     #0c1014      (input background)
  --s3:     #10141a
  --border: rgba(255,255,255,0.065)
  --text:   rgba(255,255,255,0.87)
  --muted:  rgba(255,255,255,0.33)
  --red:    #ef4444      (powerball / accent)
  --red2:   rgba(239,68,68,0.12)
  --red3:   rgba(239,68,68,0.28)
  --violet: #a78bfa
  --teal:   #14b8a6
  --amber:  #f59e0b      (match highlights / gold balls)
  --green:  #34d399
  --blue:   #60a5fa
  --cyan:   #22d3ee
  --sidebar-w: 260px

Fonts:
  Headings:  Syne (700, 800) — via Google Fonts
  Body/Code: JetBrains Mono (400, 500, 600, 700)
  (loaded in base.html <head>)

Ball rendering:
  White balls: radial-gradient(circle at 32% 30%, #f0f4ff 0%, #c8d8ed 45%, #8aaac8 100%)
               color: #0d1829
  Powerballs:  radial-gradient(circle at 32% 30%, #ff8a80 0%, #ef5350 50%, #b71c1c 100%)
               color: #fff
  Matched/gold: radial-gradient(circle at 32% 30%, #fde68a 0%, #f59e0b 50%, #b45309 100%)
               color: #1c1000 + box-shadow: 0 0 8px rgba(245,158,11,0.55)
  Dimmed (no match): opacity: 0.35

CSS classes:
  .ball        40×40px white ball
  .ball-sm     32×32px small white ball
  .wb          36×36px (used in history/compare views)
  .pb          36×36px red powerball
  .generated-ball  50×50px (large, for generation display)
  .card        background s1, border, border-radius 14px
  .btn-primary blue gradient button
  .btn-secondary transparent border button
  .sb-link     sidebar nav link
  .sb-link.active  red highlight


## 6. BASE.HTML STRUCTURE
──────────────────────────
- Sidebar: fixed 260px left, collapsible on mobile
- Sidebar sections: Core / Patterns / Analysis / Search & History / Pick Generators / AI Tools
- Top header: sticky 56px, blur backdrop, hamburger on mobile
- PWA tags added in <head> (manifest, theme-color, apple meta tags)
- Service worker registration script before </body>
- Active link highlight via JS (matches window.location.pathname)


## 7. ALL TEMPLATES (33 total)
────────────────────────────────
Core:
  index.html                    → Home (tabbed: Quick Pick / Smart Pick / Custom)
  frequency_analysis.html       → Number frequency charts
  hot_cold_numbers.html         → Hot/cold with AI analysis (Groq)
  sum_of_main_balls.html        → Sum range analysis
  monthly_white_ball_analysis.html → Monthly trends
  odd_even_trends.html          → Odd/even split analysis

Patterns:
  grouped_patterns_analysis.html
  grouped_patterns_yearly_comparison.html
  co_occurrence_analysis.html
  special_patterns_analysis.html
  consecutive_trends.html
  boundary_crossing_pairs_trends.html
  weekday_trends.html
  triplets_analysis.html
  pairs_analysis.html (route: /pairs-analysis)

Analysis:
  number_age_distribution.html
  yearly_white_ball_trends.html
  powerball_frequency_by_year.html
  positional_analysis.html
  powerball_position_frequency.html  ← DB-connected (was hardcoded, fixed session 7)
  sum_trends_and_gaps.html
  white_ball_gap_analysis.html

Search & History:
  strict_positional_search.html
  find_results_by_sum.html
  find_results_by_first_white_ball.html
  historical_data.html
  generated_numbers_history.html     ← Scheduler picks + Analyse feature
  check_my_numbers.html

Pick Generators:
  smart_pick_generator.html
  custom_combinations.html
  my_jackpot_pick.html

AI Tools:
  ai_assistant.html
  simulate_multiple_draws.html


## 8. KEY FUNCTIONS IN index.py
─────────────────────────────────
get_all_official_draws()
  → Fetches ALL draws from Supabase with pagination
  → Returns list of dicts with draw_date, Number 1..5, Powerball
  → IMPORTANT: uses 'draw_date' key (lowercase)

check_pick_against_draws_cmn(white_balls, powerball, official_draws, min_matches=2)
  → Compares a pick against draws list
  → Returns matches where white_matches >= min_matches
  → Each result: draw_date, official_numbers, official_powerball,
                 white_matches, powerball_match, total_matches

check_generated_against_history(white_balls, powerball, df_historical)
  → Compares pick against last 2 years of df (pandas DataFrame)
  → Returns summary dict with tier counts and draw lists
  → Each draw includes: date, white_balls, powerball,
                        matched_whites, powerball_match  ← ADDED SESSION 8
  → Tiers: Match 5WB+PB, Match 5WB, Match 4WB+PB, Match 4WB,
           Match 3WB+PB, Match 3WB, Match 2WB+PB,
           Match 1WB+PB, Match PB Only, No Match

get_cached_analysis(key, func, *args)
  → Simple in-memory cache for expensive analysis functions

_get_supabase_headers(is_service_key=False)
  → Returns headers dict for Supabase REST API calls

AI routes pattern (all use requests.post, NOT call_groq helper):
  response = requests.post(
      "https://api.groq.com/openai/v1/chat/completions",
      headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
      json={"model": "llama-3.1-8b-instant", "messages": [...], "max_tokens": 1000},
      timeout=30
  )


## 9. SCHEDULER (scheduler.py)
───────────────────────────────
- Uses APScheduler BackgroundScheduler + CronTrigger
- Runs on draw days: Monday, Wednesday, Saturday at 08:00 ET
- Generates 3 picks using dedicated scheduler logic (NOT smart_pick_generator)
- Saves picks to generated_powerball_numbers with source='SCHEDULER'
- Init called in index.py at bottom: init_scheduler(app)
- Route: /api/scheduler-status → returns next run times


## 10. PWA SETUP
─────────────────
Files added:
  static/manifest.json          → App name, icons, theme, display:standalone
  static/service-worker.js      → Caching strategy (network-first HTML, cache-first static)
  static/icons/icon-192.png     → Home screen icon
  static/icons/icon-512.png     → Splash screen icon

Routes added to index.py:
  @app.route('/manifest.json')
  def manifest():
      static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
      return send_from_directory(static_dir, 'manifest.json')

  @app.route('/service-worker.js')
  def service_worker():
      static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
      response = make_response(send_from_directory(static_dir, 'service-worker.js'))
      response.headers['Cache-Control'] = 'no-cache'
      response.headers['Content-Type']  = 'application/javascript'
      return response

base.html additions (in <head>):
  <link rel="manifest" href="/manifest.json">
  <meta name="theme-color" content="#040608">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="apple-mobile-web-app-title" content="PB Analytics">
  <link rel="apple-touch-icon" href="/static/icons/icon-192.png">

base.html service worker (before </body>):
  <script>
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('/service-worker.js')
      });
    }
  </script>

Install: iPhone via Safari → Share → Add to Home Screen ✅


## 11. BUGS FIXED (all sessions)
──────────────────────────────────
Session 1-2:
  - Hot/cold route 500 error: missing function, wrong column names, function name collision
  - Groq dependency missing from requirements.txt
  - Decommissioned Groq model replaced with llama-3.1-8b-instant

Session 3-5:
  - call_groq helper undefined → replaced with requests.post pattern in all AI routes
  - consecutive_trends missing all_draws_json variable
  - Difference pairs returning only 2026 data (date filter bug)
  - Date formatting showing "00:00:00 GMT" → fixed to clean YYYY-MM-DD

Session 6:
  - source parameter NameError in save routes
  - Scheduler double-save bug (was saving twice per run)
  - check_my_numbers: picks with no matches were silently dropped
  - check_my_numbers: Supabase silent 1000-row cap (no pagination)

Session 7:
  - last_digit_analysis / powerball_position_frequency: hardcoded draw data
    replaced with live Supabase fetch + draws_json passed to template
  - Supabase silent 1000-row pagination cap → paginated fetch with while loop

Session 8:
  - generated_numbers_history Analyse button: showed tier+count+dates only
    → Now shows each matching draw with gold highlighted matched balls
  - check_generated_against_history: added matched_whites + powerball_match
    to each draw object in results
  - api_check_saved_picks: indentation error (for loop outside try block)

PWA Session:
  - manifest.json / service-worker.js returning 404
    → Fixed: index.py is in api/ subfolder, static/ is at repo root
    → Used send_from_directory with os.path.dirname(__file__) navigation


## 12. IMPORTANT PATTERNS & GOTCHAS
──────────────────────────────────────
1. Static files: index.py is in api/, static/ is at repo root.
   Always use send_from_directory with:
   static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')

2. Supabase draw data has TWO formats:
   - df (pandas): columns 'Draw Date', 'Number 1'...'Number 5', 'Powerball', 'Draw Date_dt'
   - get_all_official_draws() dicts: keys 'draw_date', 'Number 1'...'Number 5', 'Powerball'

3. Always paginate Supabase fetches (1000 row silent cap):
   offset = 0
   while True:
       rows = fetch(limit=1000, offset=offset)
       if not rows or len(rows) < 1000: break
       offset += 1000

4. Groq AI: use requests.post directly, never a helper function.
   Model: llama-3.1-8b-instant
   Always set timeout=30

5. Scheduler source tag: saved picks have source='SCHEDULER'
   Manual picks: source='manual'
   Smart picks: source='smart_pick'

6. generated_numbers_history.html:
   - Analyse button calls /analyze_generated_historical_matches (POST)
   - Response includes match_summary with draws containing matched_whites[]
   - Gold balls = matched, dimmed (opacity 0.35) = not matched

7. PWA install: Firefox does NOT support PWA on desktop.
   Use Chrome (Android), Safari (iPhone), or Edge (desktop).


## 13. PENDING / FUTURE TASKS
───────────────────────────────
- [ ] check_my_numbers saved picks: apply pagination fix (1000 row cap)
      params needs 'limit': '1000', 'offset': '0'
- [ ] Consider adding push notifications to PWA for draw day alerts
- [ ] Scheduler page: add "next draw countdown" timer
- [ ] Consider adding a "favourites" feature to save best picks


## 14. HOW TO USE THIS FILE IN A NEW CHAT
──────────────────────────────────────────
1. Start a new Claude chat
2. Upload this file
3. Say: "This is my project reference. I need help with [your task]."
4. Claude will have full context of the entire project.

For code changes, also upload the relevant file:
  - For backend changes: upload index.py
  - For frontend changes: upload the specific template .html file
  - For scheduler changes: upload scheduler.py
