#!/usr/bin/env python3
"""NRL Tips — Kivy Android app"""

import importlib.util
import json
import marshal
import os
import queue
import runpy
import shutil
import sys
import threading
import warnings

# Suppress noisy requests warning about missing chardet/charset_normalizer on Android
warnings.filterwarnings("ignore", message=".*character detection.*")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.image import Image as KivyImage
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.utils import platform

# ── Themes ────────────────────────────────────────────────────────────────────
THEMES = {
    'dark': {
        'BG':     "1a1a2e",
        'PANEL':  "16213e",
        'ACCENT': "0f3460",
        'OUTPUT': "0d0d22",
        'GREEN':  "00b894",
        'YELLOW': "fdcb6e",
        'RED':    "d63031",
        'WHITE':  "ffffff",
        'GREY':   "ffffff",
    },
    'light': {
        'BG':     "dde9f5",
        'PANEL':  "e8edf3",
        'ACCENT': "b3cce6",
        'OUTPUT': "dce4ec",
        'GREEN':  "27ae60",
        'YELLOW': "f39c12",
        'RED':    "e74c3c",
        'WHITE':  "1a1a2e",
        'GREY':   "444444",
    },
}


_CURRENT_THEME = 'dark'


def _rgba(h):
    return int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, 1


# These module-level vars are updated by _set_theme() at startup and on toggle
BG_HEX = PANEL_HEX = ACCENT_HEX = OUTPUT_HEX = "000000"
GREEN_HEX = YELLOW_HEX = RED_HEX = WHITE_HEX = GREY_HEX = "ffffff"
BG_C = PANEL_C = ACCENT_C = GREEN_C = WHITE_C = GREY_C = (0, 0, 0, 1)


def _set_theme(name: str):
    global _CURRENT_THEME
    global BG_HEX, PANEL_HEX, ACCENT_HEX, OUTPUT_HEX
    global GREEN_HEX, YELLOW_HEX, RED_HEX, WHITE_HEX, GREY_HEX
    global BG_C, PANEL_C, ACCENT_C, GREEN_C, WHITE_C, GREY_C
    _CURRENT_THEME = name
    t = THEMES[name]
    BG_HEX     = t['BG'];     PANEL_HEX  = t['PANEL']
    ACCENT_HEX = t['ACCENT']; OUTPUT_HEX = t['OUTPUT']
    GREEN_HEX  = t['GREEN'];  YELLOW_HEX = t['YELLOW']
    RED_HEX    = t['RED'];    WHITE_HEX  = t['WHITE']
    GREY_HEX   = t['GREY']
    BG_C     = _rgba(BG_HEX)
    PANEL_C  = _rgba(PANEL_HEX)
    ACCENT_C = _rgba(ACCENT_HEX)
    GREEN_C  = _rgba(GREEN_HEX)
    WHITE_C  = _rgba(WHITE_HEX)
    GREY_C   = _rgba(GREY_HEX)


_set_theme('dark')  # apply defaults before any widget is created

RADIUS = dp(14)   # button corner radius


class RoundedButton(Button):
    """Button with rounded corners and press-darken feedback."""

    def __init__(self, btn_color, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ''
        self.background_down   = ''
        self.background_color  = (0, 0, 0, 0)
        self._base = btn_color
        with self.canvas.before:
            self._ci   = Color(*btn_color)
            self._rr   = RoundedRectangle(pos=self.pos, size=self.size,
                                          radius=[RADIUS])
        self.bind(pos=self._upd, size=self._upd)

    def _upd(self, *_):
        self._rr.pos  = self.pos
        self._rr.size = self.size

    def on_state(self, _w, state):
        c = self._base
        self._ci.rgba = (
            (max(0, c[0]-.12), max(0, c[1]-.12), max(0, c[2]-.12), c[3])
            if state == 'down' else c
        )

# ── Paths ─────────────────────────────────────────────────────────────────────
BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = BUNDLE_DIR   # overwritten in NRLTipsApp.build() on Android

_DATA_FILES = [
    'nrl_source_data.csv',
    'nrl_model.npz',
    'nrl_model_no_odds.npz',
    'nrl_model_info.json',
    'nrl_model_no_odds_info.json',
    'round_template.csv',
    'squad_player_cache.json',
    'nrl_odds_raw.xlsx',
]
_SCRIPT_FILES = [
    's1_history.py', 's2_stats.py', 's3_weather.py',
    's4_squads.py',  's5_odds.py',  's6_tips.py',
    's9_performance.py', 'm5_nrl.py', 'nrl_predict.py',
    'u1_travel.py',  'u2_weather.py', 'u3_squad.py',
]


def _refresh_script(fname):
    """Ensure the latest version of a Python module is importable from DATA_DIR.

    p4a compiles .py → .pyc and strips the source, so BUNDLE_DIR contains
    only .pyc files.  If we only copy .py we silently leave a stale copy in
    DATA_DIR that shadows the bundle.

    Strategy:
      1. If bundle has .py  → copy it to DATA_DIR (overwrites old copy).
      2. If bundle has .pyc only → copy it into DATA_DIR/__pycache__/ with the
         correct CPython version tag so Python's import system finds it, then
         delete the stale .py from DATA_DIR so the cache wins.
      3. Clear the module from sys.modules so the next import re-reads disk.
    """
    mod_name = fname[:-3]  # strip .py
    src_py   = os.path.join(BUNDLE_DIR, fname)
    src_pyc  = os.path.join(BUNDLE_DIR, mod_name + '.pyc')
    dst_py   = os.path.join(DATA_DIR,   fname)

    if os.path.exists(src_py):
        shutil.copy(src_py, dst_py)
    elif os.path.exists(src_pyc):
        # Place in __pycache__ with the version-tagged filename Python expects
        cache_path = importlib.util.cache_from_source(dst_py)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        shutil.copy(src_pyc, cache_path)
        # Remove stale .py so the __pycache__ version is used
        if os.path.exists(dst_py):
            os.remove(dst_py)

    # Force re-import on next use
    sys.modules.pop(mod_name, None)


def _init_data():
    """Copy bundled files to writable DATA_DIR.
    Scripts are always refreshed so updated APKs take effect immediately.
    Data files are only copied if not already present (preserve user data).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname in _DATA_FILES:
        src = os.path.join(BUNDLE_DIR, fname)
        dst = os.path.join(DATA_DIR, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

    # Also copy any bundled historical tips (CSV and TXT caches)
    import glob
    for pattern in ["tips_*.csv", "tips_cache_*.txt"]:
        for src in glob.glob(os.path.join(BUNDLE_DIR, pattern)):
            fname = os.path.basename(src)
            dst   = os.path.join(DATA_DIR, fname)
            if not os.path.exists(dst):
                shutil.copy(src, dst)

    for fname in _SCRIPT_FILES:
        _refresh_script(fname)
    if DATA_DIR not in sys.path:
        sys.path.insert(0, DATA_DIR)


# ── Settings ──────────────────────────────────────────────────────────────────

def _settings_path():
    return os.path.join(DATA_DIR, 'android_settings.json')


def _load_settings():
    p = _settings_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def _save_settings(d):
    with open(_settings_path(), 'w') as f:
        json.dump(d, f)


# ── Output pipe ───────────────────────────────────────────────────────────────

class _Pipe:
    def __init__(self, q, kind):
        self._q, self._kind = q, kind

    def write(self, t):
        if t:
            self._q.put((self._kind, t))

    def flush(self):
        pass

    def isatty(self):
        return False


# ── Background helper ─────────────────────────────────────────────────────────

def _bg(widget, color, radius=0):
    with widget.canvas.before:
        Color(*color)
        r = (RoundedRectangle(pos=widget.pos, size=widget.size, radius=[dp(radius)])
             if radius else Rectangle(pos=widget.pos, size=widget.size))
    widget.bind(
        pos =lambda w, v: setattr(r, 'pos',  v),
        size=lambda w, v: setattr(r, 'size', v),
    )


# ── System theme detection ────────────────────────────────────────────────────

def _detect_system_theme() -> str:
    """Return 'dark' or 'light' based on the Android system UI mode.
    Falls back to 'dark' on desktop / if jnius is unavailable.
    """
    if platform != 'android':
        return 'dark'
    try:
        from jnius import autoclass  # type: ignore
        Configuration  = autoclass('android.content.res.Configuration')
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        cfg   = PythonActivity.mActivity.getResources().getConfiguration()
        night = cfg.uiMode & Configuration.UI_MODE_NIGHT_MASK
        return 'dark' if night == Configuration.UI_MODE_NIGHT_YES else 'light'
    except Exception:
        return 'dark'


# ── Tips caching helpers ──────────────────────────────────────────────────────

def _tips_cache_path(season: int, rnd: int, no_odds: bool) -> str:
    model = 'no_odds' if no_odds else 'odds'
    return os.path.join(DATA_DIR, f'tips_cache_{season}_R{rnd:02d}_{model}.txt')


def _round_has_started(season: int, rnd: int) -> bool:
    """Return True if the earliest game in this round has kicked off.
    Checks NRL API for precise kickoff times; falls back to date-based check on failure."""
    import datetime as _dt
    import requests as _req
    now = _dt.datetime.now().astimezone()

    try:
        r = _req.get(
            "https://www.nrl.com/draw/data",
            params={"competition": 111, "season": season, "round": rnd},
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json",
                     "Referer": "https://www.nrl.com/"},
            timeout=6,
        )
        if r.status_code == 200:
            fixtures = r.json().get("fixtures", [])
            if fixtures:
                # Find the earliest kickoff
                first_ko = None
                for f in fixtures:
                    ko_str = f.get("clock", {}).get("kickOffTimeLong") or f.get("matchDate", "")
                    if not ko_str: continue
                    try:
                        ko = _dt.datetime.fromisoformat(ko_str)
                        if ko.tzinfo is None:
                            ko = ko.replace(tzinfo=_dt.timezone.utc)
                        if first_ko is None or ko < first_ko:
                            first_ko = ko
                    except ValueError:
                        pass
                
                if first_ko:
                    return now >= first_ko
    except Exception:
        pass

    # Fallback to date-only check using CSV
    today = _dt.date.today()
    data_path = os.path.join(DATA_DIR, 'nrl_source_data.csv')
    if os.path.exists(data_path):
        try:
            import pandas as pd
            df = pd.read_csv(data_path, usecols=['season', 'round', 'date'])
            mask = (df['season'].astype(int) == season) & (df['round'].astype(int) == rnd)
            dates = pd.to_datetime(df.loc[mask, 'date'], errors='coerce').dropna()
            if not dates.empty:
                return dates.min().date() < today  # Yesterday or earlier
        except Exception:
            pass

    return False


# ── Main layout ───────────────────────────────────────────────────────────────

class MainLayout(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self._q       = queue.Queue()
        self._running = False
        self._out_buf = ""
        Window.clearcolor = BG_C
        self._build_ui()
        Clock.schedule_interval(self._pump, 0.1)
        # Poll system theme every 5 s; only applies when no manual preference saved
        Clock.schedule_interval(self._sync_system_theme, 5)

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header is always visible, outside the ScreenManager
        self.add_widget(self._header())

        sm = ScreenManager()

        # Main screen — round selector + button grid
        main_screen = Screen(name='main')
        main_box = BoxLayout(orientation='vertical')
        main_box.add_widget(self._round_row())
        main_box.add_widget(self._button_grid())
        main_screen.add_widget(main_box)

        # Results screen — nav bar + output area
        results_screen = Screen(name='results')
        results_box = BoxLayout(orientation='vertical')
        results_box.add_widget(self._results_nav())
        results_box.add_widget(self._output_area())
        results_screen.add_widget(results_box)

        sm.add_widget(main_screen)
        sm.add_widget(results_screen)
        self._sm = sm
        self.add_widget(sm)

        # Status bar always visible at the bottom — tap to jump to results
        self.add_widget(self._status_bar())

    def _header(self):
        h = BoxLayout(size_hint_y=None, height=dp(64),
                      padding=[dp(12), dp(8), dp(8), dp(8)], spacing=dp(8))
        _bg(h, ACCENT_C)
        icon_path = os.path.join(BUNDLE_DIR, 'icon.png')
        if os.path.exists(icon_path):
            h.add_widget(KivyImage(
                source=icon_path,
                size_hint=(None, None),
                size=(dp(42), dp(42)),
            ))
        lbl = Label(
            text="[b]NRL Tips[/b]", markup=True,
            font_size=dp(24), color=WHITE_C,
            halign='left', valign='middle',
        )
        lbl.bind(size=lbl.setter('text_size'))
        h.add_widget(lbl)
        theme_icon = "☀" if _CURRENT_THEME == 'dark' else "☾"
        theme_btn = Button(
            text=theme_icon, font_name='DejaVuSans',
            font_size=dp(20), size_hint=(None, 1), width=dp(44),
            background_color=(0, 0, 0, 0), color=WHITE_C,
        )
        theme_btn.bind(on_press=self._toggle_theme)
        h.add_widget(theme_btn)
        cog = Button(
            text="[b]...[/b]", markup=True, font_size=dp(20),
            size_hint=(None, 1), width=dp(50),
            background_color=(0, 0, 0, 0), color=WHITE_C,
        )
        cog.bind(on_press=self._settings_popup)
        h.add_widget(cog)
        return h

    def _round_row(self):
        row = BoxLayout(
            size_hint_y=None, height=dp(58),
            padding=[dp(16), dp(8), dp(16), dp(8)], spacing=dp(12),
        )
        _bg(row, PANEL_C)
        row.add_widget(Label(
            text="[b]Round:[/b]", markup=True, color=WHITE_C,
            size_hint=(None, 1), width=dp(80), font_size=dp(17),
        ))
        # Rounded wrapper for the TextInput
        inp_wrap = BoxLayout(size_hint=(None, None), width=dp(90), height=dp(40))
        _bg(inp_wrap, _rgba(ACCENT_HEX), radius=12)
        self.round_input = TextInput(
            text="auto", multiline=False,
            background_normal='', background_active='',
            background_color=(0, 0, 0, 0),
            foreground_color=WHITE_C, cursor_color=WHITE_C,
            font_size=dp(17), padding=[dp(10), dp(8)],
        )
        inp_wrap.add_widget(self.round_input)
        row.add_widget(inp_wrap)
        row.add_widget(Widget())
        return row

    def _button_grid(self):
        g = GridLayout(
            cols=2,
            padding=[dp(12), dp(10), dp(12), dp(4)], spacing=dp(8),
        )
        _bg(g, PANEL_C)

        defs = [
            ("Tips (with Odds)",  "1565c0", WHITE_HEX, self._get_tips),
            ("Tips (No Odds)",    "1e88e5", WHITE_HEX, self._get_tips_no_odds),
            ("Compare Models",    "1976d2", WHITE_HEX, self._compare),
            ("Show Models",       "00695c", WHITE_HEX, self._show_model),
            ("Collect New Data",  "00796b", WHITE_HEX, self._collect_new),
            ("Collect All Data",  "00897b", WHITE_HEX, self._collect_all),
            ("Cancel",            RED_HEX,  WHITE_HEX, self._cancel),
            ("Clear",             "2a2a4a", GREY_HEX,  self._clear),
        ]
        self._action_btns = []
        for label, bg, fg, fn in defs:
            b = RoundedButton(
                btn_color=_rgba(bg),
                text=label, bold=True, font_size=dp(18),
                color=_rgba(fg),
            )
            b.bind(on_press=fn)
            g.add_widget(b)
            if fn not in (self._cancel, self._clear):
                self._action_btns.append(b)
        return g

    def _output_area(self):
        sv = ScrollView(do_scroll_x=False)
        _bg(sv, _rgba(OUTPUT_HEX))
        self.out_lbl = Label(
            text="", markup=True, valign='top', halign='left',
            size_hint_y=None, font_size=dp(16), color=WHITE_C,
            padding=[dp(16), dp(14)],
            font_name='DejaVuSans',   # full Unicode coverage for ✔ ✘ █ ⚡ ★ etc.
        )
        self.out_lbl.bind(
            width       =lambda w, v: setattr(w, 'text_size', (v, None)),
            texture_size=lambda w, v: setattr(w, 'height',    v[1]),
        )
        sv.add_widget(self.out_lbl)
        return sv

    def _status_bar(self):
        sb = BoxLayout(size_hint_y=None, height=dp(48),
                       padding=[dp(14), dp(6), dp(10), dp(6)], spacing=dp(8))
        _bg(sb, ACCENT_C)
        self.status_lbl = Label(
            text="Ready", color=WHITE_C, font_size=dp(14), bold=True,
            halign='left', valign='middle',
        )
        self.status_lbl.bind(size=self.status_lbl.setter('text_size'))
        sb.add_widget(self.status_lbl)
        view_btn = RoundedButton(
            btn_color=_rgba(GREEN_HEX),
            text="[b]View Output[/b]", markup=True,
            font_size=dp(13), color=_rgba("1a1a2e"),
            size_hint=(None, 1), width=dp(110),
        )
        view_btn.bind(on_press=lambda *_: self._go_to_results(self.results_title_lbl.text))
        sb.add_widget(view_btn)
        return sb

    def _results_nav(self):
        nav = BoxLayout(size_hint_y=None, height=dp(58),
                        padding=[dp(8), dp(6), dp(8), dp(6)], spacing=dp(8))
        _bg(nav, ACCENT_C)
        back_btn = RoundedButton(
            btn_color=_rgba("0f3460"),
            text="[b]< Back[/b]", markup=True,
            font_size=dp(14), color=WHITE_C,
            size_hint=(None, 1), width=dp(85),
        )
        back_btn.bind(on_press=self._go_back)
        nav.add_widget(back_btn)
        self.results_title_lbl = Label(
            text="Results", color=WHITE_C,
            font_size=dp(17), bold=True,
            halign='left', valign='middle',
        )
        self.results_title_lbl.bind(size=self.results_title_lbl.setter('text_size'))
        nav.add_widget(self.results_title_lbl)
        cancel_btn = RoundedButton(
            btn_color=_rgba(RED_HEX),
            text="[b]Cancel[/b]", markup=True,
            font_size=dp(14), color=WHITE_C,
            size_hint=(None, 1), width=dp(85),
        )
        cancel_btn.bind(on_press=self._cancel)
        nav.add_widget(cancel_btn)
        return nav

    def _go_to_results(self, title):
        self.results_title_lbl.text = title
        self._sm.transition = SlideTransition(direction='left', duration=0.25)
        self._sm.current = 'results'

    def _go_back(self, *_):
        self._sm.transition = SlideTransition(direction='right', duration=0.25)
        self._sm.current = 'main'

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _rnd(self):
        v = self.round_input.text.strip()
        return ["--round", v] if v and v.lower() != "auto" and v.isdigit() else []

    def _get_tips(self, *_):
        self._go_to_results("Tips — With Odds")
        args = self._rnd()
        self._start_worker("Fetching tips (with odds)...",
                           lambda: self._run_tips_cached(False, args))

    def _get_tips_no_odds(self, *_):
        self._go_to_results("Tips — No Odds")
        args = self._rnd()
        self._start_worker("Fetching tips (no odds)...",
                           lambda: self._run_tips_cached(True, args))

    def _resolve_tips_round(self, args: list) -> tuple:
        """Return (season, round) from args, or detect via NRL API with CSV fallback."""
        import datetime as _dt
        season = _dt.datetime.now().year
        rnd    = None
        for i, a in enumerate(args):
            if a == '--season' and i + 1 < len(args):
                try: season = int(args[i + 1])
                except ValueError: pass
            elif a == '--round' and i + 1 < len(args):
                try: rnd = int(args[i + 1])
                except ValueError: pass
        if rnd is None:
            rnd = self._detect_current_round(season)
        return season, rnd

    def _detect_current_round(self, season: int) -> int:
        """Find current NRL round via API (first round with a future game).
        Falls back to max-completed+1 from CSV if the API is unreachable."""
        import datetime as _dt, requests as _req
        today = str(_dt.date.today())
        # Date-based estimate for search starting point
        days     = (_dt.datetime.now() - _dt.datetime(season, 3, 6)).days
        est      = max(1, min(27, days // 7 + 1))
        headers  = {"User-Agent": "Mozilla/5.0", "Accept": "application/json",
                    "Referer": "https://www.nrl.com/"}
        for rnd in range(max(1, est - 3), 28):
            try:
                r = _req.get(
                    "https://www.nrl.com/draw/data",
                    params={"competition": 111, "season": season, "round": rnd},
                    headers=headers, timeout=6,
                )
                if r.status_code != 200:
                    continue
                fixtures = r.json().get("fixtures", [])
                if not fixtures:
                    continue
                dates = [
                    (f.get("clock", {}).get("kickOffTimeLong") or f.get("matchDate", ""))[:10]
                    for f in fixtures
                    if (f.get("clock", {}).get("kickOffTimeLong") or f.get("matchDate", ""))
                ]
                if any(d >= today for d in dates):
                    return rnd
            except Exception:
                break
        # CSV fallback
        data_path = os.path.join(DATA_DIR, 'nrl_source_data.csv')
        try:
            import pandas as pd
            df        = pd.read_csv(data_path, usecols=['season', 'round', 'winner'])
            completed = df[(df['season'].astype(int) == season) & df['winner'].notna()]
            return int(completed['round'].max()) + 1 if not completed.empty else 1
        except Exception:
            return est

    def _run_tips_cached(self, no_odds: bool, args: list):
        """Serve cached tips once the round has kicked off; regenerate before that."""
        season, rnd = self._resolve_tips_round(args)
        cache_path  = _tips_cache_path(season, rnd, no_odds)

        if os.path.exists(cache_path):
            if _round_has_started(season, rnd):
                # Round is live — serve cache, do not run the model
                with open(cache_path) as fh:
                    for line in fh:
                        print(line.rstrip(), flush=True)
                return
            else:
                # Round hasn't started — delete stale cache so fresh tips are saved
                try:
                    os.remove(cache_path)
                except Exception:
                    pass

        # Run live and capture output for next time.
        # Always pass --round explicitly so s6_tips.py uses the same round we resolved.
        lines: list = []
        original_write = sys.stdout.write

        def _capturing_write(text):
            if text:
                lines.append(text)
            original_write(text)

        sys.stdout.write = _capturing_write
        try:
            full_args = ['--season', str(season), '--round', str(rnd)]
            if no_odds:
                full_args.append('--no-odds')
            self._exec('s6_tips.py', full_args)
        finally:
            sys.stdout.write = original_write

        if lines:
            try:
                with open(cache_path, 'w') as fh:
                    fh.write(''.join(lines))
            except Exception:
                pass

    def _compare(self, *_):
        self._go_to_results("Compare Models")
        self._start_worker("Comparing models...",
                           lambda: self._exec('s9_performance.py', ["--compare"] + self._rnd()))

    def _collect_new(self, *_):
        self._go_to_results("Collect New Data")
        self._start_worker("Collecting new data...", self._do_collect_new)

    def _collect_all(self, *_):
        self._go_to_results("Collect All Data")
        self._start_worker("Collecting all data...", self._do_collect_all)

    def _show_model(self, *_):
        self._go_to_results("Model Info")
        self._clear()
        for path, label in [
            (os.path.join(DATA_DIR, 'nrl_model_info.json'),        "With Odds"),
            (os.path.join(DATA_DIR, 'nrl_model_no_odds_info.json'), "No Odds"),
        ]:
            if not os.path.exists(path):
                self._append_line(f"[{label}] No model info — run Retrain first.")
                continue
            with open(path) as f:
                d = json.load(f)
            self._append_line(f"=== {label} ===")
            self._append_line(f"  CV acc:    {d['cv_accuracy']:.3f} ± {d['cv_std']:.3f}")
            self._append_line(f"  Train acc: {d['train_accuracy']:.3f}")
            self._append_line(f"  Brier:     {d['brier_score']:.4f}")
            self._append_line("  Top features:")
            for feat in d.get("features", [])[:8]:
                filled = int(feat["importance"] * 20)
                bar = "█" * filled + "░" * (20 - filled)
                self._append_line(f"  {feat['name'][:26]:<26} {bar}")
            self._append_line("")

    def _cancel(self, *_):
        # runpy threads can't be killed cleanly; signal done so UI unlocks
        if self._running:
            self._q.put(('done', -9))

    def _clear(self, *_):
        self._out_buf = ""
        self.out_lbl.text = ""

    # ── Worker infrastructure ─────────────────────────────────────────────────

    def _start_worker(self, status, fn):
        if self._running:
            self._append_line("[busy — already running, please wait]")
            return
        self._clear()
        self._running = True
        self.status_lbl.text = status
        for b in self._action_btns:
            b.disabled = True

        q = self._q

        def _wrap():
            old_argv = sys.argv
            old_out  = sys.stdout
            old_err  = sys.stderr
            sys.stdout = _Pipe(q, 'out')
            sys.stderr = _Pipe(q, 'err')
            s = _load_settings()
            if s.get('ODDS_API_KEY'):
                os.environ['ODDS_API_KEY'] = s['ODDS_API_KEY']
            try:
                fn()
            except Exception as e:
                import traceback
                q.put(('err', f"Error: {e}\n{traceback.format_exc()}\n"))
            finally:
                sys.argv   = old_argv
                sys.stdout = old_out
                sys.stderr = old_err
            q.put(('done', 0))

        threading.Thread(target=_wrap, daemon=True).start()

    def _exec(self, script, args):
        """Run a backend script via runpy inside the worker thread."""
        data_path = os.path.join(DATA_DIR, script)
        pyc_code  = None

        # Always copy from bundle so updated APKs refresh cached scripts.
        # 1. Try .py source in bundle / parent
        for sd in [BUNDLE_DIR, os.path.dirname(BUNDLE_DIR)]:
            src = os.path.join(sd, script)
            if os.path.exists(src):
                shutil.copy(src, data_path)
                break
        else:
            # 2. p4a bundles .pyc files flat at the bundle root (e.g. s6_tips.pyc)
            pyc_name = script[:-3] + '.pyc'  # s6_tips.py -> s6_tips.pyc
            for sd in [BUNDLE_DIR, os.path.dirname(BUNDLE_DIR)]:
                pyc = os.path.join(sd, pyc_name)
                if os.path.exists(pyc):
                    with open(pyc, 'rb') as fh:
                        fh.read(16)   # skip header: magic(4)+flags(4)+mtime(4)+size(4)
                        pyc_code = marshal.loads(fh.read())
                    break
            else:
                # 3. Fallback: __pycache__ path (older p4a versions)
                for sd in [BUNDLE_DIR, os.path.dirname(BUNDLE_DIR)]:
                    pyc = importlib.util.cache_from_source(os.path.join(sd, script))
                    if os.path.exists(pyc):
                        with open(pyc, 'rb') as fh:
                            fh.read(16)
                            pyc_code = marshal.loads(fh.read())
                        break
                else:
                    if not os.path.exists(data_path):
                        try:
                            bfiles = sorted(os.listdir(BUNDLE_DIR))[:15]
                        except Exception:
                            bfiles = ['<cannot list>']
                        raise FileNotFoundError(
                            f"{script} not found.\n"
                            f"  DATA_DIR:    {DATA_DIR}\n"
                            f"  BUNDLE_DIR:  {BUNDLE_DIR}\n"
                            f"  Bundle files: {bfiles}"
                        )

        old_argv = sys.argv
        sys.argv = [data_path] + list(args)
        try:
            if pyc_code is not None:
                import builtins
                exec(pyc_code, {
                    '__file__':     data_path,
                    '__name__':     '__main__',
                    '__doc__':      None,
                    '__package__':  None,
                    '__spec__':     None,
                    '__builtins__': builtins,
                })
            else:
                runpy.run_path(data_path, run_name='__main__')
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

    # ── Collect (Android-safe: no subprocess) ────────────────────────────────

    def _collect_step(self, label, script, args=()):
        self._exec(script, list(args))

    def _collect_step_safe(self, label, script, args=()):
        """Like _collect_step but swallows errors so the overall flow continues."""
        try:
            self._exec(script, list(args))
        except Exception as e:
            print(f"[warning] {label} failed — skipping ({e})", flush=True)

    def _do_collect_new(self):
        import pandas as pd
        data_path = os.path.join(DATA_DIR, 'nrl_source_data.csv')

        def existing_rounds():
            if not os.path.exists(data_path):
                return set()
            df = pd.read_csv(data_path)
            c  = df[df['winner'].notna()]
            return set(zip(c['season'].astype(int), c['round'].astype(int)))

        before = existing_rounds()
        self._collect_step("history", "s1_history.py", ["--new-only"])
        after      = existing_rounds()
        new_rounds = sorted(after - before)

        if not new_rounds:
            print("Already up to date.", flush=True)
            return

        for season, rnd in new_rounds:
            self._collect_step("stats",  "s2_stats.py",  ["--season", str(season), "--round", str(rnd)])
            self._collect_step("squads", "s4_squads.py", ["--season", str(season), "--round", str(rnd)])
        self._collect_step("weather", "s3_weather.py")
        self._collect_step_safe("odds", "s5_odds.py")
        print("Done.", flush=True)

    def _do_collect_all(self):
        self._collect_step("history", "s1_history.py")
        self._collect_step("stats",   "s2_stats.py")
        self._collect_step("squads",  "s4_squads.py")
        self._collect_step("weather", "s3_weather.py")
        self._collect_step_safe("odds", "s5_odds.py")
        print("Done.", flush=True)

    # ── Output pump (main thread, every 0.1s) ─────────────────────────────────

    def _pump(self, _dt):
        while not self._q.empty():
            try:
                kind, text = self._q.get_nowait()
            except queue.Empty:
                break
            if kind == 'done':
                self._running = False
                msg = "Done" if text == 0 else "Cancelled"
                self.status_lbl.text = msg
                for b in self._action_btns:
                    b.disabled = False
            elif kind == 'out':
                for ln in text.splitlines():
                    if ln:
                        self._append_line(ln)
                    else:
                        self._append('\n')
            elif kind == 'err':
                for ln in text.split('\n'):
                    if ln:
                        safe = (ln.replace('&', '&amp;')
                                  .replace('[', '&bl;')
                                  .replace(']', '&br;'))
                        self._append(f"[color={RED_HEX}]{safe}[/color]\n")

    def _append_line(self, line):
        s = line.strip()
        safe = (line.replace('&', '&amp;')
                    .replace('[', '&bl;')
                    .replace(']', '&br;'))

        if s.startswith("▶"):
            # Tip pick line — green bold
            self._append(f"[color={GREEN_HEX}][b]{safe}[/b][/color]\n")
        elif s.startswith("⚠"):
            # Players out — yellow
            self._append(f"[color={YELLOW_HEX}]{safe}[/color]\n")
        elif s.startswith("✔"):
            # Correct result
            self._append(f"[color={GREEN_HEX}]{safe}[/color]\n")
        elif s.startswith("✘"):
            # Wrong result
            self._append(f"[color={RED_HEX}]{safe}[/color]\n")
        elif s.startswith("Round ") and "·" in s:
            # Round header
            self._append(f"\n[b]{safe}[/b]\n")
        elif s.startswith("==="):
            # Section header (model info)
            self._append(f"\n[color={YELLOW_HEX}][b]{safe}[/b][/color]\n")
        elif s.startswith("RESULT:") or s.startswith("Odds:") or s.startswith("No Odds:") or s.startswith("Winner:"):
            # Summary lines — space before block
            self._append(f"\n[color={WHITE_HEX}][b]{safe}[/b][/color]\n")
        elif "error" in s.lower() or s.startswith("[!]"):
            self._append(f"[color={RED_HEX}][b]{safe}[/b][/color]\n")
        elif line.startswith("  ") or line.startswith("\t"):
            # Indented detail (model stats, progress sub-lines)
            self._append(f"[color={GREY_HEX}]{safe}[/color]\n")
        else:
            self._append(f"[color={WHITE_HEX}]{safe}[/color]\n")

    def _append(self, markup):
        self._out_buf += markup
        self.out_lbl.text = self._out_buf

    # ── Theme toggle ─────────────────────────────────────────────────────────

    def _apply_theme_rebuild(self, new_theme: str):
        """Switch to new_theme and rebuild the UI, preserving output."""
        _set_theme(new_theme)
        buf = self._out_buf
        self.clear_widgets()
        self._action_btns = []
        self._build_ui()
        Window.clearcolor = BG_C
        self._out_buf = buf
        self.out_lbl.text = buf

    def _toggle_theme(self, *_):
        new_theme = 'light' if _CURRENT_THEME == 'dark' else 'dark'
        s = _load_settings()
        s['theme'] = new_theme
        _save_settings(s)
        self._apply_theme_rebuild(new_theme)

    def _sync_system_theme(self, _dt):
        """Follow Android system theme automatically when no manual pref is saved."""
        if _load_settings().get('theme'):
            return   # user has a manual preference — don't override
        system = _detect_system_theme()
        if system != _CURRENT_THEME:
            self._apply_theme_rebuild(system)

    # ── Settings popup ────────────────────────────────────────────────────────

    def _settings_popup(self, *_):
        content = BoxLayout(orientation='vertical',
                            padding=dp(16), spacing=dp(10))
        s = _load_settings()

        content.add_widget(Label(
            text="ODDS API Key", color=WHITE_C,
            size_hint_y=None, height=dp(28), halign='left',
        ))
        key_inp = TextInput(
            text=s.get('ODDS_API_KEY', ''),
            multiline=False, password=True,
            size_hint_y=None, height=dp(48),
            background_color=_rgba(ACCENT_HEX),
            foreground_color=WHITE_C,
            font_size=dp(14),
        )
        content.add_widget(key_inp)
        save_btn = Button(
            text="Save", size_hint_y=None, height=dp(48),
            background_color=GREEN_C, background_normal='',
        )
        content.add_widget(save_btn)

        popup = Popup(
            title="Settings", content=content,
            size_hint=(0.88, None), height=dp(240),
        )

        def _save(*_):
            d = _load_settings()
            d['ODDS_API_KEY'] = key_inp.text.strip()
            _save_settings(d)
            popup.dismiss()

        save_btn.bind(on_press=_save)
        popup.open()


# ── App ───────────────────────────────────────────────────────────────────────

class NRLTipsApp(App):
    def build(self):
        global DATA_DIR
        self.title = "NRL Tips"
        try:
            if platform == 'android':
                from android.storage import app_storage_path  # type: ignore
                DATA_DIR = app_storage_path()
            _init_data()
            saved = _load_settings().get('theme')
            _set_theme(saved if saved else _detect_system_theme())
            return MainLayout()
        except Exception:
            import traceback
            tb = traceback.format_exc()
            # Write crash log to a file we can retrieve
            try:
                with open(os.path.join(DATA_DIR, 'crash.log'), 'w') as f:
                    f.write(tb)
            except Exception:
                pass
            err = Label(
                text=tb, font_size='11sp',
                color=(1, 0.3, 0.3, 1),
                halign='left', valign='top',
                size_hint_y=None,
                markup=False,
            )
            err.bind(width=lambda w, v: setattr(w, 'text_size', (v, None)),
                     texture_size=lambda w, v: setattr(w, 'height', v[1]))
            sv = ScrollView()
            sv.add_widget(err)
            return sv

    def on_start(self):
        if platform == 'android':
            from android.permissions import request_permissions, Permission  # type: ignore
            request_permissions([Permission.INTERNET])


if __name__ == '__main__':
    NRLTipsApp().run()
