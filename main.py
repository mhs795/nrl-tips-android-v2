#!/usr/bin/env python3
"""
NRL Tips v2 — Kivy Android app
Modernized version based on the desktop Python app.
"""

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
import datetime

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
        'WHITE':  "dfe6e9",
        'GREY':   "636e72",
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

BG_C = PANEL_C = ACCENT_C = GREEN_C = YELLOW_C = RED_C = WHITE_C = GREY_C = (0, 0, 0, 1)
BG_HEX = PANEL_HEX = ACCENT_HEX = OUTPUT_HEX = "000000"
GREEN_HEX = YELLOW_HEX = RED_HEX = WHITE_HEX = GREY_HEX = "ffffff"

def _set_theme(name: str):
    global _CURRENT_THEME
    global BG_HEX, PANEL_HEX, ACCENT_HEX, OUTPUT_HEX
    global GREEN_HEX, YELLOW_HEX, RED_HEX, WHITE_HEX, GREY_HEX
    global BG_C, PANEL_C, ACCENT_C, GREEN_C, YELLOW_C, RED_C, WHITE_C, GREY_C
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
    YELLOW_C = _rgba(YELLOW_HEX)
    RED_C    = _rgba(RED_HEX)
    WHITE_C  = _rgba(WHITE_HEX)
    GREY_C   = _rgba(GREY_HEX)

_set_theme('dark')

RADIUS = dp(14)

class RoundedButton(Button):
    def __init__(self, btn_color, radius=RADIUS, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ''
        self.background_down   = ''
        self.background_color  = (0, 0, 0, 0)
        self._base = btn_color
        self._radius = radius
        with self.canvas.before:
            self._ci   = Color(*btn_color)
            self._rr   = RoundedRectangle(pos=self.pos, size=self.size,
                                          radius=[self._radius])
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
DATA_DIR   = BUNDLE_DIR

_DATA_FILES = [
    'nrl_source_data.csv', 'nrl_model.npz', 'nrl_model_no_odds.npz',
    'nrl_model_info.json', 'nrl_model_no_odds_info.json',
    'round_template.csv', 'squad_player_cache.json',
]
_SCRIPT_FILES = [
    's1_history.py', 's2_stats.py', 's3_weather.py',
    's4_squads.py',  's5_odds.py',  's6_tips.py',
    's9_performance.py', 'm5_nrl.py', 'collect_data.py',
    'u1_travel.py',  'u2_weather.py', 'u3_squad.py', 'nrl_predict.py',
]

def _init_data():
    global DATA_DIR
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        # Copy data files
        for fname in _DATA_FILES:
            src = os.path.join(BUNDLE_DIR, fname)
            dst = os.path.join(DATA_DIR, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                try: shutil.copy(src, dst)
                except: pass
        
        # Copy scripts (prefer .py, but handle .pyc)
        for fname in _SCRIPT_FILES:
            base = fname[:-3]
            for ext in ['.py', '.pyc']:
                f = base + ext
                src = os.path.join(BUNDLE_DIR, f)
                dst = os.path.join(DATA_DIR, f)
                if os.path.exists(src):
                    try: shutil.copy(src, dst)
                    except: pass
                    
        if DATA_DIR not in sys.path:
            sys.path.insert(0, DATA_DIR)
    except Exception as e:
        print(f"Data init error: {e}")

# ── UI Components ─────────────────────────────────────────────────────────────

def _bg(widget, color, radius=0):
    with widget.canvas.before:
        Color(*color)
        r = (RoundedRectangle(pos=widget.pos, size=widget.size, radius=[dp(radius)])
             if radius else Rectangle(pos=widget.pos, size=widget.size))
    widget.bind(
        pos =lambda w, v: setattr(r, 'pos',  v),
        size=lambda w, v: setattr(r, 'size', v),
    )

class Header(BoxLayout):
    def __init__(self, title="NRL Tips", on_toggle_theme=None, on_settings=None, **kwargs):
        super().__init__(size_hint_y=None, height=dp(64),
                         padding=[dp(16), dp(8), dp(12), dp(8)], spacing=dp(12), **kwargs)
        _bg(self, ACCENT_C)
        
        icon_path = os.path.join(BUNDLE_DIR, 'icon.png')
        if os.path.exists(icon_path):
            self.add_widget(KivyImage(source=icon_path, size_hint=(None, None), size=(dp(40), dp(40))))
        
        self.add_widget(Label(text=f"[b]{title}[/b]", markup=True, font_size=dp(22), 
                              color=WHITE_C, halign='left', valign='middle'))
        
        if on_toggle_theme:
            theme_btn = Button(text="☀" if _CURRENT_THEME == 'dark' else "☾", font_name='DejaVuSans',
                               font_size=dp(20), size_hint=(None, 1), width=dp(44),
                               background_color=(0,0,0,0), color=WHITE_C)
            theme_btn.bind(on_press=on_toggle_theme)
            self.add_widget(theme_btn)
            
        if on_settings:
            cog = Button(text="[b]...[/b]", markup=True, font_size=dp(20),
                         size_hint=(None, 1), width=dp(44),
                         background_color=(0,0,0,0), color=WHITE_C)
            cog.bind(on_press=on_settings)
            self.add_widget(cog)

# ── Screens ───────────────────────────────────────────────────────────────────

class HomeScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        layout = BoxLayout(orientation='vertical')
        _bg(layout, BG_C)
        
        layout.add_widget(Header(on_toggle_theme=app.toggle_theme, on_settings=app.show_settings))
        
        scroll = ScrollView()
        content = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(20), size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))
        
        # Summary Card
        summary = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10), size_hint_y=None, height=dp(180))
        _bg(summary, PANEL_C, radius=16)
        summary.add_widget(Label(text="Season 2026", font_size=dp(24), bold=True, color=WHITE_C))
        
        stats_row = BoxLayout(spacing=dp(20))
        self.acc_lbl = Label(text="Loading...", font_size=dp(18), color=GREEN_C)
        self.rnd_lbl = Label(text="Next: R1", font_size=dp(18), color=YELLOW_C)
        stats_row.add_widget(self.acc_lbl)
        stats_row.add_widget(self.rnd_lbl)
        summary.add_widget(stats_row)
        content.add_widget(summary)
        
        # Quick Actions
        content.add_widget(Label(text="Quick Actions", font_size=dp(18), bold=True, color=GREY_C, size_hint_y=None, height=dp(30)))
        
        actions = GridLayout(cols=2, spacing=dp(12), size_hint_y=None, height=dp(140))
        b1 = RoundedButton(text="Get Latest Tips", btn_color=_rgba("1565c0"), color=WHITE_C)
        b1.bind(on_press=lambda *_: app.go_to_tips())
        b2 = RoundedButton(text="Compare Models", btn_color=_rgba("00695c"), color=WHITE_C)
        b2.bind(on_press=lambda *_: app.run_action("compare"))
        actions.add_widget(b1)
        actions.add_widget(b2)
        content.add_widget(actions)
        
        # Recent Performance
        content.add_widget(Label(text="Recent Performance", font_size=dp(18), bold=True, color=GREY_C, size_hint_y=None, height=dp(30)))
        self.perf_box = BoxLayout(orientation='vertical', padding=dp(15), spacing=dp(8), size_hint_y=None, height=dp(100))
        _bg(self.perf_box, PANEL_C, radius=16)
        self.perf_lbl = Label(text="No data yet", color=WHITE_C)
        self.perf_box.add_widget(self.perf_lbl)
        content.add_widget(self.perf_box)
        
        scroll.add_widget(content)
        layout.add_widget(scroll)
        self.add_widget(layout)
        
    def on_enter(self):
        self.refresh_stats()

    def refresh_stats(self):
        try:
            path = os.path.join(DATA_DIR, 'nrl_model_info.json')
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                self.acc_lbl.text = f"Accuracy: {d['cv_accuracy']:.1%}"
            else:
                self.acc_lbl.text = "Model: OK"
        except:
            self.acc_lbl.text = "Stats: --"

class TipsScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        layout = BoxLayout(orientation='vertical')
        _bg(layout, BG_C)
        
        nav = BoxLayout(size_hint_y=None, height=dp(64), padding=[dp(8), dp(8), dp(16), dp(8)], spacing=dp(12))
        _bg(nav, ACCENT_C)
        back = Button(text="<", font_size=dp(24), size_hint=(None, 1), width=dp(50), background_color=(0,0,0,0), color=WHITE_C)
        back.bind(on_press=lambda *_: app.go_home())
        nav.add_widget(back)
        nav.add_widget(Label(text="[b]Tips & Actions[/b]", markup=True, font_size=dp(20), color=WHITE_C, halign='left', valign='middle'))
        layout.add_widget(nav)
        
        content = BoxLayout(orientation='vertical', padding=dp(16), spacing=dp(16))
        
        rnd_row = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(12))
        rnd_row.add_widget(Label(text="Round:", size_hint_x=None, width=dp(70), color=WHITE_C))
        self.round_input = TextInput(text="auto", multiline=False, size_hint_x=None, width=dp(100), 
                                     background_color=_rgba(ACCENT_HEX), foreground_color=WHITE_C)
        rnd_row.add_widget(self.round_input)
        content.add_widget(rnd_row)
        
        grid = GridLayout(cols=1, spacing=dp(10))
        btns = [
            ("Tips (with Odds)", "1565c0", "tips"),
            ("Tips (No Odds)",   "1e88e5", "tips_no_odds"),
            ("Model Info",       "004d40", "model_info"),
            ("Collect New Data", "00796b", "collect_new"),
            ("Clear Cache",      "34495e", "clear"),
        ]
        for label, color, action in btns:
            b = RoundedButton(text=label, btn_color=_rgba(color), color=WHITE_C, size_hint_y=None, height=dp(56))
            b.bind(on_press=lambda w, a=action: app.run_action(a))
            grid.add_widget(b)
        
        content.add_widget(grid)
        content.add_widget(Widget())
        layout.add_widget(content)
        self.add_widget(layout)

class OutputScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        layout = BoxLayout(orientation='vertical')
        _bg(layout, BG_C)
        
        nav = BoxLayout(size_hint_y=None, height=dp(64), padding=[dp(8), dp(8), dp(12), dp(8)], spacing=dp(12))
        _bg(nav, ACCENT_C)
        back = Button(text="< Back", font_size=dp(18), size_hint=(None, 1), width=dp(80), background_color=(0,0,0,0), color=WHITE_C)
        back.bind(on_press=lambda *_: app.go_back_from_output())
        nav.add_widget(back)
        self.title_lbl = Label(text="Output", font_size=dp(18), bold=True, color=WHITE_C)
        nav.add_widget(self.title_lbl)
        
        clr_btn = Button(text="Clear", font_size=dp(14), size_hint=(None, 0.7), width=dp(60), pos_hint={'center_y': 0.5})
        clr_btn.bind(on_press=lambda *_: setattr(self.out_lbl, 'text', ""))
        nav.add_widget(clr_btn)
        
        self.cancel_btn = RoundedButton(text="Stop", btn_color=_rgba(RED_HEX), color=WHITE_C, size_hint=(None, 0.8), width=dp(70), pos_hint={'center_y': 0.5})
        self.cancel_btn.bind(on_press=app.cancel_action)
        nav.add_widget(self.cancel_btn)
        layout.add_widget(nav)
        
        sv = ScrollView()
        _bg(sv, _rgba(OUTPUT_HEX))
        self.out_lbl = Label(text="", markup=True, valign='top', halign='left', size_hint_y=None, 
                             font_size=dp(15), color=WHITE_C, padding=[dp(16), dp(16)], font_name='DejaVuSans')
        self.out_lbl.bind(width=lambda w, v: setattr(w, 'text_size', (v, None)),
                          texture_size=lambda w, v: setattr(w, 'height', v[1]))
        sv.add_widget(self.out_lbl)
        layout.add_widget(sv)
        
        self.status_lbl = Label(text="Ready", size_hint_y=None, height=dp(30), color=GREY_C, font_size=dp(13))
        layout.add_widget(self.status_lbl)
        
        self.add_widget(layout)

# ── Main App ──────────────────────────────────────────────────────────────────

class NRLTipsApp(App):
    def build(self):
        global DATA_DIR
        try:
            if platform == 'android':
                from android.storage import app_storage_path # type: ignore
                DATA_DIR = app_storage_path()
            _init_data()
            
            self.sm = ScreenManager(transition=SlideTransition())
            self.home = HomeScreen(self, name='home')
            self.tips = TipsScreen(self, name='tips')
            self.output = OutputScreen(self, name='output')
            
            self.sm.add_widget(self.home)
            self.sm.add_widget(self.tips)
            self.sm.add_widget(self.output)
            
            self._q = queue.Queue()
            self._running = False
            Clock.schedule_interval(self._pump, 0.1)
            
            return self.sm
        except Exception as e:
            import traceback
            err_box = BoxLayout(orientation='vertical', padding=dp(20))
            err_box.add_widget(Label(text="[b]Startup Crash[/b]", markup=True, font_size=dp(24), color=(1,0,0,1)))
            sv = ScrollView()
            lbl = Label(text=traceback.format_exc(), font_size=dp(12), color=(1,1,1,1), valign='top', halign='left', size_hint_y=None)
            lbl.bind(width=lambda w,v: setattr(w, 'text_size', (v, None)), texture_size=lambda w,v: setattr(w, 'height', v[1]))
            sv.add_widget(lbl)
            err_box.add_widget(sv)
            return err_box

    def go_home(self):
        self.sm.transition.direction = 'right'
        self.sm.current = 'home'
        
    def go_to_tips(self):
        self.sm.transition.direction = 'left'
        self.sm.current = 'tips'
        
    def go_back_from_output(self):
        self.sm.transition.direction = 'right'
        self.sm.current = 'tips' if self.sm.previous_screen.name == 'tips' else 'home'

    def toggle_theme(self, *_):
        new = 'light' if _CURRENT_THEME == 'dark' else 'dark'
        _set_theme(new)
        Window.clearcolor = BG_C

    def show_settings(self, *_):
        content = BoxLayout(orientation='vertical', padding=dp(16), spacing=dp(10))
        current_key = ""
        settings_path = os.path.join(DATA_DIR, 'android_settings.json')
        if os.path.exists(settings_path):
            try:
                with open(settings_path) as f:
                    s = json.load(f)
                    current_key = s.get('ODDS_API_KEY', '')
            except: pass
        content.add_widget(Label(text="ODDS API Key", color=WHITE_C, size_hint_y=None, height=dp(28)))
        key_inp = TextInput(text=current_key, multiline=False, password=True, size_hint_y=None, height=dp(48), background_color=_rgba(ACCENT_HEX), foreground_color=WHITE_C)
        content.add_widget(key_inp)
        save_btn = Button(text="Save", size_hint_y=None, height=dp(48), background_color=GREEN_C, background_normal='')
        content.add_widget(save_btn)
        popup = Popup(title="Settings", content=content, size_hint=(0.9, None), height=dp(250))
        def _save(*_):
            d = {}
            if os.path.exists(settings_path):
                try:
                    with open(settings_path) as f: d = json.load(f)
                except: pass
            d['ODDS_API_KEY'] = key_inp.text.strip()
            with open(settings_path, 'w') as f: json.dump(d, f)
            popup.dismiss()
        save_btn.bind(on_press=_save)
        popup.open()

    def run_action(self, action):
        rnd = self.tips.round_input.text.strip()
        args = ["--round", rnd] if rnd and rnd.lower() != "auto" and rnd.isdigit() else []
        self.output.title_lbl.text = action.replace("_", " ").title()
        self.output.out_lbl.text = ""
        self.sm.transition.direction = 'left'
        self.sm.current = 'output'
        if action == "tips":
            self._start_worker("Fetching tips...", lambda: self._exec('s6_tips.py', args))
        elif action == "tips_no_odds":
            self._start_worker("Fetching tips (no odds)...", lambda: self._exec('s6_tips.py', args + ["--no-odds"]))
        elif action == "compare":
            self._start_worker("Comparing models...", lambda: self._exec('s9_performance.py', ["--compare"] + args))
        elif action == "collect_new":
            self._start_worker("Collecting data...", lambda: self._exec('collect_data.py', []))
        elif action == "model_info":
            self._show_model_info()
        elif action == "clear":
            self._clear_cache()

    def _clear_cache(self):
        import glob
        files = glob.glob(os.path.join(DATA_DIR, "tips_cache_*.txt"))
        for f in files:
            try: os.remove(f)
            except: pass
        self.output.out_lbl.text = f"Cleared {len(files)} cache files.\n"
        self.output.status_lbl.text = "Done"

    def _start_worker(self, status, fn):
        if self._running: return
        self._running = True
        self.output.status_lbl.text = status
        def _wrap():
            try: fn()
            except Exception as e: 
                import traceback
                self._q.put(('err', f"Worker error: {e}\n{traceback.format_exc()}"))
            finally:
                self._q.put(('done', 0))
        threading.Thread(target=_wrap, daemon=True).start()

    def _exec(self, script, args):
        """Run a backend script. Prefers subprocess for .py, falls back to exec() for .py/.pyc."""
        import subprocess
        import marshal
        import builtins
        mod_name = script[:-3]
        py_name = script
        pyc_name = mod_name + '.pyc'
        target_py = None
        target_pyc = None
        for sd in [DATA_DIR, BUNDLE_DIR]:
            if os.path.exists(os.path.join(sd, py_name)):
                target_py = os.path.join(sd, py_name)
                break
            if os.path.exists(os.path.join(sd, pyc_name)):
                target_pyc = os.path.join(sd, pyc_name)
                break
        if not target_py and not target_pyc:
            self._q.put(('err', f"Script not found: {script}"))
            return
        # Try subprocess first if .py exists
        if target_py:
            cmd = [sys.executable, target_py] + list(args)
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{DATA_DIR}:{BUNDLE_DIR}:{env.get('PYTHONPATH', '')}"
            env["PYTHONUNBUFFERED"] = "1"
            try:
                settings_path = os.path.join(DATA_DIR, 'android_settings.json')
                if os.path.exists(settings_path):
                    with open(settings_path) as f:
                        s = json.load(f)
                        if s.get('ODDS_API_KEY'): env['ODDS_API_KEY'] = s['ODDS_API_KEY']
            except: pass
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=DATA_DIR, env=env, text=True, bufsize=1)
                for line in self.proc.stdout:
                    if line.strip(): self._q.put(('out', line.strip()))
                self.proc.wait()
                return
            except: pass
        # Fallback: exec()
        file_to_run = target_py or target_pyc
        try:
            if file_to_run.endswith('.pyc'):
                with open(file_to_run, 'rb') as f:
                    f.read(16)
                    code_obj = marshal.loads(f.read())
            else:
                with open(file_to_run, 'r') as f:
                    code_obj = compile(f.read(), file_to_run, 'exec')
            old_stdout, old_stderr, old_argv = sys.stdout, sys.stderr, sys.argv
            class Capturer:
                def __init__(self, q, kind): self.q, self.kind = q, kind
                def write(self, t):
                    if t.strip(): self.q.put((self.kind, t.strip()))
                def flush(self): pass
            sys.stdout = Capturer(self._q, 'out')
            sys.stderr = Capturer(self._q, 'err')
            sys.argv = [file_to_run] + list(args)
            try:
                exec(code_obj, {'__file__': file_to_run, '__name__': '__main__', '__builtins__': builtins})
            except SystemExit: pass
            finally:
                sys.stdout, sys.stderr, sys.argv = old_stdout, old_stderr, old_argv
        except Exception as e:
            self._q.put(('err', f"Execution failed: {e}"))

    def _pump(self, _):
        while not self._q.empty():
            kind, text = self._q.get_nowait()
            if kind == 'done':
                self._running = False
                self.output.status_lbl.text = "Done"
            elif kind == 'out':
                if text: self._append_formatted(text)
            elif kind == 'err':
                if text: self.output.out_lbl.text += f"[color={RED_HEX}]{text}[/color]\n"

    def _append_formatted(self, line):
        s = line.strip()
        if s.startswith("▶") or s.startswith("TIP:"):
            self.output.out_lbl.text += f"[color={GREEN_HEX}][b]{line}[/b][/color]\n"
        elif s.startswith("⚠") or "⚡" in s:
            self.output.out_lbl.text += f"[color={YELLOW_HEX}]{line}[/color]\n"
        elif s.startswith("Round ") or s.startswith("==="):
            self.output.out_lbl.text += f"\n[b]{line}[/b]\n"
        elif "error" in s.lower() or s.startswith("[!]"):
            self.output.out_lbl.text += f"[color={RED_HEX}][b]{line}[/b][/color]\n"
        else:
            self.output.out_lbl.text += f"{line}\n"

    def cancel_action(self, *_):
        if self._running:
            self._q.put(('done', -1))
            self.output.out_lbl.text += "\n[color=ff0000]Cancelled[/color]\n"

    def _show_model_info(self):
        self.output.out_lbl.text = ""
        for path, label in [(os.path.join(DATA_DIR, 'nrl_model_info.json'), "With Odds"), (os.path.join(DATA_DIR, 'nrl_model_no_odds_info.json'), "No Odds")]:
            if not os.path.exists(path):
                self.output.out_lbl.text += f"[color={YELLOW_HEX}][{label}][/color] No model info found.\n\n"
                continue
            try:
                with open(path) as f:
                    d = json.load(f)
                self.output.out_lbl.text += f"[b][color={YELLOW_HEX}]=== {label} ===[/color][/b]\n"
                self.output.out_lbl.text += f"  CV acc:    {d['cv_accuracy']:.3f} ± {d['cv_std']:.3f}\n"
                self.output.out_lbl.text += f"  Train acc: {d['train_accuracy']:.3f}\n"
                self.output.out_lbl.text += f"  Brier:     {d['brier_score']:.4f}\n"
                self.output.out_lbl.text += "  Top features:\n"
                for feat in d.get("features", [])[:8]:
                    filled = int(feat["importance"] * 20)
                    bar = "█" * filled + "░" * (20 - filled)
                    self.output.out_lbl.text += f"  {feat['name'][:20]:<20} {bar}\n"
                self.output.out_lbl.text += "\n"
            except Exception as e:
                self.output.out_lbl.text += f"Error loading {label}: {e}\n"
        self.output.status_lbl.text = "Done"

if __name__ == '__main__':
    NRLTipsApp().run()
