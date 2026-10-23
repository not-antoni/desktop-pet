"""Desktop Pet.

A pixel buddy that lives on the desktop: walks the taskbar, perches on window
titlebars, climbs window frames, reacts to what you are doing, and optionally
talks for itself through Groq.

Modules:
    ai_companion        model calls, rate limits, memory, screenshot policy
    window_tracker      OS window/ledge/edge geometry (cached, thread-safe)
    screen_capture      pixel grabbing + blocklist redaction
    companion_dialogue  every canned line, for every companion

Behaviour worth knowing about:
- One QTimer drives everything at 20 FPS. `game_loop` is the only writer of
  the pet's position, and every Qt slot is exception-guarded: PyQt5 turns an
  unhandled exception in a slot into qFatal()/abort(), so a bad ledge must not
  be able to take the process down.
- The pet works in Qt *logical* pixels. OS window APIs report native pixels,
  so `window_tracker` is told the display's devicePixelRatio once and does the
  conversion.
- All OS window work happens on a background poller thread. Nothing in the
  game loop touches the OS.
"""

import functools
import math
import os
import random
import sys
import threading
import time
import traceback

# HiDPI attributes must be set before QApplication exists, so this import has
# to stay above main().
from PyQt5.QtCore import Qt, QTimer, QPoint, QObject, pyqtSignal, QRect
from PyQt5.QtGui import (QPainter, QPixmap, QColor, QFont, QCursor,
                         QFontMetrics, QPolygon, QFontDatabase,
                         QIcon, QImage)
from PyQt5.QtWidgets import QApplication, QWidget, QMenu, QAction

import ai_companion
import pet_state
import pet_sounds
import pet_fx
import pet_personality as persona
import screen_capture
import wayland_capture
from focus_timer import FocusWatcher, MODE_FOCUS

try:
    import window_tracker
except Exception:
    window_tracker = None

try:
    import ocr_guard
except Exception:
    ocr_guard = None

try:
    import companion_dialogue
except Exception:
    companion_dialogue = None

try:
    from PyQt5.QtWidgets import QSystemTrayIcon
except Exception:  # pragma: no cover - some Qt builds lack it
    QSystemTrayIcon = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKINS_DIR = os.path.join(BASE_DIR, "skins")
LOG_PATH = os.path.join(BASE_DIR, "pet.log")

# Keep the log from growing without bound: it is appended to by every guarded
# failure and nothing ever truncated it.
LOG_MAX_BYTES = 512 * 1024

# The pet's own window must never be a ledge for the pet to stand on.
PET_WINDOW_TITLE = "desktop-pet-overlay"

# Sprite state basenames, in load order. Each skin ships a subset.
STATE_FILES = [
    "sit", "idle", "blink", "groom", "spook", "stalk", "walk",
    "run", "fly", "swat", "pat", "sleep", "climb", "drag",
    "jump", "fall", "happy", "pet", "eat", "dizzy", "yawn", "dance",
    "alert", "sulk",
]

# ---- Animation: ticks between frame advances (main timer is 50ms = 20 FPS)
FRAME_SPEED = {
    "SIT": 12, "IDLE": 12, "WALK": 6, "RUN": 4, "SLEEP": 15, "GROOM": 8,
    "STALK": 7, "BLINK": 5, "SWAT": 4, "CLIMB": 5, "DRAG": 6, "FALL": 4,
    "PERCH": 12, "PAT": 3, "SPOOK": 4, "JUMP": 5, "POUNCE": 3, "CHASE": 4,
    "LEAP": 6, "HAPPY": 4, "PETTED": 10, "EAT": 6, "DIZZY": 5, "DANCE": 6,
    "ALERT": 6, "SULK": 14, "FLY": 6, "HAUNT": 7, "PHASE": 4, "ZOOMIES": 3,
    "SNIFF": 9, "SPIN": 4, "FOLLOW": 6,
}

# One-shot states: play for a fixed number of ticks, then settle back down.
# They are what turn a click, a treat or a shake into a visible reaction.
ONE_SHOT_TICKS = {
    "HAPPY": 32, "PETTED": 36, "EAT": 72, "DIZZY": 64, "DANCE": 120,
    "ALERT": 18, "SULK": 60, "SPOOK": 30, "PAT": 40,
}

# Chasing: lined up with the cursor means within this many px; inside it the
# pet stops turning (no flip-flopping on the spot).
STEER_DEADZONE = 10
# How far above the pet a pounce can reach.
POUNCE_REACH_PX = 70

# Free-flight and personality states (see pet_personality.py)
FLY_MAX_TICKS = 420
HAUNT_TICKS = 220
HAUNT_DISTANCE = 85
PHASE_TICKS = 28
ZOOMIES_SPEED = 5.2
SNIFF_TICKS = 170
SPIN_TICKS = 30
FOLLOW_TICKS = 240
# A flyer's cruising band, as fractions of the screen height from the top.
FLY_BAND = {"ghost": (0.15, 1.0), "jarvis": (0.35, 1.0)}
# Coming back to the machine after this long away gets a welcome.
WELCOME_AWAY_SECONDS = 120

# Visual-only poses layered over a state for a few ticks (no FSM change):
# the yawn as the pet nods off or wakes up, the crouch after a landing.
YAWN_TICKS = 28
LAND_SQUASH_TICKS = 4

# Cursor flicks near the pet catch its attention.
ALERT_RADIUS_PX = 170
ALERT_SPEED_PX = 38         # px per tick of cursor travel
ALERT_COOLDOWN_TICKS = 20 * 45
# Dancing and other idle flourishes are rolled this rarely.
DANCE_CHANCE = 0.03
# Shaking the pet while dragging it: this many direction reversals within
# SHAKE_WINDOW seconds makes it dizzy.
SHAKE_REVERSALS = 5
SHAKE_WINDOW = 1.4
LASER_SECONDS = 30
# Babble for lines the pet volunteers is rarer still than its ambient sounds.
AMBIENT_BABBLE_CHANCE = 0.8
# Characters revealed per tick in the speech bubble (typewriter effect).
SPEECH_REVEAL_PER_TICK = 2
# A spoken line counts as an answer to the user (and babbles reliably) if
# the user did something this recently.
USER_MOMENT_TICKS = 20 * 12
# What a pat sounds like, per companion.
PET_CUES = {"cat": "purr", "shiba": "pop", "ghost": "pop", "jarvis": "beep",
            "ultron": "pop"}

# ---- State durations, in ticks (20 ticks = 1s)
STATE_MIN_DURATION = {
    "SIT": 60, "IDLE": 60, "WALK": 40, "GROOM": 60, "SLEEP": 200,
    "STALK": 60, "BLINK": 6, "SWAT": 20, "PERCH": 40,
}
STATE_MAX_DURATION = {
    "SIT": 110, "IDLE": 110, "WALK": 100, "GROOM": 100, "SLEEP": 300,
    "STALK": 120, "RUN": 90, "PERCH": 120,
}

# Transitions are only rolled on this interval, so the pet is calm rather than
# twitchy.
TRANSITION_EVAL_INTERVAL = 40

# Physics
GRAVITY = 0.8          # px/tick^2
TERMINAL_VEL = 12.0    # px/tick
CLIMB_SPEED = 2.5      # px/tick upward (50 px/s)
CLIMB_GRACE_TICKS = 30  # added to the computed budget before giving up
JUMP_VELOCITY = 7.5    # px/tick upward for a playful hop in place
CHASE_SPEED = 3.4      # px/tick while chasing the cursor

# Leaping. A hop straight up only clears ~35px, which is not enough to reach any
# window titlebar, so the pet launches itself ballistically at whatever it
# decided it wanted.
#
# The leap gets its own gravity and no terminal-velocity clamp, so the arc is a
# clean parabola that can be solved for exactly: the pet lands on the titlebar
# it aimed at instead of wherever the integrator happens to end up. These bound
# how far it will commit to one jump.
LEAP_GRAVITY = 3.0      # px/tick^2, snappier than falling
LEAP_ARC = 1.35        # apex overshoot: the arc has to clear the target and
                       # come back down onto it, not graze it at the top
LEAP_MAX_RISE = 820.0   # px above the pet: a full-height window from the floor
LEAP_MAX_REACH = 520.0  # px horizontally from the pet's centre
LEAP_VX_MAX = 9.0       # px/tick horizontal cap
LEAP_AIR_DRAG = 0.985   # horizontal bleed while airborne
LEAP_MAX_TICKS = 140    # ~7s: past this the leap has clearly missed
LEAP_GOAL_WALK = 900.0  # how far the pet will walk to line up under a window
LEAP_GOAL_SPEED = 1.7   # px/tick while heading for a goal, vs 1.1 ambling
LEAP_GOAL_MAX_TICKS = 320

# Energy: drained by exertion, restored by rest. Drives the pet's mood and how
# likely it is to nap instead of playing.
ENERGY_DRAIN = {
    "WALK": 0.02, "STALK": 0.09, "RUN": 0.16, "CHASE": 0.18, "CLIMB": 0.07,
    "JUMP": 0.12, "POUNCE": 0.14, "SWAT": 0.05, "GROOM": 0.01,
    "DANCE": 0.06, "HAPPY": 0.04, "ZOOMIES": 0.22, "FOLLOW": 0.04,
    "SNIFF": 0.02, "HAUNT": 0.05, "FLY": 0.02,
}
ENERGY_GAIN_IDLE = 0.05     # sitting around
ENERGY_GAIN_SLEEP = 0.55    # asleep
ENERGY_GAIN_PET = 15.0      # being petted
ENERGY_GAIN_WAKE = 40.0     # waking up startled

MOOD_SLEEPY_MAX = 30.0   # below this energy: sleepy
MOOD_HAPPY_MIN = 70.0    # above this energy and recently petted: happy

# Ticks between polls of the focused window (5s) and between unsolicited
# contextual lines (30s).
FOCUS_POLL_TICKS = 100
CONTEXT_CHECK_INTERVAL = 600
SAVE_INTERVAL_TICKS = 600   # 30s at 20 FPS
# How often the config file is checked for edits (5s), and how often the
# fullscreen state is looked at (1s).
CONFIG_RELOAD_TICKS = 100
FULLSCREEN_CHECK_TICKS = 20
RAISE_TICKS = 40
TICKS_PER_SECOND = 20.0
# A context line is handed to the model at most this often, and only when the
# idle rate-limit pool agrees.
AI_CONTEXT_CHANCE = 0.35
# A context check only counts as activity if the focused window actually
# changed, otherwise the "is anyone there" gate is satisfied forever.
CURSOR_ACTIVITY_PX = 3

# A drag release keeps its momentum: below FLING_MIN_SPEED it is a placement,
# above that the pet is thrown (and lands with an in-character comment).
FLING_MIN_SPEED = 1.5
FLING_MAX_SPEED = 6.0

SCALE_PRESETS = {
    "1x (Tiny)": 1,
    "2x (Small)": 2,
    "3x (Medium)": 3,
    "4x (Large)": 4,
    "5x (Huge)": 5,
    "6x (Giant)": 6,
    "8x (Kaiju)": 8,
}

# FSM states that count as "exertion" for the purposes of chattiness.
CALM_STATES = ("SIT", "IDLE", "SLEEP", "PERCH")

# ---- Speech bubble
SPEECH_FONT_FAMILY = "Consolas"
SPEECH_FONT_SIZE = 9
SPEECH_MAX_LINES = 3
SPEECH_PADDING_X = 10
SPEECH_PADDING_Y = 6
SPEECH_BUBBLE_RADIUS = 6
SPEECH_TAIL_H = 6
SPEECH_PIXEL = 2   # the bubble's "art pixel", in screen pixels
SPEECH_MIN_WIDTH = 60
SPEECH_MAX_WIDTH = 220
SPEECH_DISPLAY_TICKS = 80  # 4 seconds
SPEECH_MAX_CHARS = 120

# How long after a context menu closes before a "Look At This" capture fires.
# Long enough for the menu to be unmapped and its shadow to disappear.
MENU_CLOSE_DELAY_MS = 450


# =====================================================================
# Crash safety
# =====================================================================

_log_lock = threading.Lock()


def log_exception(context=""):
    """Record a handled exception to pet.log.

    PyQt5 aborts the process on an unhandled exception raised inside a slot
    (it calls qFatal), so slots are guarded and anything that still escapes
    gets written down here instead of taking the pet with it.
    """
    text = traceback.format_exc()
    try:
        with _log_lock:
            _rotate_log_unlocked()
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write("[%s] %s\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                          context or "exception", text))
    except Exception:
        pass
    if os.environ.get("PET_DEBUG"):
        sys.stderr.write(text)


def _rotate_log_unlocked():
    """Rename pet.log to pet.log.1 when it gets too big.

    Called under `_log_lock` from both the rotating writer and startup.
    """
    try:
        if os.path.getsize(LOG_PATH) < LOG_MAX_BYTES:
            return
        rotated = LOG_PATH + ".1"
        try:
            os.unlink(rotated)
        except OSError:
            pass
        os.replace(LOG_PATH, rotated)
    except OSError:
        pass


def rotate_log():
    """Startup size check (see _rotate_log_unlocked)."""
    with _log_lock:
        _rotate_log_unlocked()


def guarded(slot):
    """Wrap a Qt slot so a failure logs instead of aborting the process."""
    @functools.wraps(slot)
    def wrapper(self, *args, **kwargs):
        try:
            return slot(self, *args, **kwargs)
        except Exception:
            log_exception("%s.%s" % (type(self).__name__, slot.__name__))
            return None
    return wrapper


# =====================================================================
# System status (cached: /proc reads do not need to happen per quote)
# =====================================================================

STATUS_TTL = 30.0
_status_cache = {"at": 0.0, "mem": None, "batt": None}


def _mem_percent():
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MS()
            stat.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.dwMemoryLoad)
        with open("/proc/meminfo") as f:
            data = f.read().splitlines()
        total = int([l for l in data if l.startswith("MemTotal")][0].split()[1])
        avail = int([l for l in data if l.startswith("MemAvailable")][0].split()[1])
        return int(100 * (1 - avail / total))
    except Exception:
        return None


def _battery_percent():
    """Battery charge, from the machine's own battery where possible.

    The old code returned the first `/sys/class/power_supply` entry that had a
    `capacity` file, which on a laptop with a mouse, a headset or a UPS also
    plugged in can be any of those: the pet then announced "low battery" for
    a peripheral. `type` has to say "Battery", a non-present bay is skipped,
    and the main battery is preferred by name when there is more than one.
    """
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class SPS(ctypes.Structure):
                _fields_ = [("ACLineStatus", ctypes.c_byte),
                            ("BatteryFlag", ctypes.c_byte),
                            ("BatteryLifePercent", ctypes.c_byte),
                            ("Reserved1", ctypes.c_byte),
                            ("BatteryLifeTime", ctypes.c_ulong),
                            ("BatteryFullLifeTime", ctypes.c_ulong)]

            status = SPS()
            ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
            return status.BatteryLifePercent if 0 <= status.BatteryLifePercent <= 100 else None
        base = "/sys/class/power_supply"
        candidates = []
        for name in sorted(os.listdir(base)):
            directory = os.path.join(base, name)
            cap_path = os.path.join(directory, "capacity")
            if not os.path.exists(cap_path):
                continue
            if _read_sys_file(os.path.join(directory, "present")) == "0":
                continue
            kind = (_read_sys_file(os.path.join(directory, "type")) or "").lower()
            if kind and kind != "battery":
                continue          # mains adapter, USB, mouse, UPS...
            candidates.append((0 if name.upper().startswith("BAT") else 1, name, cap_path))
        for _rank, _name, cap_path in sorted(candidates):
            raw = _read_sys_file(cap_path)
            if raw is None:
                continue
            try:
                value = int(float(raw))
            except (TypeError, ValueError):
                continue
            if 0 <= value <= 100:
                return value
        return None
    except Exception:
        return None


def _read_sys_file(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


def system_status():
    """(mem_percent, battery_percent), recomputed at most every 30s."""
    now = time.monotonic()
    if now - _status_cache["at"] > STATUS_TTL:
        _status_cache["mem"] = _mem_percent()
        _status_cache["batt"] = _battery_percent()
        _status_cache["at"] = now
    return _status_cache["mem"], _status_cache["batt"]


def get_system_status_quote(skin):
    """Fallback line when the model is unavailable or rate-limited.

    Picks a system-state line when something notable is happening, otherwise a
    plain line from the companion's own dialogue table -- which is the single
    source of each companion's voice, shared with the contextual lines.
    """
    mem, batt = system_status()
    event = None
    if batt is not None and batt < 20:
        event = "low_battery"
    elif mem is not None and mem > 85:
        event = "low_memory"
    if event and companion_dialogue is not None:
        return companion_dialogue.get_contextual_quote(skin, event_type=event)
    if companion_dialogue is not None:
        return companion_dialogue.get_contextual_quote(skin, event_type="idle")
    return "..."


# =====================================================================
# Speech bubble layout
# =====================================================================

def _elide(text, font_metrics, max_width):
    """Trim `text` with an ellipsis so it fits `max_width`.

    Never returns empty for non-empty input: a bubble with nothing in it looks
    broken, and a single character wider than the box should still say
    something.
    """
    ellipsis = "..."
    if font_metrics.horizontalAdvance(text) <= max_width:
        return text
    trimmed = text
    while trimmed and font_metrics.horizontalAdvance(trimmed + ellipsis) > max_width:
        trimmed = trimmed[:-1]
    return (trimmed.rstrip() + ellipsis) if trimmed else ellipsis


def _wrap_text(text, font_metrics, max_width, max_lines=SPEECH_MAX_LINES):
    """Word-wrap into at most `max_lines` lines, eliding the overflow.

    The old version rebuilt the final line with `words.index(word)`, which
    finds the *first* occurrence of a word, so any text that repeated a word
    before the cut point lost or duplicated words. This indexes by position.
    """
    words = (text or "").split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for index in range(1, len(words)):
        word = words[index]
        candidate = current + " " + word
        if font_metrics.horizontalAdvance(candidate) <= max_width:
            current = candidate
            continue
        if len(lines) == max_lines - 1:
            # Last line available: everything left over goes here, elided.
            remainder = " ".join([current] + words[index:])
            lines.append(_elide(remainder, font_metrics, max_width))
            return lines
        lines.append(current)
        current = word

    lines.append(current)
    return lines[:max_lines]


def _compute_bubble_geometry(text, font):
    """(lines, width, height) for a bubble. Height includes the tail."""
    metrics = QFontMetrics(font)
    content_max_w = SPEECH_MAX_WIDTH - 2 * SPEECH_PADDING_X
    lines = _wrap_text(text, metrics, content_max_w)

    widest = max((metrics.horizontalAdvance(line) for line in lines), default=0)
    bubble_w = max(SPEECH_MIN_WIDTH, widest + 2 * SPEECH_PADDING_X)
    bubble_w = min(bubble_w, SPEECH_MAX_WIDTH)

    bubble_h = metrics.height() * len(lines) + 2 * SPEECH_PADDING_Y + SPEECH_TAIL_H
    return lines, bubble_w, bubble_h


def discover_skins():
    """Skin directory names, sorted. Used by the widget and by main()."""
    try:
        found = [d for d in os.listdir(SKINS_DIR)
                 if os.path.isdir(os.path.join(SKINS_DIR, d))
                 and not d.startswith("_")]
        return sorted(found)
    except Exception:
        return []


def strong_on_top():
    """Whether to use an unmanaged (override-redirect) window on X11.

    A managed "always on top" window still loses to a focused fullscreen
    window in Mutter/Muffin/KWin's stacking layers -- the pet vanished behind
    every fullscreen video. An unmanaged window sits above all of them. Set
    behaviour.always_on_top to "normal" to go back to the managed kind.
    """
    try:
        mode = ai_companion.general_settings().get("behaviour", {}).get(
            "always_on_top", "strong")
    except Exception:
        mode = "strong"
    app = QApplication.instance()
    platform = app.platformName() if app is not None else ""
    return mode == "strong" and platform == "xcb"


def window_flags(click_through=False):
    flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
    if strong_on_top():
        flags |= Qt.X11BypassWindowManagerHint
    if click_through:
        flags |= Qt.WindowTransparentForInput
    return flags


def _parse_clock(text):
    """'23:00' -> 1380 minutes past midnight, or None."""
    try:
        hours, _, minutes = str(text).partition(":")
        return (int(hours) % 24) * 60 + int(minutes or 0) % 60
    except (TypeError, ValueError):
        return None


class AIBridge(QObject):
    """Relays model replies from worker threads back onto the GUI thread."""

    line_ready = pyqtSignal(str, str)  # (text, status)
    notice = pyqtSignal(str)           # worker-side status, no model text


class DesktopPet(QWidget):
    """Transparent always-on-top widget holding the pet, its FSM and its UI."""

    def __init__(self, skin="cat", scale=2):
        super().__init__()
        self.setWindowTitle(PET_WINDOW_TITLE)
        self.setWindowFlags(window_flags(False))
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.NoFocus)

        # ---- sprites ----
        self.scale_factor = scale
        self.sprites = {}
        # Frame lists are resolved once per skin (see _build_frame_table) and
        # mirrored copies are cached, rather than rebuilt on every repaint.
        self._frame_table = {}
        self._flipped_cache = {}
        self.sprite_w = 32 * scale
        self.sprite_h = 32 * scale
        self.available_skins = discover_skins()
        self.current_skin = skin if skin in self.available_skins else (
            self.available_skins[0] if self.available_skins else "cat")
        self.load_skin(self.current_skin)
        self.heart_pixmap = self._make_heart()
        # ---- screen geometry ----
        self.screen_rect = (0, 0, 1280, 720)  # absolute, refreshed below
        self.screen_left = 0
        self.screen_top = 0
        self.screen_w = 1280
        self.screen_h = 720
        self.floor_y = 0.0
        self._refresh_screen_geometry()

        # ---- position (float internally, int for move()) ----
        self.x_pos = float(self.screen_left + self.screen_w // 2 - self.sprite_w // 2)
        self.y_pos = self.floor_y

        # ---- physics / FSM ----
        self.vy = 0.0
        self.vx = 0.0            # horizontal velocity, airborne only
        self.land_event = None   # what to say when this landing happens
        self.goal_ledge = None   # the window titlebar the pet is heading for
        self.leap_target = None  # the titlebar the current leap is aiming at
        self.leap_y0 = self.leap_x0 = 0.0
        self.leap_vy0 = self.leap_vx0 = 0.0
        self.leap_t = 0.0
        # Recently visited titlebars, so the pet tours the desktop instead of
        # ping-ponging between the same two windows forever.
        self.recent_ledges = []
        self.facing = 1
        self.state = "SIT"
        self.state_timer = 0
        self.tick_count = 0
        self.energy = 100.0
        self.target_x = None
        self.petted_until_tick = None
        self.session_start = time.time()

        # ---- drag ----
        self.dragging = False
        self.drag_offset = QPoint()
        self.drag_start_y = 0.0

        # ---- speech ----
        self.speech_text = ""
        self.speech_lines = []
        self.speech_bubble_w = 0
        self.speech_bubble_h = 0  # total bubble height including tail
        self.speech_timer = 0
        self.speech_below = False  # bubble drawn under the sprite when no room above
        self.speech_font = QFont(SPEECH_FONT_FAMILY, SPEECH_FONT_SIZE, QFont.Bold)

        self.heart_timer = 0

        # ---- skin flags / hover ----
        self.is_ghost = self.current_skin == "ghost"
        self.flies = persona.flies(self.current_skin)
        self.profile = persona.profile(self.current_skin)
        self.ghost_hover_y = self.floor_y - 6.0

        # ---- climbing / perching ----
        self.climb_enabled = True
        self.climb_target_edge = None
        self.climb_budget_ticks = 0
        self.current_ledge = None
        # Movement extras, toggled from the menu.
        self.allow_jump = True
        self.allow_chase = True

        # ---- context tracking ----
        self.last_cursor = QCursor.pos()
        self.last_activity_tick = 0
        self.last_window_title = None
        self.last_window_proc = ""

        # ---- AI ----
        self.awaiting_ai = False
        self.ai_deadline = 0
        self.ai_request_kind = "idle"   # 'idle' | 'user' | 'screenshot'
        self._ai_timed_out = False      # a canned line already covered this one
        self.ai_bridge = AIBridge(self)
        self.ai_bridge.line_ready.connect(self._on_ai_line)
        self.ai_bridge.notice.connect(self._on_notice)

        # ---- window tracker ----
        self.tracker = None
        if window_tracker is not None:
            try:
                self.tracker = window_tracker.get_tracker()
                self.tracker.set_coordinate_scale(self._display_scale())
                self.tracker.set_screen_rect(self.screen_rect)
                # Never generate ledges from the pet's own window.
                self.tracker.add_excluded_pid(os.getpid())
                self.tracker.set_ignored_windows(self._behaviour().get("ignore_windows", []))
                self.tracker.start_background()
            except Exception:
                log_exception("tracker init")
                self.tracker = None

        self.interaction = (companion_dialogue.InteractionTracker()
                            if companion_dialogue is not None else None)

        # ---- screenshot state ----
        self._screenshot_inflight = False
        self._pending_capture = False
        # Fullscreen apps (games, video) are not for perching on, and not for
        # interrupting: the pet hides for the duration.
        self._hidden_for_fullscreen = False
        self._capture_was_visible = False
        self._screenshot_source = "auto"   # 'auto' | 'manual' | 'local'
        # "Shush..." (session-only snooze), 0 = not snoozed.
        self._snoozed_until = 0

        # ---- interaction extras ----
        self.click_through = False
        self._treat_until_tick = 0
        self.tray = None
        self.tray_menu = None
        self._drag_history = []
        self.sounds = pet_sounds.PetSounds(
            (ai_companion.CONFIG.get("audio") or {})
            if ai_companion.CONFIG else {})

        # ---- juice: particles, reactions, speech reveal ----
        self.fx = None
        try:
            self.fx = pet_fx.PetFX(PET_WINDOW_TITLE, bypass=strong_on_top())
        except Exception:
            log_exception("fx overlay")
        self._squash_ticks = 0
        self._yawn_until = 0
        self._alert_ready_tick = 0
        self._laser_until = 0
        self._shake_times = []
        self._shake_dir = 0
        self._shaken = False
        self._user_moment = -USER_MOMENT_TICKS
        self._heart_seen = 0
        self.speech_reveal = 0
        self._state_before_press = None
        self._state_timer_before_press = 0
        self._suppress_release = False
        self._user_drop = False

        # ---- personality ----
        self.chooser = persona.Chooser()
        self._next_action = None        # a follow-up the last action asked for
        self._last_explore_tick = 0
        self._fly_x = self._fly_y = 0.0
        self._zoom_laps = 0
        self._cursor = QPoint(0, 0)
        self._heavy_landing = False
        self._was_away = False

        # ---- focus / pomodoro awareness ----
        focus_cfg = self._settings().get("focus", {})
        self.focus = FocusWatcher(
            focus_cfg.get("state_files") or [],
            focus_cfg.get("poll_seconds", 5)) if focus_cfg.get("enabled", True) \
            else FocusWatcher([], 5)
        self.focus.poll(force=True)

        # ---- persistent state ----
        self.state_store = None
        self._restore_state()

        # ---- monitor change handling ----
        self._screen_signals = set()
        self._bind_screen_signals()
        app = QApplication.instance()
        if app is not None:
            for signal in ("screenAdded", "screenRemoved", "primaryScreenChanged"):
                try:
                    getattr(app, signal).connect(self._on_screen_change)
                except Exception:
                    pass

        self._sync_widget_geometry()
        self.show()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.game_loop)
        self.timer.start(50)

    # =================================================================
    # Config
    # =================================================================

    @staticmethod
    def _settings():
        return ai_companion.general_settings()

    def _behaviour(self):
        return self._settings().get("behaviour", {})

    # =================================================================
    # Persistent state
    # =================================================================

    def _state_path(self):
        cfg = self._settings().get("state", {})
        name = cfg.get("file") or "pet_state.json"
        return name if os.path.isabs(name) else os.path.join(BASE_DIR, name)

    def _restore_state(self):
        """Come back as we were, minus the energy spent while we were away."""
        cfg = self._settings().get("state", {})
        if not cfg.get("enabled", True):
            return
        try:
            store = pet_state.PetState(
                self._state_path(),
                save_interval=cfg.get("save_interval_seconds", 30),
                energy_decay_per_hour=cfg.get("energy_decay_per_hour",
                                               pet_state.ENERGY_DECAY_PER_HOUR))
            data = store.load()
            self.state_store = store

            skin = data.get("skin")
            if skin and skin in self.available_skins and skin != self.current_skin:
                self.load_skin(skin)
            scale = data.get("scale")
            if scale in SCALE_PRESETS.values() and scale != self.scale_factor:
                self.scale_factor = scale
                self.load_skin(self.current_skin)

            self.energy = store.rested_energy()
            self.climb_enabled = bool(data.get("climb_enabled", True))
            # The menu toggles win over the config file: they are the more
            # recent, explicit choice, and they now survive a restart.
            behaviour = self._behaviour()
            self.allow_jump = bool(data.get("allow_jump",
                                            behaviour.get("allow_jump", True)))
            self.allow_chase = bool(data.get("allow_chase",
                                             behaviour.get("allow_chase", True)))
            self.click_through = bool(data.get("click_through", False))

            data["pet_count"] = (data.get("pet_count") or 0) + 1
            store.touch(pet_count=data["pet_count"])
            store.save(force=True)

            # Only resume a position that still lands on a connected display,
            # and prefer the monitor the pet was left on: a laptop unplugged
            # from its dock has to put the pet back on the laptop screen.
            self._restore_display(data.get("display"))
            x, y = data.get("x"), data.get("y")
            if x is not None and y is not None:
                screen = self._screen_for_point(x + self.sprite_w / 2,
                                               y + self.sprite_h / 2)
                if screen is not None and screen.geometry().contains(int(x), int(y)):
                    self.x_pos, self.y_pos = float(x), float(y)
                    self.current_ledge = None
                    self.vy = 0.0
                    self._enter_sit()
                    if not self._is_on_ledge():
                        self.y_pos = self.floor_y
            self._apply_click_through()
        except Exception:
            log_exception("state restore")

    def _restore_display(self, name):
        """Follow the pet back to the monitor it was left on, if it is here."""
        if not name:
            return
        app = QApplication.instance()
        if app is None:
            return
        for screen in app.screens():
            if screen.name() == name:
                if self._screen_for_point(self.x_pos, self.y_pos) is not screen:
                    self.screen_left = screen.geometry().x()
                    self.screen_top = screen.geometry().y()
                    self.screen_w = screen.geometry().width()
                    self.screen_h = screen.geometry().height()
                    self.x_pos = float(self.screen_left + self.screen_w // 2
                                       - self.sprite_w // 2)
                    self.y_pos = float(self.screen_top + self.screen_h
                                       - self.sprite_h)
                return

    def _save_state(self, force=False):
        if self.state_store is None:
            return
        try:
            self.state_store.touch(
                skin=self.current_skin, scale=self.scale_factor,
                energy=self.energy, x=self.x_pos, y=self.y_pos,
                display=self._screen_name(),
                climb_enabled=self.climb_enabled,
                allow_jump=self.allow_jump,
                allow_chase=self.allow_chase,
                click_through=self.click_through,
                seconds_alive=self.state_store.total_seconds_alive(
                    time.time() - self.session_start),
            )
            self.state_store.save(force=force)
        except Exception:
            log_exception("state save")

    def _check_milestone(self):
        """Once per completed day of lifetime: one line, then wait a day.

        Uses the accumulated lifetime counter, so the line shows up on the
        anniversary of the pet's whole life together, not of a single session.
        """
        store = self.state_store
        if store is None or companion_dialogue is None:
            return
        try:
            alive = store.total_seconds_alive(time.time() - self.session_start)
        except Exception:
            return
        days = int(alive // 86400)
        if days <= int(store.data.get("milestone_days", 0)):
            return
        store.data["milestone_days"] = days
        # Persist immediately: a crash five minutes after the anniversary must
        # not let the same milestone repeat.
        self._save_state(force=True)
        if not self._quiet_now():
            self.sounds.play("fanfare")   # it's a party, unless it's 3am
        line = companion_dialogue.get_contextual_quote(
            self.current_skin, event_type="milestone")
        self.speak(line.replace("{days}", str(days)))

    def _screen_name(self):
        try:
            screen = self._screen_for_point(self.x_pos + self.sprite_w / 2,
                                            self.y_pos + self.sprite_h / 2)
            return screen.name() if screen is not None else None
        except Exception:
            return None

    def forget_state(self):
        if self.state_store is not None:
            self.state_store.forget()
        self.energy = 100.0

    # =================================================================
    # Screen / monitor geometry
    # =================================================================

    def _display_scale(self):
        """devicePixelRatio of the primary screen, for HiDPI coordinate fixes."""
        try:
            screen = QApplication.primaryScreen()
            return float(screen.devicePixelRatio()) if screen else 1.0
        except Exception:
            return 1.0

    def _screen_for_point(self, x, y):
        """The screen containing a point, else the primary screen."""
        app = QApplication.instance()
        if app is None:
            return None
        for screen in app.screens():
            if screen.geometry().contains(int(x), int(y)):
                return screen
        return app.primaryScreen()

    @guarded
    def _on_screen_change(self, *_args):
        """Monitor plugged, unplugged, or resolution changed."""
        self._bind_screen_signals()
        self._refresh_screen_geometry()
        if self.tracker is not None:
            self.tracker.set_coordinate_scale(self._display_scale())
            self.tracker.set_screen_rect(self.screen_rect)
            self.tracker.invalidate_cache()
        # If the pet was left off the visible desktop, bring it back.
        screen = self._screen_for_point(
            self.x_pos + self.sprite_w / 2, self.y_pos + self.sprite_h / 2)
        if screen is not None and not screen.geometry().contains(
                int(self.x_pos + self.sprite_w / 2), int(self.y_pos + self.sprite_h / 2)):
            self._reset_position()
        if self.flies:
            # keep a flyer's cruising height on this screen (a geometry
            # refresh happens every few seconds; it must not ground it)
            self.ghost_hover_y = max(float(self.screen_top),
                                     min(getattr(self, "ghost_hover_y",
                                                 self.floor_y),
                                         self.floor_y - 6.0))
        self._sync_widget_geometry()

    def _bind_screen_signals(self):
        """Follow the monitors' own geometry signals.

        `screenAdded`/`screenRemoved` never fire for a resolution change or a
        taskbar moving, so the floor was stale after a dock/undock or a
        rotate, and the pet stood in mid-air until something else happened to
        refresh it. `geometryChanged`/`availableGeometryChanged` do fire.
        """
        app = QApplication.instance()
        if app is None:
            return
        for screen in app.screens():
            for emitter in (getattr(screen, "geometryChanged", None),
                            getattr(screen, "availableGeometryChanged", None)):
                if emitter is None:
                    continue
                try:
                    if id(emitter) in self._screen_signals:
                        continue
                    emitter.connect(self._on_screen_change)
                    self._screen_signals.add(id(emitter))
                except Exception:
                    pass

    def _refresh_screen_geometry(self):
        """Recompute the display box the pet lives in.

        The pet is confined to the screen it currently occupies, not the whole
        virtual desktop, so dragging it to a second monitor keeps working. The
        rect keeps its origin: treating a secondary monitor's coordinates as
        starting at zero is what used to put the pet outside its own floor.
        """
        center_x = int(getattr(self, "x_pos", 0) + self.sprite_w / 2)
        center_y = int(getattr(self, "y_pos", 0) + self.sprite_h / 2)
        screen = self._screen_for_point(center_x, center_y) or QApplication.primaryScreen()
        if screen is None:
            return

        geometry = screen.geometry()
        self.screen_left = geometry.x()
        self.screen_top = geometry.y()
        self.screen_w = geometry.width()
        self.screen_h = geometry.height()
        self.screen_rect = (self.screen_left, self.screen_top,
                            self.screen_left + self.screen_w,
                            self.screen_top + self.screen_h)
        # Stand on top of the taskbar/panel when the platform tells us where
        # the usable area ends, rather than guessing a margin.
        available = screen.availableGeometry()
        floor_bottom = available.y() + available.height()
        self.floor_y = float(floor_bottom - self.sprite_h)

    # =================================================================
    # Asset loading
    # =================================================================

    def _make_heart(self):
        """Load the heart sprite, sized to the pet rather than to a constant.

        It used to be scaled to 18x16 no matter what: a speck next to a 4x
        pet, and half the pet's size at 1x.
        """
        path = os.path.join(SKINS_DIR, "_heart.png")
        if not os.path.exists(path):
            return None
        source = QPixmap(path)
        if source.isNull():
            return None
        # half the pet's pixel size, but always a whole multiple: the heart is
        # pixel art too
        factor = max(1, int(self.scale_factor) // 2)
        return source.scaled(source.width() * factor, source.height() * factor,
                             Qt.KeepAspectRatio, Qt.FastTransformation)

    def _pixel_ratio(self):
        """Integer device pixel ratio for pixel-perfect sprites.

        On a 2x screen a 4x pet is drawn with 8 physical pixels per art pixel
        instead of being upscaled again by the compositor. Fractional ratios
        (1.25, 1.5) cannot be pixel-perfect, so they fall back to 1 and let
        Qt place whole logical pixels.
        """
        try:
            ratio = float(self.devicePixelRatioF())
        except Exception:
            ratio = 1.0
        return int(ratio) if ratio >= 1.0 and ratio == int(ratio) else 1

    def _load_scaled(self, path):
        """A skin frame scaled by a whole number, nearest-neighbour.

        Scaling goes through QImage at the exact multiple so every art pixel
        becomes an identical N x N block -- no smoothing, no uneven columns.
        """
        image = QImage(path)
        if image.isNull():
            return QPixmap()
        ratio = self._pixel_ratio()
        factor = max(1, int(self.scale_factor)) * ratio
        image = image.convertToFormat(QImage.Format_ARGB32_Premultiplied)
        image = image.scaled(image.width() * factor, image.height() * factor,
                             Qt.IgnoreAspectRatio, Qt.FastTransformation)
        pix = QPixmap.fromImage(image)
        pix.setDevicePixelRatio(ratio)
        return pix

    @staticmethod
    def _logical_size(pix):
        ratio = pix.devicePixelRatio() or 1.0
        return int(round(pix.width() / ratio)), int(round(pix.height() / ratio))

    @staticmethod
    def _mirrored(pix):
        """Horizontal mirror that keeps the device pixel ratio.

        QPixmap.transformed() resets the ratio to 1, which on a HiDPI screen
        draws every left-facing frame at double size.
        """
        out = QPixmap.fromImage(pix.toImage().mirrored(True, False))
        out.setDevicePixelRatio(pix.devicePixelRatio())
        return out

    def load_skin(self, skin_name):
        """Load every frame a skin has, and rebuild the frame lookup table."""
        skin_path = os.path.join(SKINS_DIR, skin_name)
        sprites = {}
        if os.path.isdir(skin_path):
            for state in STATE_FILES:
                frames = []
                index = 0
                while True:
                    path = os.path.join(skin_path, "%s_%d.png" % (state, index))
                    if not os.path.exists(path):
                        break
                    pix = self._load_scaled(path)
                    if not pix.isNull():
                        frames.append(pix)
                    index += 1
                if not frames:
                    path = os.path.join(skin_path, "%s.png" % state)
                    if os.path.exists(path):
                        pix = self._load_scaled(path)
                        if not pix.isNull():
                            frames.append(pix)
                if frames:
                    sprites[state] = frames

        self.sprites = sprites
        self.current_skin = skin_name
        self.is_ghost = skin_name == "ghost"
        self.flies = persona.flies(skin_name)
        self.profile = persona.profile(skin_name)
        if getattr(self, "chooser", None) is not None:
            self.chooser.forget()
        self._flipped_cache = {}

        for frames in sprites.values():
            if frames:
                self.sprite_w, self.sprite_h = self._logical_size(frames[0])
                break

        self._build_frame_table()
        self._refresh_screen_geometry()
        if self.flies:
            self.ghost_hover_y = self.floor_y - 6.0
        # Rebuild the heart at the new scale: 16px art next to a 4x pet used to
        # look like a speck.
        self.heart_pixmap = self._make_heart()

    def _build_frame_table(self):
        """Resolve each FSM state to a concrete frame list, once."""
        sprites = self.sprites
        fallbacks = {
            "SIT":    ["sit", "idle"],
            "IDLE":   ["idle", "sit"],
            "BLINK":  ["blink", "sit", "idle"],
            "GROOM":  ["groom", "spook", "pat", "sit", "idle"],
            "STALK":  ["stalk", "fly", "run", "walk"],
            "WALK":   ["walk", "fly", "sit", "idle"],
            "RUN":    ["run", "fly", "walk"],
            "SWAT":   ["swat", "pat", "sit"],
            "SLEEP":  ["sleep", "sit", "idle"],
            "CLIMB":  ["climb", "walk", "sit"],
            "DRAG":   ["drag", "sit", "idle"],
            "FALL":   ["drag", "climb", "sit", "idle"],
            "PERCH":  ["sit", "idle", "walk"],
            "PAT":    ["pat", "sit", "idle"],
            "SPOOK":  ["spook", "sit", "idle"],
            "JUMP":   ["jump", "walk", "stalk", "climb", "sit"],
            "POUNCE": ["jump", "swat", "stalk", "walk", "sit"],
            "CHASE":  ["run", "stalk", "walk", "sit"],
            "LEAP":   ["jump", "climb", "walk", "stalk", "sit"],
            "HAPPY":  ["happy", "pat", "sit", "idle"],
            "PETTED": ["pet", "happy", "sit", "idle"],
            "EAT":    ["eat", "groom", "sit", "idle"],
            "DIZZY":  ["dizzy", "drag", "sit", "idle"],
            "DANCE":  ["dance", "happy", "pat", "sit", "idle"],
            "ALERT":  ["alert", "sit", "idle"],
            "SULK":   ["sulk", "sit", "idle"],
            "FLY":    ["fly", "walk", "idle", "sit"],
            "HAUNT":  ["fly", "walk", "idle", "sit"],
            "PHASE":  ["spook", "idle", "sit"],
            "ZOOMIES": ["run", "walk", "sit"],
            "SNIFF":  ["stalk", "walk", "sit"],
            "SPIN":   ["sit", "idle"],
            "FOLLOW": ["walk", "fly", "sit"],
        }
        # FALL prefers its own spread-eagle frames when the skin has them
        if sprites.get("fall"):
            fallbacks["FALL"] = ["fall"] + fallbacks["FALL"]
        table = {}
        for state, names in fallbacks.items():
            for name in names:
                frames = sprites.get(name)
                if frames:
                    table[state] = frames
                    break
            else:
                table[state] = sprites.get("sit") or sprites.get("idle") or []
        self._frame_table = table

    def get_frames(self):
        return self._frame_table.get(self.state) or []

    AIRBORNE = ("JUMP", "POUNCE", "LEAP")

    def _frame_choice(self):
        """(cache key, frame list, index) for this tick.

        Mostly "the state's frames, cycled", with a few poses chosen by what
        the body is doing rather than by the clock: airborne frames follow
        the velocity (launch going up, spread-eagle coming down), a landing
        shows a crouch for a moment, and the first and last moments of a
        nap show a yawn.
        """
        state = self.state
        frames = self.get_frames()
        speed = FRAME_SPEED.get(state, 6)
        index = self.tick_count // speed
        sprites = self.sprites
        if state in self.AIRBORNE and sprites.get("jump"):
            if self.vy < 0 or state == "LEAP" and self.leap_t < 2:
                jump = sprites["jump"]
                return "jump", jump, len(jump) - 1
            fall = sprites.get("fall")
            if fall:
                return "fall", fall, index
        if getattr(self, "_squash_ticks", 0) > 0 and sprites.get("jump") \
                and state in ("SIT", "IDLE", "PERCH", "WALK"):
            return "jump", sprites["jump"], 0
        yawn = sprites.get("yawn")
        if yawn and (state == "SLEEP" and self.state_timer < YAWN_TICKS
                     or getattr(self, "_yawn_until", 0) > self.tick_count
                     and state in ("SIT", "IDLE")):
            elapsed = (self.state_timer if state == "SLEEP"
                       else YAWN_TICKS - (self._yawn_until - self.tick_count))
            step = max(1, YAWN_TICKS // len(yawn))
            return "yawn", yawn, min(len(yawn) - 1, max(0, elapsed) // step)
        return state, frames, index

    def get_current_frame(self):
        """Current frame, mirrored when facing left. No per-call allocation."""
        key, frames, index = self._frame_choice()
        if not frames:
            return None
        if self.facing < 0 and self.state != "CLIMB":
            mirrored = self._flipped_cache.get(key)
            if mirrored is None or len(mirrored) != len(frames):
                mirrored = [self._mirrored(f) for f in frames]
                self._flipped_cache[key] = mirrored
            frames = mirrored
        return frames[index % len(frames)]

    # =================================================================
    # Widget geometry
    # =================================================================

    def _sync_widget_geometry(self):
        """Keep the sprite anchored at (x_pos, y_pos) and grow upward for the
        bubble, clamped so a bubble near a screen edge stays visible."""
        bubble_h = self.speech_bubble_h if self.speech_text else 0
        width = max(self.sprite_w, self.speech_bubble_w if self.speech_text else 0)
        height = self.sprite_h + bubble_h

        # Bubble sits above the sprite; if there is no room above, flip it
        # below rather than pushing the sprite off the top of the screen.
        y = self.y_pos - bubble_h
        if y < self.screen_top and bubble_h:
            y = self.y_pos
            self.speech_below = True
        else:
            self.speech_below = False

        x = int(self.x_pos)
        if bubble_h and not self.speech_below:
            # Keep the bubble inside the screen horizontally.
            x = max(self.screen_left,
                    min(x, self.screen_left + self.screen_w - width))

        # Avoid pointless relayout: resizing/moving a translucent always-on-top
        # window every tick is what causes repaint storms.
        if self.width() != width or self.height() != height:
            self.resize(width, height)
        target = QPoint(x, int(y))
        if self.pos() != target:
            self.move(target)

    # =================================================================
    # Speech
    # =================================================================

    def speak(self, text):
        if not text:
            return
        text = " ".join(str(text).split())[:SPEECH_MAX_CHARS]
        self.speech_text = text
        lines, width, height = _compute_bubble_geometry(text, self.speech_font)
        self.speech_lines = lines
        self.speech_bubble_w = width
        self.speech_bubble_h = height
        # the bubble is sized for the whole line up front; the text types in
        # (and the display time is extended by however long that takes)
        self.speech_reveal = 0
        reveal_ticks = len(text) // SPEECH_REVEAL_PER_TICK
        self.speech_timer = SPEECH_DISPLAY_TICKS + reveal_ticks
        self._babble(text)
        self._sync_widget_geometry()

    def _mark_user_moment(self):
        """The user just did something: the next line is an answer to them."""
        self._user_moment = self.tick_count

    def _babble(self, text):
        """Voice the line: always when answering the user, rarely otherwise."""
        sounds = getattr(self, "sounds", None)
        if sounds is None:
            return
        try:
            answering = self.tick_count - self._user_moment <= USER_MOMENT_TICKS
            if answering:
                sounds.say(text, self.current_skin)
            elif not self._quiet_now() and not self.in_focus() \
                    and random.random() < AMBIENT_BABBLE_CHANCE:
                sounds.say(text, self.current_skin, ambient=True)
        except Exception:
            log_exception("babble")

    def _clear_speech(self):
        self.speech_text = ""
        self.speech_reveal = 0
        self.speech_lines = []
        self.speech_bubble_w = 0
        self.speech_bubble_h = 0
        self._sync_widget_geometry()

    @guarded
    def _on_ai_line(self, text, status):
        """Model reply, on the GUI thread (queued signal).

        `status` decides whether the pet speaks at all: a rate-limited or
        deliberately skipped request must stay silent instead of firing off an
        unsolicited random line, which is what used to happen every 15 minutes
        once the screenshot caps were hit.

        A user-initiated request always says *something*: if the model was
        throttled ("skipped") or unavailable, the companion's canned line
        covers it. Silently doing nothing when the user clicks Speak reads as
        a broken menu item.
        """
        kind = self.ai_request_kind
        self.awaiting_ai = False
        late = self._ai_timed_out
        self._ai_timed_out = False

        if status == ai_companion.STATUS_OK and text:
            if late and self.speech_timer > 0:
                # A newer line is already on screen: a reply that missed its
                # own deadline must not overwrite it mid-sentence.
                return
            self.speak(text)
            return

        if kind == "user" and status in (ai_companion.STATUS_UNAVAILABLE,
                                         ai_companion.STATUS_SKIPPED):
            self.speak(get_system_status_quote(self.current_skin))

    def request_dialogue(self, user_initiated=True):
        """Ask for a line: model first, then the canned fallback."""
        if self._quiet_now() and not user_initiated:
            return
        if ai_companion.ai_enabled():
            self.awaiting_ai = True
            self.ai_deadline = self.tick_count + 90
            self.ai_request_kind = "user" if user_initiated else "idle"
            ai_companion.maybe_idle_line(
                self.current_skin, self.ai_bridge.line_ready.emit,
                kind="user" if user_initiated else "idle")
            return
        self.speak(get_system_status_quote(self.current_skin))

    def _quiet_now(self, now=None):
        """Whether unsolicited chatter is being suppressed right now.

        Two switches: the daily `quiet_hours` window, and the session-only
        snooze from the "Shush..." menu item (handy for a film or a call,
        without having to edit config). Only the pet's own initiative is
        muted: clicking Speak, petting it, or asking for a look always
        answers. Opening the menu clears a snooze early.
        """
        if now is None:
            epoch = time.time()
        else:
            epoch = now
        if self._snoozed_until and epoch < self._snoozed_until:
            return True
        if self._snoozed_until and epoch >= self._snoozed_until:
            # Expired: clear it so the status line stops saying "snoozed".
            self._snoozed_until = 0
        cfg = self._behaviour().get("quiet_hours") or {}
        if not cfg.get("enabled", False):
            return False
        try:
            start = _parse_clock(cfg.get("start", "23:00"))
            end = _parse_clock(cfg.get("end", "07:00"))
        except Exception:
            return False
        if start is None or end is None:
            return False
        current = time.localtime(epoch)
        minutes = current.tm_hour * 60 + current.tm_min
        if start == end:
            return False
        if start < end:
            return start <= minutes < end
        return minutes >= start or minutes < end   # wraps midnight

    def snooze_for(self, seconds):
        """Shush unsolicited chatter for a while. 0 cancels."""
        if seconds <= 0:
            self._snoozed_until = 0
            return
        self._snoozed_until = time.time() + seconds
        self._speak_contextual("snooze")

    @guarded
    def _menu_shush(self, minutes):
        if minutes <= 0:
            self.snooze_for(0)
            self.speak("unshushed.")
            return
        self.snooze_for(minutes * 60)

    def _snooze_status_text(self):
        if not self._snoozed_until:
            return ""
        remaining = self._snoozed_until - time.time()
        if remaining <= 0:
            return ""
        return "shushed for %d more min" % math.ceil(remaining / 60.0)

    def _speak_contextual(self, event_type, **details):
        """Speak a canned line from the companion's dialogue table."""
        if companion_dialogue is None:
            return
        line = companion_dialogue.get_contextual_quote(
            self.current_skin, event_type=event_type, **details)
        self.speak(line)

    # =================================================================
    # Screenshots
    # =================================================================

    def _hide_for_capture(self):
        """Unmap the pet so it is not in its own screenshot.

        Its speech bubble ("one sec, looking") is exactly the sort of thing
        the model would otherwise comment on.
        """
        self._capture_was_visible = self.isVisible()
        if self._capture_was_visible:
            self.hide()

    def _show_after_capture(self):
        if self._capture_was_visible and not self._hidden_for_fullscreen:
            self.show()
            self._restack_on_top()

    def _restack_on_top(self, force=False):
        """Re-assert always-on-top after the window was unmapped.

        Hiding and showing a window gives the window manager a new mapping,
        and the "above" state is not guaranteed to survive it: the pet came
        back *behind* the window it was covering, which is exactly what
        happens after every Look At This. Re-applying the flags forces the
        request back onto the WM; raise_() puts it back on top of the stack.
        """
        try:
            if not force and not self.isVisible():
                return
            self.setWindowFlags(window_flags(self.click_through))
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setAttribute(Qt.WA_ShowWithoutActivating)
            self.show()
            self.raise_()
        except Exception:
            log_exception("restack")

    def maybe_screenshot(self, force=False):
        """Occasionally grab the screen for a comment.

        The policy (interval, hourly and daily caps, AI availability) lives in
        ai_companion so there is one implementation of it. Pixels are grabbed
        here on the GUI thread because QPixmap demands it; redaction, the OCR
        content filter, JPEG encoding and the API call all happen on a worker.

        `force` is the "Look at this" menu action: it skips the interval and
        hourly caps, but never the daily cap, the blocklist or the scan --
        privacy is not a quota.
        """
        if self._screenshot_inflight:
            return
        if self._hidden_for_fullscreen:
            return
        # Config edits apply immediately, so switching screenshots off in the
        # file actually switches them off in the running pet.
        ai_companion.reload_config_if_changed()
        settings = ai_companion.screenshot_settings()
        app = QApplication.instance()

        manual = bool(force)
        if not manual:
            # Hard gate: an automatic capture only ever happens when the
            # feature is enabled. Nothing else may produce one.
            if not settings.get("enabled", False):
                return
            if self._quiet_now():
                return
            required = float(settings.get("require_recent_activity_minutes", 3)) * 60 * 20
            if self.tick_count - self.last_activity_tick > required:
                return
            if not ai_companion.screenshot_slot_ok():
                return
        else:
            if not ai_companion.ai_enabled():
                # No model to describe a JPEG to, but the capture policy still
                # applies, the OCR pass is local, and a canned line keyed by
                # what is on screen costs nothing. Nothing is transmitted.
                self._look_locally(app, settings)
                return
            if not ai_companion.reserve_forced_screenshot():
                # The result of this used to be dropped, so a user-initiated
                # capture ignored the daily cap entirely.
                ai_companion.note_capture_decision("refused", "daily capture cap reached")
                self.sounds.play("no")
                self._speak_contextual("screenshot_blocked")
                return

        # Native Wayland: the compositor does not let us enumerate windows, so
        # the blocklist cannot be honoured and prepare_capture refuses. The
        # portal path covers the one legitimate manual case: the user asked,
        # and the portal dialog makes THEM the privacy gate. Automatic
        # captures never reach here -- the enabled gate above is hard.
        if manual and window_tracker.wayland_session() and \
                settings.get("wayland_portal", False):
            ok, why = self._capture_via_portal(settings)
            if not ok and why:
                ai_companion.note_capture_decision("refused", why)
                self.sounds.play("no")
                self._speak_contextual("screenshot_blocked")
            return

        self._screenshot_source = "manual" if manual else "auto"
        self._hide_for_capture()
        try:
            image, reason = screen_capture.prepare_capture(
                app, settings.get("blocklist", []), self.screen_rect)
        finally:
            self._show_after_capture()
        if image is None:
            # Blocked or unavailable: release the slot, and say so once in a
            # while so a refusal is visible rather than mysterious.
            ai_companion.record_capture_attempt(False)
            ai_companion.note_capture_decision("refused", reason or "no capture")
            if reason:
                self._speak_contextual("screenshot_blocked")
            return

        self._screenshot_inflight = True
        skin = self.current_skin
        quality = int(settings.get("jpeg_quality", 70))
        blocklist = settings.get("blocklist", [])
        ocr_config = ai_companion.ocr_settings()
        screen_rect = self.screen_rect
        manual_flag = manual

        threading.Thread(
            target=self._capture_worker_for(image, manual=manual_flag,
                                            skin=skin, quality=quality,
                                            blocklist=blocklist,
                                            ocr_config=ocr_config,
                                            screen_rect=screen_rect),
            name="screenshot", daemon=True).start()

    def _capture_worker_for(self, image, manual, skin, quality, blocklist,
                            ocr_config, screen_rect):
        """The background half of a capture: scan, discard-or-encode, send.

        Built as a closure over one already-grabbed image so both capture
        front-ends (native grab, portal) share exactly one send policy.
        """

        def worker():
            # QImage work: safe off the GUI thread, and this is where the
            # 100-800ms used to be spent freezing the pet.
            captured = False
            try:
                if screen_rect is not None:
                    screen_capture.redact(image, blocklist, screen_rect)
                allowed, why = screen_capture.scan_capture(image, ocr_config)
                if not allowed:
                    # Discard, do not send, do not keep: the whole point of
                    # the filter is that this image never leaves the machine.
                    ai_companion.note_capture_decision("discarded", why)
                    self._report_discarded_capture()
                    return
                encoded = screen_capture.encode_jpeg_base64(image, quality)
                captured = bool(encoded)
                if encoded:
                    ai_companion.note_capture_decision("sent", "")
                    ai_companion.request_screenshot_comment(
                        skin, encoded, self.ai_bridge.line_ready.emit,
                        manual=manual)
            except Exception:
                log_exception("screenshot worker")
            finally:
                ai_companion.record_capture_attempt(captured)
                self._screenshot_inflight = False

        return worker

    def _capture_via_portal(self, settings):
        """A user-confirmed screenshot on native Wayland, via the portal.

        Returns (started, reason). The result is asynchronous: the pet is
        hidden while the compositor dialog is up, and the callback picks the
        normal pipeline back up once the user has answered.
        """
        if wayland_capture.QT_DBUS_OK is False:
            return False, "the portal needs QtDBus, which this PyQt5 lacks"
        ok, why = wayland_capture.available()
        if not ok:
            return False, why
        self._screenshot_source = "portal"
        self._screenshot_inflight = True
        self._hide_for_capture()

        def on_result(image, reason):
            self._show_after_capture()
            if image is None:
                self._screenshot_inflight = False
                ai_companion.note_capture_decision("refused", reason or "cancelled")
                if reason:
                    self._speak_contextual("screenshot_blocked")
                return
            skin = self.current_skin
            quality = int(settings.get("jpeg_quality", 70))
            threading.Thread(
                target=self._capture_worker_for(
                    image, manual=True, skin=skin, quality=quality,
                    blocklist=settings.get("blocklist", []),
                    ocr_config=ai_companion.ocr_settings(),
                    screen_rect=None),
                name="portal-capture", daemon=True).start()

        ok, why = wayland_capture.PortalScreenshot(on_result).request()
        if not ok:
            self._screenshot_inflight = False
            return False, why

    def _look_locally(self, app, settings):
        """A Look At This with no model attached.

        Same privacy gates (blocklist, fullscreen, Wayland guarantee), the grab
        happens, and the OCR pass runs locally to pick a canned line keyed by
        what the screen holds -- code, a wall of text, numbers, or nothing.
        The image is then dropped: at no point in this flow is anything sent
        anywhere, which is the difference between "describe" and "upload".
        """
        self._screenshot_source = "local"
        self._hide_for_capture()
        try:
            image, reason = screen_capture.prepare_capture(
                app, settings.get("blocklist", []), self.screen_rect)
        finally:
            self._show_after_capture()
        if image is None:
            ai_companion.note_capture_decision("refused", reason or "no capture")
            if reason:
                self._speak_contextual("screenshot_blocked")
            return
        # OCR on this thread too: one local pass, a fraction of a second, and
        # it is what decides the category -- no image data is encoded at all.
        topic = screen_capture.capture_topics(image, ai_companion.ocr_settings())
        ai_companion.note_capture_decision("described locally", topic)
        self._speak_contextual("local_look_%s" % topic)

    def _report_discarded_capture(self):
        """Tell the user a capture was dropped, from the GUI thread.

        Queued rather than called directly: the worker is not allowed to touch
        the widget.
        """
        try:
            self.ai_bridge.notice.emit("discarded")
        except Exception:
            pass

    @guarded
    def _on_notice(self, kind):
        """Side-channel notice from a worker (no model involved)."""
        if kind == "discarded":
            self.sounds.play("no")
            self._speak_contextual("screenshot_blocked")

    # =================================================================
    # Context checks
    # =================================================================

    def _poll_focus(self):
        """Track the focused window for dialogue and for 'is anyone there'.

        Only a genuine change counts as activity. Marking the pet active on
        every poll meant the recent-activity gate was always satisfied, so
        screenshots fired on an unattended machine.
        """
        if self.tracker is None:
            return
        try:
            active = self.tracker.get_active_window()
        except Exception:
            return
        if not active or not active.title:
            return
        if active.title != self.last_window_title:
            self.last_window_title = active.title
            self.last_window_proc = active.process or ""
            self.last_activity_tick = self.tick_count

    def _check_window_context(self):
        """Occasionally say something about what the user is doing.

        With the model available the pet gets a real line about the window it
        is looking at; the canned table covers everything else (no key, rate
        limited, quiet hours). Both paths share the idle rate-limit pool, so
        the cost is bounded either way.
        """
        if self.speech_timer > 0 or self.awaiting_ai:
            return
        if self.state in ("SLEEP", "DRAG", "FALL", "PERCH"):
            return
        if companion_dialogue is None:
            return
        # Mid-focus the pet keeps its mouth shut: that is the whole point of a
        # focus block. Quiet hours do the same for unsolicited lines.
        if self.in_focus() or self._quiet_now():
            return

        # Mood first: a drowsy pet does not have opinions about your window.
        mood_event = self._mood_event()
        if mood_event:
            self._speak_contextual(mood_event)
            return

        if self._ai_context_lines_enabled() and random.random() < AI_CONTEXT_CHANCE \
                and self._request_context_line():
            return

        if self.last_window_title and random.random() < 0.7:
            event = "window"
        else:
            event = "time"
        self.speak(companion_dialogue.get_contextual_quote(
            self.current_skin, event_type=event,
            window_title=self.last_window_title or "",
            window_proc=self.last_window_proc or ""))

    def _ai_context_lines_enabled(self):
        """Whether idle chatter may use the model (behaviour.ai_context_lines).

        Off means the pet still talks about your windows, just from the canned
        table: no tokens, no network, no surprises on a metered connection.
        """
        return bool(self._behaviour().get("ai_context_lines", True))

    def _request_context_line(self):
        """Ask the model about the focused window. True if the request started.

        The prompt carries only what the window tracker already sees -- a
        title and a process name, never pixels -- so this is not a screenshot
        by another name.
        """
        if not ai_companion.ai_enabled():
            return False
        if not self.last_window_title:
            return False
        context = "window title: %r (process %r)" % (
            self.last_window_title[:80], (self.last_window_proc or "")[:40])
        started = ai_companion.maybe_context_line(
            self.current_skin, context, self.ai_bridge.line_ready.emit)
        if started:
            self.awaiting_ai = True
            self.ai_deadline = self.tick_count + 90
            self.ai_request_kind = "idle"
        return started

    # =================================================================
    # Mood and energy
    # =================================================================

    def mood(self):
        """'sleepy', 'happy' or 'content'.

        Cheap to compute every tick: it is just energy, whether the pet was
        petted recently, and whether the user is in a focus block.
        """
        if self.state == "SLEEP" or self.energy < MOOD_SLEEPY_MAX:
            return "sleepy"
        recently_petted = (self.petted_until_tick is not None
                           and self.tick_count - self.petted_until_tick < 600)
        if self.energy >= MOOD_HAPPY_MIN and recently_petted:
            return "happy"
        return "content"

    def _drain_energy(self):
        """Account for this tick's exertion, and rest when doing nothing."""
        drain = ENERGY_DRAIN.get(self.state)
        if drain is not None:
            self.energy = max(0.0, self.energy - drain)
        elif self.state == "SLEEP":
            self.energy = min(100.0, self.energy + ENERGY_GAIN_SLEEP)
        elif self.state in CALM_STATES:
            self.energy = min(100.0, self.energy + ENERGY_GAIN_IDLE)

    def _mood_event(self):
        """The dialogue event matching the current mood, if any."""
        mood = self.mood()
        if mood == "sleepy":
            return "sleepy"
        if mood == "happy" and random.random() < 0.35:
            return "happy"
        return None

    # =================================================================
    # Focus sessions
    # =================================================================

    def _poll_focus_session(self):
        """Watch the focus timer and react when the mode changes."""
        if not self.focus.configured():
            return
        previous = self.focus.mode
        self.focus.poll()
        change = self.focus.transition(previous)
        if not change:
            return
        # Only physical states suppress this. A session ending is exactly the
        # sort of thing worth interrupting for -- the old code dropped the line
        # whenever the pet happened to still be talking, which was most of the
        # time.
        if self.state in ("DRAG", "FALL"):
            return
        if change == "started":
            self._speak_contextual("focus_start")
        elif change == "ended":
            self.energy = min(100.0, self.energy + 10.0)
            self._speak_contextual("focus_end")
        elif change == "break":
            self.energy = min(100.0, self.energy + 15.0)
            self.petted_until_tick = self.tick_count
            self._speak_contextual("happy")

    def in_focus(self):
        try:
            return self.focus.is_focus()
        except Exception:
            return False

    # =================================================================
    # Ledges
    # =================================================================

    def _is_on_ledge(self):
        return self.current_ledge is not None and not self.current_ledge.is_floor

    def _ledge_under_pet(self):
        """The ledge the pet is standing on, or None.

        Uses overlap rather than centre containment, matching
        `find_landing_y`. The two used to disagree, and the disagreement is
        what let the pet enter PERCH holding no ledge at all -- an
        AttributeError on the next tick, and PyQt5 turns exceptions in slots
        into abort().
        """
        if self.tracker is None:
            return None
        try:
            return self.tracker.find_ledge_at(
                self.y_pos + self.sprite_h, self.x_pos, self.sprite_w,
                self.screen_rect)
        except Exception:
            return None

    def _stand_y(self, ledge):
        """Y for the pet's top-left when sitting on a ledge.

        A titlebar at the very top of the screen would otherwise put the pet
        half off it, so it is seated no higher than the screen allows.
        """
        return max(float(ledge.y) - self.sprite_h, float(self.screen_top))

    def _ledge_at_landing(self, landing_y):
        """Which titlebar a fall landed on, matched on the landing height.

        Matched on height rather than by the pet's feet, because a pet seated
        at the top of the screen has its feet *below* the titlebar it is
        standing on.
        """
        for ledge in self._titlebar_ledges():
            if abs((float(ledge.y) - self.sprite_h) - landing_y) < 2.0:
                return ledge
        return None

    def _ledge_exists(self, ledge):
        """Whether a titlebar is still on the desktop."""
        if ledge is None:
            return False
        for candidate in self._titlebar_ledges():
            if (candidate.y == ledge.y and candidate.x_min == ledge.x_min
                    and candidate.x_max == ledge.x_max):
                return True
        return False

    def _validate_current_ledge(self):
        """Whether the ledge the pet is perched on still exists.

        Identity, not position: a pet seated at the very top of the screen has
        its feet below the titlebar it is standing on, so "is the pet over the
        ledge" is the wrong question to ask.
        """
        if self.current_ledge is None or self.current_ledge.is_floor:
            return True
        if self.tracker is None:
            return False
        if not self._ledge_exists(self.current_ledge):
            return False
        # Re-seat: a window that moved, or the screen that changed, drags the
        # pet along instead of leaving it in mid-air.
        self.y_pos = self._stand_y(self.current_ledge)
        return True

    def _clamp_to_ledge(self):
        """Keep the pet inside its ledge's horizontal bounds.

        Called by every state that moves horizontally. STALK and RUN used to
        move without this, so the pet walked off a titlebar and then hung in
        mid-air, because nothing ever corrected its y again.
        """
        if not self._is_on_ledge():
            return False
        ledge = self.current_ledge
        if self.x_pos < ledge.x_min:
            self.x_pos = float(ledge.x_min)
            return True
        if self.x_pos + self.sprite_w > ledge.x_max:
            self.x_pos = float(ledge.x_max - self.sprite_w)
            return True
        return False

    # =================================================================
    # State transitions
    # =================================================================

    def _enter_state(self, new_state):
        if os.environ.get("PET_TRACE") and new_state != self.state:
            import traceback
            sys.stderr.write("STATE %s -> %s from:\n%s\n" % (self.state, new_state, "".join(traceback.format_stack())))
        self.state = new_state
        self.state_timer = 0

    def _enter_fall(self):
        if os.environ.get("PET_TRACE"):
            import traceback
            sys.stderr.write("ENTER_FALL from:\n" + "".join(traceback.format_stack()) + "\n")
        self._enter_state("FALL")
        self.current_ledge = None
        self.goal_ledge = None
        self.vx = 0.0

    def _enter_sit(self):
        self._enter_state("IDLE" if self.is_ghost else "SIT")
        if self.flies:
            # Keep the hover target where the pet already is, so a finished
            # flight doesn't snap to the floor and then ease back up.
            self.ghost_hover_y = self.y_pos
        elif not self._is_on_ledge():
            self.y_pos = self.floor_y

    # ---- one-shot reactions -------------------------------------------

    def _start_one_shot(self, state):
        """Play a reaction (HAPPY, EAT, DIZZY...) in place, then settle."""
        if self.state in ("DRAG", "FALL", "CLIMB", "LEAP", "JUMP", "POUNCE"):
            return False
        self._enter_state(state)
        return True

    def _tick_one_shot(self):
        if self.state_timer >= ONE_SHOT_TICKS.get(self.state, 40):
            finished = self.state
            self._finish_one_shot()
            if finished == "EAT":
                # a happy little hop once the snack is gone
                self._start_one_shot("HAPPY")

    def _finish_one_shot(self):
        if self._is_on_ledge():
            self._enter_state("PERCH")
        else:
            self._enter_sit()

    # =================================================================
    # Main loop
    # =================================================================

    @guarded
    def game_loop(self):
        self.tick_count += 1

        # Cursor movement counts as activity.
        cursor = QCursor.pos()
        cursor_step = (cursor - self.last_cursor).manhattanLength()
        if cursor_step > CURSOR_ACTIVITY_PX:
            self.last_activity_tick = self.tick_count
        self.last_cursor = cursor

        if self.tick_count % CONFIG_RELOAD_TICKS == 0:
            self._apply_config_changes()

        if self.tick_count % FOCUS_POLL_TICKS == 0:
            self._poll_focus()        # which window is focused
            self._poll_focus_session()  # whether the user is in a focus block

        if self.tick_count % FULLSCREEN_CHECK_TICKS == 0:
            self._check_fullscreen()

        # Re-assert the top of the stack every couple of seconds: a window
        # that maps later (or another overlay) can land above the pet.
        if self.tick_count % RAISE_TICKS == 0 and self.isVisible() \
                and not self.dragging and QApplication.activePopupWidget() is None:
            self.raise_()

        if self.tick_count % CONTEXT_CHECK_INTERVAL == 0:
            self._check_window_context()

        # Model took too long: fall back to a canned line so the pet never
        # looks stuck, and let a late reply through only if nothing newer is
        # showing.
        if self.awaiting_ai and self.tick_count > self.ai_deadline:
            self.awaiting_ai = False
            self._ai_timed_out = True
            if self.speech_timer == 0:
                self.speak(get_system_status_quote(self.current_skin))

        if self.speech_timer > 0:
            self.speech_timer -= 1
            if self.speech_timer == 0:
                self._clear_speech()

        if self.speech_text and self.speech_reveal < len(self.speech_text):
            self.speech_reveal += SPEECH_REVEAL_PER_TICK

        # a heart that was just asked for floats up from the pet
        if self.heart_timer > self._heart_seen:
            self._emit_fx("hearts")
        self._heart_seen = self.heart_timer
        if self.heart_timer > 0:
            self.heart_timer -= 1
            self._heart_seen = self.heart_timer

        if self._squash_ticks > 0:
            self._squash_ticks -= 1

        if not self.dragging and not self._hidden_for_fullscreen:
            self._update_fsm(cursor)
            self._ambient_flourish(cursor, cursor_step)
            self.maybe_screenshot()

        if self.tick_count % SAVE_INTERVAL_TICKS == 0:
            self._check_milestone()
            self._save_state()

        self._sync_widget_geometry()
        self.update()
        self._tick_fx()

    # =================================================================
    # Particles and flourishes
    # =================================================================

    def _emit_fx(self, name):
        if self.fx is not None:
            try:
                self.fx.burst(name)
            except Exception:
                log_exception("fx")

    def _tick_fx(self):
        if self.fx is None:
            return
        try:
            self.fx.tick(max(1, int(self.scale_factor)), self.x(),
                         self.y() + self._sprite_offset(),
                         self.isVisible() and not self._hidden_for_fullscreen,
                         self.facing)
        except Exception:
            log_exception("fx tick")

    def _check_welcome_back(self):
        """The user comes back after a while away: the pet notices.

        A shiba loses its mind, a robot reports in, a cat... looks up.
        """
        if self.tick_count % 20:
            return
        idle = self._user_idle_seconds()
        if idle is None:
            return
        if idle >= WELCOME_AWAY_SECONDS:
            self._was_away = True
            return
        if not self._was_away or idle > 5:
            return
        self._was_away = False
        if self.state in ("DRAG", "FALL", "LEAP", "JUMP", "CLIMB", "SLEEP"):
            return
        skin = self.current_skin
        self._face_cursor(self._cursor.x() - (self.x_pos + self.sprite_w / 2.0))
        if skin == "shiba":
            self._start_one_shot("HAPPY")
            self.sounds.play("bark", ambient=True)
        elif skin in ("jarvis", "ultron"):
            self._start_one_shot("ALERT")
            self.sounds.play(self.sounds.voice_cue(skin), ambient=True)
        elif skin == "ghost":
            self._start_one_shot("PAT")
        else:
            self._enter_state("IDLE")       # a cat merely acknowledges you
        if not self._quiet_now() and random.random() < 0.6:
            self._speak_contextual("greeting")

    def _ambient_flourish(self, cursor, cursor_step):
        """Small things the pet does on its own: Zzz, dust, sweat, notes,
        and noticing a cursor that whips past it."""
        t = self.tick_count
        state = self.state
        if state == "SLEEP" and self.state_timer > YAWN_TICKS \
                and t % 34 == 0:
            self._emit_fx("zzz")
        elif state in ("RUN", "CHASE", "ZOOMIES") and t % 5 == 0 \
                and not self.flies:
            self._emit_fx("kick")
            if self.energy < MOOD_SLEEPY_MAX + 10 and t % 40 == 0:
                self._emit_fx("sweat")
        elif state == "DANCE" and t % 12 == 0:
            self._emit_fx("notes")
        elif state == "EAT" and self.state_timer % 14 == 7:
            self._emit_fx("sparkles" if self.is_ghost else
                          "sparks" if self.flies or self.current_skin == "ultron"
                          else "crumbs")
        elif state == "DIZZY" and self.state_timer == 1:
            self._emit_fx("stars")
        elif state == "SULK" and self.state_timer == 2:
            self._emit_fx("anger")
        elif state == "WALK" and self.profile.get("heavy") and t % 12 == 0:
            self._emit_fx("kick")           # every step lands with a thud
        elif state == "SNIFF" and self.state_timer % 44 == 30:
            self._emit_fx("question" if random.random() < 0.15 else "sniff")
        elif state == "FLY" and self.current_skin == "jarvis" and t % 6 == 0:
            self._emit_fx("spark")

        self._check_welcome_back()

        # a ghost gets spooky when the cursor creeps up on it
        if self.is_ghost and state in ("IDLE", "SIT") and \
                t >= self._alert_ready_tick:
            cx = self.x_pos + self.sprite_w / 2.0
            cy = self.y_pos + self.sprite_h / 2.0
            if math.hypot(cursor.x() - cx, cursor.y() - cy) < 45:
                self._alert_ready_tick = t + ALERT_COOLDOWN_TICKS
                self._start_one_shot("SPOOK")
                self.sounds.play("boo", ambient=True)
                return

        if self._laser_until and t >= self._laser_until:
            self._laser_until = 0
            if state == "CHASE":
                self._enter_sit()
            self._speak_contextual("laser_end")

        # a cursor flick close by: ears up, "!"
        if state in ("SIT", "IDLE", "PERCH") and cursor_step > ALERT_SPEED_PX \
                and t >= self._alert_ready_tick and not self._laser_until:
            cx = self.x_pos + self.sprite_w / 2.0
            cy = self.y_pos + self.sprite_h / 2.0
            if math.hypot(cursor.x() - cx, cursor.y() - cy) < ALERT_RADIUS_PX:
                self._alert_ready_tick = t + ALERT_COOLDOWN_TICKS
                self.facing = 1 if cursor.x() >= cx else -1
                self._start_one_shot("ALERT")
                self._emit_fx("exclaim")
                self.sounds.play("alert", ambient=True)
                if random.random() < 0.25 and not self._quiet_now():
                    self._speak_contextual("alert")

    # =================================================================
    # Environment awareness
    # =================================================================

    def _apply_config_changes(self):
        """Pick up edits to companion_config.json without a restart."""
        try:
            if not ai_companion.reload_config_if_changed():
                return
        except Exception:
            return
        if self.tracker is not None:
            try:
                self.tracker.set_ignored_windows(
                    self._behaviour().get("ignore_windows", []))
            except Exception:
                log_exception("config: ignore list")
        try:
            self.sounds.update(ai_companion.CONFIG.get("audio") or {})
        except Exception:
            log_exception("config: audio")
        self._apply_click_through()
        self._refresh_screen_geometry()

    def _check_fullscreen(self):
        """Get out of the way of a fullscreen window.

        Cutscenes, games, presentations and video full-screen windows are
        exactly where a floating pet is least welcome -- and where it used to
        sit on top of the titlebar ledge it cannot even see.
        """
        if self.tracker is None:
            return
        wanted = bool(self._behaviour().get("hide_on_fullscreen", True))
        try:
            fullscreen = bool(self.tracker.fullscreen_active()) and wanted
        except Exception:
            return
        if fullscreen == self._hidden_for_fullscreen:
            return
        self._hidden_for_fullscreen = fullscreen
        if fullscreen:
            self.hide()
        else:
            self._restack_on_top(force=True)


    # =================================================================
    # Finite state machine
    # =================================================================

    def _update_fsm(self, cursor):
        self.state_timer += 1
        self._drain_energy()
        try:
            self._cursor = QPoint(int(cursor.x()), int(cursor.y()))
        except Exception:
            pass

        center_x = self.x_pos + self.sprite_w / 2.0
        center_y = self.y_pos + self.sprite_h / 2.0
        cursor_dx = cursor.x() - center_x
        cursor_dy = cursor.y() - center_y
        cursor_dist = math.hypot(cursor_dx, cursor_dy)

        if self.state == "DRAG":
            return
        if self.state == "FALL":
            self._tick_fall()
            return
        if self.state == "CLIMB":
            self._tick_climb()
            return
        if self.state == "PERCH":
            self._tick_perch(cursor_dx, cursor_dist)
            return
        if self.state == "LEAP":
            self._tick_leap()
            return
        if self.state in ("JUMP", "POUNCE"):
            # Airborne: no ground-lock, no x clamp against the floor, the arc
            # handler owns the position until it lands.
            self._tick_jump()
            return

        if self.flies:
            bob = 5.0 if self.is_ghost else 2.0
            hover = self.ghost_hover_y + math.sin(self.tick_count / 25.0) * bob
            self.y_pos += (hover - self.y_pos) * 0.15
        elif not self._is_on_ledge():
            self.y_pos = self.floor_y
        elif not self._validate_current_ledge():
            # The titlebar the pet is standing on closed (or was renamed)
            # while it idled, walked, groomed or slept on it. This validation
            # only ran from PERCH, so every other state kept the pet floating
            # at the vanished window's height until something else moved it.
            self._leave_ledge()
            return

        # Horizontal bounds of the current screen, honouring its origin.
        low = float(self.screen_left)
        high = float(self.screen_left + self.screen_w - self.sprite_w)
        self.x_pos = max(low, min(high, self.x_pos))

        # Nobody has touched the machine in a while: settle down and nap until
        # they come back. Cheap (one cached number) and it stops the pet from
        # walking around an empty screen all night.
        if self._should_nap_while_idle():
            self._enter_state("SLEEP")
            return

        # The laser pointer game: anything calm turns into a chase (flyers
        # glide after it instead).
        if self._laser_until and self.state in (
                "SIT", "IDLE", "BLINK", "WALK", "GROOM", "STALK", "ALERT",
                "FLY", "SNIFF", "FOLLOW"):
            if abs(cursor_dx) > STEER_DEADZONE:
                self.facing = 1 if cursor_dx > 0 else -1
            # only give chase when there is somewhere to run to; a dot
            # straight overhead gets watched, not run at on the spot
            if self.flies and cursor_dist > 30 or \
                    abs(cursor_dx) > STEER_DEADZONE * 3:
                self._enter_state("HAUNT" if self.flies else "CHASE")
                return

        if self.state in ONE_SHOT_TICKS:
            self._tick_one_shot()
        elif self.state == "FLY":
            self._tick_fly()
        elif self.state == "HAUNT":
            self._tick_haunt(cursor_dx, cursor_dist)
        elif self.state == "PHASE":
            self._tick_phase()
        elif self.state == "ZOOMIES":
            self._tick_zoomies()
        elif self.state == "SNIFF":
            self._tick_sniff()
        elif self.state == "SPIN":
            if self.state_timer % 5 == 0:
                self.facing = -self.facing
            if self.state_timer >= SPIN_TICKS:
                self._enter_state("SLEEP")
                self.sounds.play("yawn", ambient=True)
        elif self.state == "FOLLOW":
            self._tick_follow(cursor_dx, cursor_dist)
        elif self.state in ("SIT", "IDLE"):
            self._tick_idle(cursor_dx, cursor_dist)
        elif self.state == "BLINK":
            if self.state_timer > STATE_MIN_DURATION.get("BLINK", 6):
                self._enter_sit()
        elif self.state == "GROOM":
            if self.state_timer > STATE_MAX_DURATION.get("GROOM", 100):
                self._enter_sit()
        elif self.state == "CHASE":
            self._tick_chase(cursor_dx, cursor_dist)
        elif self.state == "STALK":
            self._tick_stalk(cursor_dx, cursor_dist)
        elif self.state == "WALK":
            self._tick_walk()
        elif self.state == "RUN":
            self._tick_run(cursor_dx, cursor_dist)
        elif self.state == "SWAT":
            if self.state_timer > STATE_MIN_DURATION.get("SWAT", 20):
                self.energy = min(100.0, self.energy + 8)
                self._enter_sit()
        elif self.state == "SLEEP":
            still_away = self._user_idle_seconds()
            threshold = self._idle_sleep_seconds()
            still_away = (threshold > 0 and still_away is not None
                          and still_away >= threshold)
            if not still_away and (self.state_timer > STATE_MAX_DURATION.get("SLEEP", 300)
                                   or cursor_dist < 30):
                was_nap = self.state_timer > 25   # >1s asleep: a real nap
                self.energy = min(100.0, self.energy + ENERGY_GAIN_WAKE)
                self._enter_sit()
                if was_nap:
                    self._yawn_until = self.tick_count + YAWN_TICKS
                    if not self._quiet_now() and random.random() < 0.3:
                        self._speak_contextual("wake")
                    self.sounds.play("wake", ambient=True)

    # ---- personality states --------------------------------------------

    def _tick_fly(self):
        """Free flight to (_fly_x, _fly_y): glide, ease in, settle."""
        dx = self._fly_x - self.x_pos
        speed = float(self.profile.get("walk_speed", 1.5))
        step = min(speed, abs(dx) * 0.08 + 0.35)
        if abs(dx) > 1.0:
            self.facing = 1 if dx > 0 else -1
            self.x_pos += self.facing * step
        self.ghost_hover_y = self._fly_y
        arrived = abs(dx) <= 1.5 and abs(self.y_pos - self._fly_y) < 6
        if arrived or self.state_timer > FLY_MAX_TICKS:
            self._enter_sit()

    def _tick_haunt(self, cursor_dx, cursor_dist):
        """Drift after the cursor, keeping a spooky distance; boo if caught."""
        cursor = self._cursor
        laser = bool(self._laser_until)
        # Pick a side of the cursor and keep it; re-deciding every tick made
        # the target jump across the cursor and back, so it never arrived.
        if self.state_timer <= 1 or not getattr(self, "_haunt_side", 0) \
                or abs(cursor_dx) > HAUNT_DISTANCE * 3:
            self._haunt_side = -1 if cursor_dx > 0 else 1
        side = self._haunt_side
        gap = 20 if laser else HAUNT_DISTANCE
        target_x = cursor.x() + side * gap - self.sprite_w / 2.0
        top, bottom = self._fly_band()
        self.ghost_hover_y = max(top, min(bottom, cursor.y() - self.sprite_h / 2.0))
        dx = target_x - self.x_pos
        speed = float(self.profile.get("walk_speed", 1.4)) * (2.0 if laser else 0.9)
        if abs(dx) > 2:
            self.x_pos += (1 if dx > 0 else -1) * min(speed, abs(dx))
        self.facing = -side          # looking at the cursor from its side
        if not laser and cursor_dist < 50 and self.is_ghost:
            self._start_one_shot("SPOOK")
            self._emit_fx("exclaim")
            self.sounds.play("boo", ambient=True)
            return
        if not laser and self.state_timer > HAUNT_TICKS:
            self._enter_sit()

    def _tick_phase(self):
        """Fade out in a puff, reappear somewhere else in another."""
        t = self.state_timer
        if t == 1:
            self._emit_fx("puff")
        elif t == PHASE_TICKS // 2:
            top, bottom = self._fly_band()
            low = float(self.screen_left)
            high = float(self.screen_left + self.screen_w - self.sprite_w)
            self.x_pos = random.uniform(low, high)
            self.ghost_hover_y = self.y_pos = random.uniform(top, bottom)
            self._emit_fx("puff")
        elif t >= PHASE_TICKS:
            self._enter_sit()

    def phased_out(self):
        return self.state == "PHASE" and 2 <= self.state_timer < PHASE_TICKS // 2 + 2

    def _tick_zoomies(self):
        """Flat-out sprint, edge to edge, a few laps."""
        self.x_pos += self.facing * ZOOMIES_SPEED
        low = float(self.screen_left)
        high = float(self.screen_left + self.screen_w - self.sprite_w)
        hit = self._clamp_to_ledge()
        if self.x_pos <= low or self.x_pos >= high:
            self.x_pos = max(low, min(high, self.x_pos))
            hit = True
        if hit or self.state_timer % 90 == 0:
            self.facing = -self.facing
            self._zoom_laps -= 1
            self._emit_fx("dust")
        if self._zoom_laps <= 0 or self.state_timer > 400:
            self._enter_sit()

    def _tick_sniff(self):
        """Nose to the floor: shuffle, stop, sniff, maybe turn around."""
        phase = self.state_timer % 44
        if phase < 28:
            self.x_pos += self.facing * 0.7
        elif phase == 43 and random.random() < 0.35:
            self.facing = -self.facing
        low = float(self.screen_left)
        high = float(self.screen_left + self.screen_w - self.sprite_w)
        if self._clamp_to_ledge() or not low < self.x_pos < high:
            self.x_pos = max(low, min(high, self.x_pos))
            self.facing = -self.facing
        if self.state_timer > SNIFF_TICKS:
            self._enter_sit()

    def _tick_follow(self, cursor_dx, cursor_dist):
        """Trot over to wherever you are and sit there, looking up."""
        if abs(cursor_dx) < 40:
            self._face_cursor(cursor_dx)
            self._enter_sit()
            if random.random() < 0.3:
                self._start_one_shot("HAPPY")
            return
        self._face_cursor(cursor_dx)
        self.x_pos += self.facing * float(self.profile.get("walk_speed", 1.5)) * 1.2
        low = float(self.screen_left)
        high = float(self.screen_left + self.screen_w - self.sprite_w)
        if self._clamp_to_ledge() or not low <= self.x_pos <= high:
            self.x_pos = max(low, min(high, self.x_pos))
            self._enter_sit()
            return
        if self.state_timer > FOLLOW_TICKS:
            self._enter_sit()

    # ---- being alone ---------------------------------------------------

    def _idle_sleep_seconds(self):
        """How long the user has to be away before the pet naps. 0 = never."""
        try:
            return float(self._behaviour().get("sleep_when_idle_seconds", 300))
        except (TypeError, ValueError):
            return 300.0

    def _user_idle_seconds(self):
        """Seconds since the user last touched anything, or None.

        Prefers the OS's own answer (XScreenSaver / GetLastInputInfo, via the
        tracker) and falls back to the pet's own cursor-activity signal, which
        needs no platform support at all.
        """
        seconds = None
        if self.tracker is not None:
            try:
                seconds = self.tracker.user_idle_seconds()
            except Exception:
                seconds = None
        if seconds is None:
            seconds = (self.tick_count - self.last_activity_tick) / TICKS_PER_SECOND
        return seconds

    def _should_nap_while_idle(self):
        threshold = self._idle_sleep_seconds()
        if threshold <= 0:
            return False
        if self.state == "SLEEP":
            return False
        if self.state not in ("SIT", "IDLE", "BLINK", "GROOM", "WALK", "STALK",
                              "SWAT", "CHASE", "RUN", "SNIFF", "FOLLOW",
                              "FLY", "HAUNT", "ZOOMIES"):
            return False
        idle = self._user_idle_seconds()
        return idle is not None and idle >= threshold

    def _tick_idle(self, cursor_dx, cursor_dist):
        key = self.state
        min_dur = STATE_MIN_DURATION.get(key, 60)
        if self.state_timer < min_dur:
            return
        if (self.state_timer - min_dur) % TRANSITION_EVAL_INTERVAL != 0:
            return
        max_dur = STATE_MAX_DURATION.get(key, 160)
        self._pick_idle_transition(cursor_dx, cursor_dist,
                                  force=self.state_timer >= max_dur)

    def _walk_target(self):
        """A walk destination clamped to the pet's current surface."""
        if self._is_on_ledge():
            ledge = self.current_ledge
            span = ledge.x_max - ledge.x_min - self.sprite_w
            if span <= 20:
                return None
            return random.uniform(ledge.x_min, ledge.x_max - self.sprite_w)
        low = float(self.screen_left + 8)
        high = float(self.screen_left + self.screen_w - self.sprite_w - 8)
        if high <= low:
            return None
        return max(low, min(high, self.x_pos
                            + random.choice([-1, 1]) * random.randint(60, 160)))

    def _pick_idle_transition(self, cursor_dx, cursor_dist, force=False):
        """Decide what to do next, in character.

        The repertoire, weights and habits come from pet_personality: a cat
        and a shiba (and a ghost, and two robots) want different things, and
        none of them does the same thing twice in a row.
        """
        mood = self.mood()
        focusing = self.in_focus()

        # A tired pet sleeps, and a sleeping pet does not get excited.
        if self.energy < 15 or (mood == "sleepy" and random.random() < 0.35):
            if self.profile.get("air") and "charge" in self.profile["air"] \
                    and self.energy < 40 and random.random() < 0.5:
                self._run_action("charge", cursor_dx, cursor_dist)
                return
            self._run_action("nap", cursor_dx, cursor_dist)
            return

        if self.flies:
            context = "air"
        elif self._is_on_ledge():
            context = "ledge"
        else:
            context = "floor"
        tired = mood == "sleepy" or self.energy <= MOOD_SLEEPY_MAX

        queued, self._next_action = self._next_action, None
        tried = set()
        for attempt in range(4):
            if queued and attempt == 0:
                action = queued
            else:
                action = self.chooser.pick(
                    self.current_skin, context, tired=tired,
                    focusing=focusing, energy=self.energy,
                    allowed=lambda a: a not in tried and self._action_allowed(a),
                    can_explore=self._explore_ready())
            if action is None:
                break
            tried.add(action)
            if self._run_action(action, cursor_dx, cursor_dist):
                self.chooser.remember(action)
                return

        if force:
            if context == "air":
                self._run_action("drift", cursor_dx, cursor_dist)
                return
            target = self._walk_target()
            if target is not None:
                self.target_x = target
                self._enter_state("WALK")

    def _action_allowed(self, action):
        flag = persona.NEEDS.get(action)
        return flag is None or bool(getattr(self, flag, True))

    def _explore_ready(self):
        cooldown = self.profile.get("explore_cooldown")
        if cooldown is None:
            return False
        return (self.tick_count - self._last_explore_tick
                >= cooldown * TICKS_PER_SECOND)

    def _face_cursor(self, cursor_dx):
        if abs(cursor_dx) > STEER_DEADZONE:
            self.facing = 1 if cursor_dx > 0 else -1

    def _fly_band(self):
        top_f, bottom_f = FLY_BAND.get(self.current_skin, (0.3, 1.0))
        low = self.floor_y - 6.0
        top = self.screen_top + self.screen_h * top_f
        return max(float(self.screen_top), min(top, low)), low

    def _start_fly(self, x, y):
        low = float(self.screen_left)
        high = float(self.screen_left + self.screen_w - self.sprite_w)
        top, bottom = self._fly_band()
        self._fly_x = max(low, min(high, float(x)))
        self._fly_y = max(top, min(bottom, float(y)))
        self.facing = 1 if self._fly_x >= self.x_pos else -1
        self._enter_state("FLY")

    def _run_action(self, action, cursor_dx, cursor_dist):
        """Start one personality action. True if something started."""
        cursor = self._cursor
        heavy = self.profile.get("heavy")
        if action == "blink":
            self._enter_state("BLINK")
        elif action in ("stare", "loaf"):
            if action == "stare":
                self._face_cursor(cursor_dx)
            on_ledge = self._is_on_ledge()
            self._enter_state("IDLE" if action == "stare" or self.is_ghost
                              else "SIT")
            if not on_ledge and not self.flies:
                self.y_pos = self.floor_y
        elif action in ("groom", "scan"):
            self._enter_state("GROOM")
        elif action in ("wander", "march"):
            target = self._walk_target()
            if target is None:
                return False
            self.target_x = target
            self._enter_state("WALK")
        elif action == "stalk":
            if not 45 < cursor_dist < 420:
                return False
            self._face_cursor(cursor_dx)
            self._enter_state("STALK")
        elif action == "chase":
            if not 45 < cursor_dist < 360:
                return False
            self._enter_state("CHASE")
        elif action in ("hop", "stomp"):
            if self._is_on_ledge():
                return False
            self._heavy_landing = action == "stomp" or bool(heavy)
            self._start_jump()
        elif action == "climb":
            if self._is_on_ledge() or not self._try_start_climb():
                return False
            self._last_explore_tick = self.tick_count
        elif action == "explore":
            if not self._start_exploring():
                return False
            self._last_explore_tick = self.tick_count
        elif action == "go_down":
            if not self._is_on_ledge():
                return False
            if self.allow_jump and self.energy > 40:
                self._start_jump()
            else:
                self._leave_ledge()
        elif action == "nap":
            if self.profile.get("nap_spin") and not self._is_on_ledge():
                self._enter_state("SPIN")      # round and round, then down
            else:
                self._enter_state("SLEEP")
                self.sounds.play("yawn", ambient=True)
        elif action == "zoomies":
            if self._is_on_ledge():
                return False
            self._zoom_laps = random.randint(2, 4)
            self.facing = random.choice((-1, 1))
            self._enter_state("ZOOMIES")
            follow_up = (self.profile.get("after") or {}).get("zoomies")
            self._next_action = follow_up
        elif action == "dance":
            self._start_dance(user=False)
            return self.state == "DANCE"
        elif action == "follow":
            if cursor_dist < 90:
                return False
            self._enter_state("FOLLOW")
        elif action == "sniff":
            self.facing = random.choice((-1, 1))
            self._enter_state("SNIFF")
        elif action == "beg":
            if cursor_dist > 400:
                return False
            self._face_cursor(cursor_dx)
            self._start_one_shot("HAPPY")
        elif action in ("bark", "beep", "surge"):
            self._face_cursor(cursor_dx)
            if not self._start_one_shot("ALERT"):
                return False
            self._emit_fx("sparks" if action == "surge" else "exclaim")
            cue = {"bark": "bark", "beep": "beep", "surge": "menace"}[action]
            self.sounds.play(cue, ambient=True)
        elif action == "glare":
            self._face_cursor(cursor_dx)
            if not self._start_one_shot("SULK"):
                return False
        elif action == "charge":
            if not self._start_one_shot("EAT"):
                return False
            self.energy = min(100.0, self.energy + 30.0)
        elif action == "wave":
            self._face_cursor(cursor_dx)
            self._start_one_shot("PAT")
        elif action == "drift":
            top, bottom = self._fly_band()
            x = self.x_pos + random.choice((-1, 1)) * random.uniform(120, 420)
            y = random.uniform(top, bottom)
            # mostly cruise low; now and then go somewhere high
            if random.random() < 0.55:
                y = random.uniform(bottom - self.screen_h * 0.2, bottom)
            self._start_fly(x, y)
        elif action == "patrol":
            x = self.x_pos + random.choice((-1, 1)) * random.uniform(200, 600)
            self._start_fly(x, self.ghost_hover_y)
        elif action == "assist":
            side = -1 if cursor.x() > self.x_pos + self.sprite_w / 2 else 1
            self._start_fly(cursor.x() + side * 70 - self.sprite_w / 2,
                            cursor.y() - self.sprite_h)
        elif action == "haunt":
            if cursor_dist > 700:
                return False
            self._enter_state("HAUNT")
        elif action == "phase":
            self._enter_state("PHASE")
        else:
            return False
        return True

    # ---- LEAP ----------------------------------------------------------
    #
    # A hop straight up clears about 35px, which is not enough to reach any
    # window titlebar, so the pet launches itself ballistically at whatever it
    # decided it wanted. Without this the pet lives on the taskbar forever: the
    # only way up was CLIMB, which needs a window border within 35px, and that
    # almost never happens in the middle of a screen.

    @staticmethod
    def _goal_anchor(ledge, sprite_w, here):
        """Where on a ledge the pet aims: the point nearest to it.

        Anchoring on the middle of a titlebar reads as "go to the centre of the
        screen", which for a maximised window means crossing the whole desktop
        to reach a window that is directly overhead.
        """
        low = ledge.x_min + sprite_w / 2.0
        high = ledge.x_max - sprite_w / 2.0
        if high < low:
            return (ledge.x_min + ledge.x_max) / 2.0
        return max(low, min(high, here))

    @staticmethod
    def _leap_drag_adjusted(flight):
        """Distance one tick of initial vx covers over a whole flight.

        Air bleed means the naive `offset / flight` undershoots badly over a
        long arc, so the launch velocity has to be scaled up to match.
        """
        ticks = max(1, int(round(flight)))
        travelled = (1.0 - LEAP_AIR_DRAG ** ticks) / (1.0 - LEAP_AIR_DRAG)
        return max(1.0, travelled)

    def _leap_solution(self, ledge, x=None, y=None, anchor=None):
        """(vy0, vx, flight) to land on `ledge` from (x, y), or None.

        Solves the parabola for the exact height difference, then scales the
        horizontal component to compensate for air bleed so the pet arrives where
        it aimed instead of falling short. It aims at the nearest point on the
        ledge, not the middle of it: for a maximised window the middle is most
        of the screen away, which puts the leap over its speed limit and leaves
        the pet stuck on the taskbar.
        """
        if ledge is None or ledge.is_floor:
            return None
        x = self.x_pos if x is None else x
        y = self.y_pos if y is None else y
        here = x + self.sprite_w / 2.0
        if anchor is None:
            anchor = self._goal_anchor(ledge, self.sprite_w, here)

        target_top = float(ledge.y) - self.sprite_h
        rise = y - target_top
        if rise <= 2.0:                       # not going up: walk instead
            return None
        if rise > LEAP_MAX_RISE:
            return None

        offset = anchor - here
        if abs(offset) > LEAP_MAX_REACH:
            return None

        # Aim the apex LEAP_ARC times the height above the target, so the pet
        # rises past the titlebar and comes down onto it. A symmetric arc whose
        # apex is exactly the target only grazes it, with zero vertical speed,
        # and carries on upward.
        gravity = LEAP_GRAVITY
        apex_rise = rise * LEAP_ARC
        vy0 = math.sqrt(2.0 * gravity * apex_rise)
        # Time from launch to crossing the target height on the way down.
        flight = (vy0 + math.sqrt(max(0.0, 2.0 * gravity * rise
                                      * (LEAP_ARC - 1.0)))) / gravity
        if flight <= 0 or flight > LEAP_MAX_TICKS:
            return None

        vx = offset / self._leap_drag_adjusted(flight)
        if abs(vx) > LEAP_VX_MAX:
            return None
        return -vy0, vx, flight

    def _leap_reachable(self, ledge):
        return self._leap_solution(ledge) is not None

    def _start_leap(self, ledge, anchor=None):
        """Launch at a titlebar. Returns True if the leap started."""
        solution = self._leap_solution(ledge, anchor=anchor)
        if solution is None:
            return False
        self.vy, self.vx, _flight = solution

        # The arc is evaluated analytically each tick rather than integrated.
        # Stepwise integration overshoots the apex by tens of pixels, which made
        # the pet cross its target height early and land short -- most visibly
        # when leaping sideways, where "short" means missing the titlebar.
        self.leap_y0 = self.y_pos
        self.leap_x0 = self.x_pos
        self.leap_vy0 = self.vy
        self.leap_vx0 = self.vx
        self.leap_t = 0.0

        self.leap_target = ledge
        self.facing = 1 if self.vx >= 0 else -1
        self.current_ledge = None
        self.goal_ledge = None
        self.land_event = "perch"
        self._enter_state("LEAP")
        self._speak_contextual("climb")
        return True

    def _tick_leap(self):
        """Advance the arc and decide whether it connected."""
        self.leap_t += 1.0
        t = self.leap_t
        gravity = LEAP_GRAVITY

        self.y_pos = self.leap_y0 + self.leap_vy0 * t + 0.5 * gravity * t * t
        decay = (1.0 - LEAP_AIR_DRAG ** t) / (1.0 - LEAP_AIR_DRAG)
        self.x_pos = self.leap_x0 + self.leap_vx0 * decay
        self.vy = self.leap_vy0 + gravity * t
        self.vx = 0.0

        target = self.leap_target
        descending = self.vy > 0

        if target is not None and descending:
            # The window can close mid-leap. Landing on the stale ledge would
            # put the pet on a titlebar that no longer exists -- the same class
            # of bug as entering PERCH without a ledge at all.
            if not self._ledge_exists(target):
                self.leap_target = None
                self.land_event = None
                self._enter_fall()
                return

            # Crossing the surface on the way down, while horizontally over it.
            # Deliberately not "lands exactly on it": at 30+px per frame the
            # pet steps straight past the surface, and a test that required an
            # exact hit meant it never landed at all.
            if self.y_pos + self.sprite_h >= target.y and \
                    target.holds_any(self.x_pos, self.x_pos + self.sprite_w):
                self.y_pos = self._stand_y(target)
                self.vy = self.vx = 0.0
                self.current_ledge = target
                self.leap_target = None
                self.goal_ledge = None
                self._enter_state("PERCH")
                self._remember_ledge(target)
                self._speak_contextual("perch")
                return

        # Missed, or gave up: fall out of the arc and let the normal landing
        # logic take over.
        if (self.y_pos >= self.floor_y
                or self.state_timer > LEAP_MAX_TICKS
                or (descending and target is not None
                    and self.y_pos + self.sprite_h > target.y)):
            self.leap_target = None
            self.land_event = None
            self._enter_fall()

    # ---- Goals ---------------------------------------------------------
    #
    # The pet needs a reason to leave the floor. Every so often it picks a
    # window it wants to be sitting on, walks underneath it, and leaps.

    def _titlebar_ledges(self):
        """Non-floor ledges, i.e. window titlebars worth visiting."""
        if self.tracker is None:
            return []
        try:
            return [l for l in self.tracker.get_walkable_ledges(self.screen_rect)
                    if not l.is_floor and l.width > 20]
        except Exception:
            return []

    def _remember_ledge(self, ledge):
        if ledge is None:
            return
        self.recent_ledges = [y for y in self.recent_ledges if y != ledge.y]
        self.recent_ledges.append(ledge.y)
        del self.recent_ledges[:-4]

    def _pick_goal_ledge(self):
        """Choose a titlebar to head for. Returns (ledge, anchor_x).

        A ledge is only a candidate if a leap onto it is solvable from
        somewhere along the way -- either from here, or after a short walk
        underneath it. Anything else is not worth committing to.
        """
        candidates = []
        here = self.x_pos + self.sprite_w / 2.0
        for ledge in self._titlebar_ledges():
            if self.current_ledge is not None and \
                    abs(ledge.y - self.current_ledge.y) < 5 and \
                    ledge.x_min <= here <= ledge.x_max:
                continue  # already standing on this one

            anchor = self._goal_anchor(ledge, self.sprite_w, here)
            travel = abs(anchor - here)
            if travel > LEAP_GOAL_WALK:
                continue
            under = anchor - self.sprite_w / 2.0
            from_here = self._leap_solution(ledge, anchor=anchor) is not None
            from_under = (self._leap_solution(ledge, x=under, anchor=anchor)
                          is not None)
            if not (from_here or from_under):
                continue
            candidates.append((0 if from_here else 1, travel, ledge, anchor))

        if not candidates:
            return None, None
        # Prefer a leap we can take immediately, then the nearest window, so it
        # commits to something plausible rather than crossing the desktop.
        candidates.sort(key=lambda item: (item[0], item[1]))
        # Keep at least two candidates: with a single one the staleness filter
        # below has nothing to choose between, and the pet always went to the
        # same window no matter which ones it had just visited.
        pool = candidates[:max(2, len(candidates) // 2)]

        # Bias away from titlebars it was just on. Without this it settles into
        # a two-window loop: up to one, off to the floor, straight back up.
        def staleness(item):
            ledge_y = item[2].y
            for index, seen in enumerate(reversed(self.recent_ledges)):
                if seen == ledge_y:
                    return index
            return len(self.recent_ledges) + 1

        fresh = [item for item in pool if staleness(item) >= 2]
        if fresh:
            pool = fresh
        _immediate, _travel, ledge, anchor = random.choice(pool)
        return ledge, anchor

    def _start_exploring(self):
        """Head for a window: leap if it is already in range, else walk under
        it and leap from there. Returns True if the pet committed to a goal."""
        ledge, anchor = self._pick_goal_ledge()
        if ledge is None:
            return False
        self.goal_ledge = ledge
        if self._start_leap(ledge, anchor=anchor):
            return True
        # Walk to a spot under the ledge; _tick_walk launches the leap.
        self.target_x = max(float(self.screen_left),
                            min(anchor - self.sprite_w / 2.0,
                                float(self.screen_left + self.screen_w
                                      - self.sprite_w)))
        self._enter_state("WALK")
        return True

    def _maybe_launch_at_goal(self):
        """Called from WALK: leap once we are standing under the goal."""
        ledge = self.goal_ledge
        if ledge is None:
            return
        if self._is_on_ledge():
            self.goal_ledge = None
            return
        if not ledge.holds_any(self.x_pos, self.x_pos + self.sprite_w):
            return
        anchor = self._goal_anchor(ledge, self.sprite_w,
                                   self.x_pos + self.sprite_w / 2.0)
        self._start_leap(ledge, anchor=anchor)

    # ---- JUMP ----------------------------------------------------------

    def _start_jump(self, toward=None):
        """Leap upwards, optionally toward a target x."""
        self._enter_state("JUMP")
        self.current_ledge = None
        if toward is not None:
            self.facing = 1 if toward >= self.x_pos else -1
        self.vy = -JUMP_VELOCITY
        if self.flies:
            # A flyer has no legs to push off with; it just rises and settles.
            self.ghost_hover_y = self.y_pos - 40.0

    def _tick_jump(self):
        """JUMP is an arc: rise, then fall back under gravity."""
        self.vy = min(self.vy + GRAVITY, TERMINAL_VEL)
        self.y_pos += self.vy
        if self.vy < 0:
            return  # still going up
        # A hop is not a drop, so it must not announce itself as one.
        self._tick_fall(announce=False)

    def _tick_chase(self, cursor_dx, cursor_dist):
        """Sprint after the cursor, then pounce when close enough."""
        laser = bool(self._laser_until)
        speed = CHASE_SPEED * 1.35 if laser else CHASE_SPEED
        moving = self._steer(cursor_dx, speed)
        if self._clamp_to_ledge():
            self._enter_sit()
            return
        above_head = self._cursor.y() - self.y_pos
        reachable = -POUNCE_REACH_PX < above_head < self.sprite_h
        if cursor_dist < 40 or (not moving and reachable and self.allow_jump):
            # close, or straight overhead within a hop: spring at it
            self._start_jump(toward=self.x_pos + self.facing * 10)
            self._enter_state("POUNCE")
            self.sounds.play("pounce", ambient=not laser)
            return
        if not moving:
            # right under it and it is out of reach: sit and stare up at it
            # instead of running on the spot
            self._enter_state("IDLE")
            return
        if not laser and (cursor_dist > 420 or self.state_timer > 220):
            self._enter_state("SIT")
            return
        # Clamp here as well as in the dispatcher: a state handler that is only
        # safe when called from one place is a trap.
        low = float(self.screen_left)
        high = float(self.screen_left + self.screen_w - self.sprite_w)
        if self.x_pos <= low or self.x_pos >= high:
            self.x_pos = max(low, min(high, self.x_pos))
            self._enter_state("SIT")

    def _tick_pounce(self):
        """The airborne half of a pounce: same arc as JUMP."""
        self._tick_jump()

    def _steer(self, cursor_dx, speed):
        """Move toward the cursor without overshooting it.

        Returns False once the pet is lined up with the cursor (within
        STEER_DEADZONE). Turning only outside the dead zone, and never
        stepping past the target, is what stops the pet from flipping left
        and right every tick while running in place under a cursor it
        cannot reach.
        """
        if abs(cursor_dx) <= STEER_DEADZONE:
            return False
        self.facing = 1 if cursor_dx > 0 else -1
        self.x_pos += self.facing * min(speed, abs(cursor_dx) - STEER_DEADZONE / 2.0)
        return True

    def _tick_stalk(self, cursor_dx, cursor_dist):
        moving = self._steer(cursor_dx, 1.6)
        # Same ledge clamp as WALK; without it the pet strolls off the
        # titlebar and then floats there forever.
        if self._clamp_to_ledge() or cursor_dist < 45:
            self._enter_state("SWAT")
            return
        if not moving:
            # lined up but out of reach (the cursor is overhead): bat at it if
            # it is close, otherwise sit and watch it
            if cursor_dist < 120:
                self._enter_state("SWAT")
            else:
                self._enter_state("IDLE")
            return
        if self.state_timer > STATE_MAX_DURATION.get("STALK", 120):
            self._enter_state("SWAT")

    def _tick_run(self, cursor_dx, cursor_dist):
        moving = self._steer(cursor_dx, 2.2 if self.is_ghost else 3.2)
        if self._clamp_to_ledge() or cursor_dist < 50 or not moving:
            self._enter_state("SWAT")
            return
        if self.state_timer > STATE_MAX_DURATION.get("RUN", 90):
            self._enter_state("SWAT")

    def _tick_walk(self):
        if self.target_x is None:
            self.goal_ledge = None
            self._enter_sit()
            return
        self._maybe_launch_at_goal()

        chasing = self.goal_ledge is not None
        dx = self.target_x - self.x_pos
        self.facing = 1 if dx >= 0 else -1
        self.x_pos += self.facing * (LEAP_GOAL_SPEED if chasing else
                                     float(self.profile.get("walk_speed", 1.1)))

        if self._clamp_to_ledge():
            self.goal_ledge = None
            self._enter_sit()
            return

        limit = (LEAP_GOAL_MAX_TICKS if chasing
                 else STATE_MAX_DURATION.get("WALK", 100))
        if abs(dx) < 3 or self.state_timer > limit:
            # Standing under the goal but never launched: try once more from
            # here before giving up, so the pet ends up on the window rather
            # than parked underneath it.
            if self.goal_ledge is not None:
                anchor = self._goal_anchor(self.goal_ledge, self.sprite_w,
                                           self.x_pos + self.sprite_w / 2.0)
                if self._start_leap(self.goal_ledge, anchor=anchor):
                    return
            self.goal_ledge = None
            self._enter_sit()

    # ---- FALL ----------------------------------------------------------

    def _tick_fall(self, announce=True):
        self.vy = min(self.vy + GRAVITY, TERMINAL_VEL)
        self.y_pos += self.vy
        if self.vx:
            self.x_pos += self.vx
            # A little bleed so a leap settles instead of sliding forever.
            self.vx *= LEAP_AIR_DRAG
            if abs(self.vx) < 0.05:
                self.vx = 0.0

        landing_y = self.floor_y
        if self.tracker is not None:
            try:
                # `step` matters: at terminal velocity the pet moves 12px in a
                # tick, and without it a thin titlebar is rejected as already
                # passed, so the pet falls straight through it to the floor.
                landing_y = self.tracker.find_landing_y(
                    self.x_pos, self.y_pos, self.sprite_w, self.sprite_h,
                    self.floor_y, self.screen_rect, step=self.vy)
            except Exception:
                landing_y = self.floor_y

        if self.y_pos >= landing_y:
            impact = self.vy
            self.y_pos = landing_y
            self.vy = 0.0
            self.vx = 0.0
            ledge = None
            if landing_y < self.floor_y - 5:
                ledge = self._ledge_at_landing(landing_y) or self._ledge_under_pet()
            if ledge is not None:
                self.current_ledge = ledge
                self.y_pos = self._stand_y(ledge)
                self._enter_state("PERCH")
                self._remember_ledge(ledge)
            else:
                # Either the floor, or a landing surface we cannot identify.
                # Never enter PERCH without a ledge: the next tick dereferences
                # it and the process aborts.
                self.current_ledge = None
                self.y_pos = self.floor_y
                self._enter_sit()
            event = self.land_event or ("drop" if announce else None)
            self.land_event = None
            if impact > 4.0:
                heavy = self._heavy_landing or self.profile.get("heavy")
                self._emit_fx("thud" if heavy else "dust")
                self._squash_ticks = LAND_SQUASH_TICKS
                if self._heavy_landing:
                    self.sounds.play("land", ambient=True)
            self._heavy_landing = False
            if event == "fling" or self._user_drop:
                self.sounds.play("land")
            self._user_drop = False
            if self._shaken or (event == "fling" and impact >= TERMINAL_VEL
                                and random.random() < 0.4):
                self._get_dizzy()
            elif event:
                self._speak_contextual(event)

        low = float(self.screen_left)
        high = float(self.screen_left + self.screen_w - self.sprite_w)
        self.x_pos = max(low, min(high, self.x_pos))

    # ---- CLIMB ---------------------------------------------------------

    def _try_start_climb(self):
        """Grab a nearby edge. Returns True if a climb started."""
        if self.tracker is None:
            return False
        try:
            edge = self.tracker.find_nearby_climbable(
                self.x_pos, self.y_pos, self.sprite_w, self.sprite_h,
                reach_px=35.0, screen_rect=self.screen_rect)
        except Exception:
            return False
        if edge is None or edge.height < self.sprite_h:
            return False

        self.climb_target_edge = edge
        self._enter_state("CLIMB")
        if edge.is_left_edge:
            self.x_pos = float(edge.x)
        else:
            self.x_pos = float(edge.x - self.sprite_w)

        # Hugging the edge must not teleport the pet across the screen.
        low = float(self.screen_left)
        high = float(self.screen_left + self.screen_w - self.sprite_w)
        self.x_pos = max(low, min(high, self.x_pos))

        # Budget for the whole climb, measured once from where we started.
        self.climb_budget_ticks = self._climb_budget(edge)
        self.sounds.play("climb", ambient=True)
        self._speak_contextual("climb")
        return True

    def _tick_climb(self):
        edge = self.climb_target_edge
        if edge is None:
            self._enter_fall()
            return

        # The window we are climbing may have closed. Without this check the
        # pet scrambles up thin air for the rest of the timeout.
        if self.tracker is not None:
            try:
                if not self.tracker.edge_still_present(edge):
                    self.climb_target_edge = None
                    self._enter_fall()
                    return
            except Exception:
                pass

        self.y_pos -= CLIMB_SPEED
        target_y = float(edge.y_top) - self.sprite_h

        if self.y_pos <= target_y:
            self.y_pos = target_y
            self.climb_target_edge = None
            ledge = self._ledge_under_pet()
            if ledge is not None:
                self.current_ledge = ledge
                self._enter_state("PERCH")
                self._speak_contextual("perch")
            else:
                self.current_ledge = None
                self._enter_fall()
            return

        # Budget measured when the climb started. The old fixed 10s cap
        # (200 ticks at 1.5px/tick = 300px) was shorter than a single screen,
        # so climbs could never complete and the pet always fell off. It must
        # not be recomputed from the remaining distance either: that shrinks
        # every tick and trips the timeout just before the top.
        budget = self.climb_budget_ticks or self._climb_budget(edge)
        if self.state_timer > budget + CLIMB_GRACE_TICKS:
            self.climb_target_edge = None
            self._enter_fall()

    def _climb_budget(self, edge):
        """Ticks needed to climb from the current position to the top."""
        distance = max(0.0, self.y_pos - (float(edge.y_top) - self.sprite_h))
        return int(distance / max(0.1, CLIMB_SPEED)) + 4

    # ---- PERCH ---------------------------------------------------------

    def _tick_perch(self, cursor_dx, cursor_dist):
        if not self._validate_current_ledge():
            # The window moved or closed: fall, but not through a stale ledge.
            self.current_ledge = None
            self.vy = 0.0
            self._enter_fall()
            return

        ledge = self.current_ledge
        self.y_pos = self._stand_y(ledge)
        self.x_pos = max(float(ledge.x_min),
                         min(float(ledge.x_max - self.sprite_w), self.x_pos))

        min_dur = STATE_MIN_DURATION.get("PERCH", 40)
        max_dur = STATE_MAX_DURATION.get("PERCH", 120)
        if self.state_timer < min_dur:
            return
        if (self.state_timer - min_dur) % TRANSITION_EVAL_INTERVAL != 0:
            return

        # Max duration used to be folded into a roll that could not win,
        # because 200 ticks is itself one of the evaluated ticks: the pet
        # always left at exactly max_dur and the probability never applied.
        if self.state_timer >= max_dur:
            # Prefer travelling to another window over stepping off into empty
            # air; only drop to the floor if nothing else is in reach.
            if self._start_exploring():
                return
            self._leave_ledge()
            return

        # Landed: now live up here for a while, in character (groom, stare,
        # nap, wander the titlebar...). Leaving is just one of the options,
        # which is what stops the floor-to-ceiling ping-pong.
        self._pick_idle_transition(cursor_dx, cursor_dist)
        if self.state == "PERCH":
            self._enter_state("SIT")

    def _leave_ledge(self):
        self.current_ledge = None
        self.goal_ledge = None
        self.vy = 0.0
        self.vx = 0.0
        self._enter_fall()

    # =================================================================
    # Painting
    # =================================================================

    def _sprite_offset(self):
        """Y of the sprite inside its widget.

        The widget is sprite_h + bubble_h tall and the bubble is laid out above
        the sprite, so with a bubble showing the sprite belongs at the *bottom*.
        This had it backwards, which drew the sprite underneath the opaque
        bubble: the pet vanished every time it spoke, at every scale.
        """
        if not self.speech_text:
            return 0
        return 0 if self.speech_below else self.speech_bubble_h

    @guarded
    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)

        frame = self.get_current_frame()
        sprite_y = self._sprite_offset()
        if frame is not None and not self.phased_out():
            painter.drawPixmap(0, sprite_y, frame)

        # Without the particle overlay, fall back to the static heart.
        if self.fx is None and self.heart_timer > 0 \
                and self.heart_pixmap is not None:
            hx = self.sprite_w // 2 - self.heart_pixmap.width() // 2
            top = sprite_y - self.heart_pixmap.height() + 6
            if self.speech_text and not self.speech_below:
                top -= self.speech_bubble_h - SPEECH_TAIL_H
            painter.drawPixmap(hx, max(0, top), self.heart_pixmap)

        if self.speech_text and self.speech_lines:
            self._paint_speech_bubble(painter)

        painter.end()

    def _paint_speech_bubble(self, painter):
        """A pixel-art bubble: notched corners, chunky border, stepped tail.

        The text types itself in, `speech_reveal` characters at a time; the
        bubble itself is laid out for the whole line so it never resizes.
        """
        painter.setFont(self.speech_font)
        metrics = painter.fontMetrics()

        width = self.speech_bubble_w
        body_h = self.speech_bubble_h - SPEECH_TAIL_H
        below = self.speech_below
        u = SPEECH_PIXEL

        box_x = max(0, (self.width() - width) // 2)
        box_y = self.height() - body_h if below else 0

        ink = QColor(34, 24, 38)
        paper = QColor(255, 252, 244)
        shade = QColor(226, 218, 206)

        def block(x, y, w, h, colour):
            painter.fillRect(QRect(int(x), int(y), int(w), int(h)), colour)

        # border: a rectangle with its corner pixels knocked out
        block(box_x + u, box_y, width - 2 * u, body_h, ink)
        block(box_x, box_y + u, width, body_h - 2 * u, ink)
        # paper, inset by one border pixel, corners notched again
        block(box_x + 2 * u, box_y + u, width - 4 * u, body_h - 2 * u, paper)
        block(box_x + u, box_y + 2 * u, width - 2 * u, body_h - 4 * u, paper)
        # a soft shadow along the bottom inside edge
        block(box_x + 2 * u, box_y + body_h - 2 * u, width - 4 * u, u, shade)

        # stepped tail, pointing at the pet: rows of 5, 3, 1 pixels
        steps = max(2, SPEECH_TAIL_H // u)
        tail_w = (2 * steps - 1) * u
        tail_x = box_x + max(2 * u, min(width - tail_w - 2 * u,
                                        self.sprite_w // 2 - tail_w // 2))
        for i in range(steps):
            x = tail_x + i * u
            w = tail_w - 2 * i * u
            y = box_y - (i + 1) * u if below else box_y + body_h + i * u
            block(x, y, w, u, ink)
            if w > 2 * u:
                block(x + u, y, w - 2 * u, u, paper)
        # open the border where the tail joins
        seam = box_y if below else box_y + body_h - u
        block(tail_x + u, seam, tail_w - 2 * u, u, paper)

        painter.setPen(ink)
        line_h = metrics.height()
        text_y = box_y + SPEECH_PADDING_Y + metrics.ascent()
        remaining = self.speech_reveal if self.speech_reveal else 0
        for line in self.speech_lines:
            if remaining <= 0:
                break
            shown = line[:remaining]
            remaining -= len(line) + 1
            painter.drawText(box_x + SPEECH_PADDING_X, int(text_y), shown)
            text_y += line_h

    # =================================================================
    # Mouse
    # =================================================================

    @guarded
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self.dragging = False
        self.drag_offset = event.globalPos() - self.pos()
        self.drag_start_y = self.y_pos
        self._drag_history = [(time.time(), event.globalPos().x(),
                               event.globalPos().y())]
        self._state_before_press = self.state
        self._state_timer_before_press = self.state_timer
        self._shake_times = []
        self._shake_dir = 0
        self._shaken = False
        self._enter_state("DRAG")
        self.vy = 0.0
        # Grabbing a perched pet must clear the ledge, or the ground-lock and
        # the ledge clamp fight each other for the whole drag.
        self.current_ledge = None
        if self.interaction is not None:
            self.interaction.record_drag_start(self.y_pos)

    @guarded
    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return
        target = event.globalPos() - self.drag_offset
        if (target - self.pos()).manhattanLength() > 4:
            if not self.dragging:
                self._mark_user_moment()
                self.sounds.play("grab")
            self.dragging = True
        self._track_shake(event.globalPos().x())
        # Keep the last few cursor samples so a release can measure the throw.
        self._drag_history.append((time.time(), event.globalPos().x(),
                                   event.globalPos().y()))
        if len(self._drag_history) > 8:
            del self._drag_history[:-8]
        # Clamp into the *virtual desktop*: an unclamped drag could leave the
        # pet at y=-5000, which is a twenty second fall. The old clamp used
        # max(0, ...) and QDesktopWidget, so a monitor left of or above the
        # primary one was off limits.
        left, top, right, bottom = self._virtual_desktop_rect()
        bubble_h = self.speech_bubble_h if self.speech_text else 0
        self.x_pos = float(max(left, min(target.x(), right - self.sprite_w)))
        self.y_pos = float(max(top - bubble_h,
                               min(target.y() + bubble_h, bottom - self.sprite_h)))
        self._sync_widget_geometry()
        self.last_activity_tick = self.tick_count

    def _track_shake(self, x):
        """Count left/right reversals while dragging: shaken = dizzy."""
        # called before this sample is appended: [-1] is the previous one
        history = self._drag_history
        if not history:
            return
        dx = x - history[-1][1]
        if abs(dx) < 6:
            return
        direction = 1 if dx > 0 else -1
        if self._shake_dir and direction != self._shake_dir:
            now = time.time()
            self._shake_times = [t for t in self._shake_times
                                 if now - t <= SHAKE_WINDOW] + [now]
            if len(self._shake_times) >= SHAKE_REVERSALS and not self._shaken:
                self._shaken = True
                self.sounds.play("shake")
        self._shake_dir = direction

    def _virtual_desktop_rect(self):
        """Union of every connected screen, in Qt logical pixels."""
        app = QApplication.instance()
        screens = app.screens() if app is not None else []
        if not screens:
            return (self.screen_left, self.screen_top,
                    self.screen_left + self.screen_w,
                    self.screen_top + self.screen_h)
        left = min(s.geometry().x() for s in screens)
        top = min(s.geometry().y() for s in screens)
        right = max(s.geometry().x() + s.geometry().width() for s in screens)
        bottom = max(s.geometry().y() + s.geometry().height() for s in screens)
        return (left, top, right, bottom)

    def _desktop_right(self):
        return self._virtual_desktop_rect()[2]

    def _desktop_bottom(self):
        return self._virtual_desktop_rect()[3]

    @guarded
    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        was_dragging = self.dragging
        self.dragging = False
        if self._suppress_release:
            # second release of a double-click: the trick already answered
            self._suppress_release = False
            if not was_dragging:
                return

        if not was_dragging:
            self._mark_user_moment()
            was_asleep = self._state_before_press == "SLEEP"
            self.energy = min(100.0, self.energy + ENERGY_GAIN_PET)
            self.petted_until_tick = self.tick_count
            # A pet deserves attention: end a snooze early.
            self._snoozed_until = 0
            event_type = (self.interaction.record_pet()
                          if self.interaction is not None else "pet_first")
            if was_asleep and self._state_timer_before_press > 40:
                # poked awake: a grumble, not a purr
                self._enter_sit()
                self._start_one_shot("SULK")
                self.sounds.play("no")
                self._speak_contextual("grumpy")
                return
            self.heart_timer = 20
            if event_type == "pet_spam" and self.profile.get("aloof") \
                    and not self.flies and random.random() < 0.5:
                # a cat has had enough, thank you: it walks off
                self._enter_sit()
                cx = self._cursor.x()
                away = -1 if cx > self.x_pos + self.sprite_w / 2.0 else 1
                low = float(self.screen_left)
                high = float(self.screen_left + self.screen_w - self.sprite_w)
                self.target_x = max(low, min(high, self.x_pos + away * 220))
                self._enter_state("WALK")
                self._emit_fx("anger")
            elif event_type == "pet_spam":
                self._enter_sit()
                self._start_one_shot("HAPPY")
                self.sounds.play("bark" if self.current_skin == "shiba"
                                 else "happy")
            else:
                self._enter_sit()
                self._start_one_shot("PETTED")
                self.sounds.play(PET_CUES.get(self.current_skin, "pop"))
            if self.interaction is not None:
                # A sleepy pet grumbles instead of engaging.
                if self.mood() == "sleepy" and random.random() < 0.5:
                    event_type = "sleepy"
                self._speak_contextual(event_type)
            else:
                self.request_dialogue()
            return

        # Dragged to a different monitor: the geometry signals only fire for
        # monitor *changes* the window manager announces, not for a widget the
        # user just carried across the seam and threw. Re-read which screen
        # the pet is on before the arc decides where the floor is.
        self._refresh_screen_geometry()

        self._mark_user_moment()
        if self.flies:
            # Resume hovering from wherever it was released, then ease back
            # down. The old code assigned y_pos and then immediately
            # overwrote it with the floor value, so it snapped instead.
            self.ghost_hover_y = self.y_pos
            self._enter_sit()
            if self._shaken:
                self._get_dizzy()
            return

        vx, vy = self._fling_velocity()
        self._enter_fall()
        self._user_drop = True
        if vx is not None:
            # _enter_fall zeroes the momentum (drops should not drift), and a
            # throw is not a drop: the velocity is applied after the reset.
            self.vx = vx
            self.vy = vy
            self.land_event = "fling"
            self.sounds.play("whoosh")

    def _fling_velocity(self):
        """(vx, vy) in px/tick from the drag's last samples, or (None, None).

        Releasing mid-drag used to dump the momentum: the cursor was moving
        fast, the pet stopped dead, and a "thrown" pet fell like a dropped
        one. Momentum is capped so the fling stays readable (and inside the
        desktop), and a gentle release stays a plain drop.
        """
        history = self._drag_history
        if len(history) < 2:
            return None, None
        (t0, x0, y0), (t1, x1, y1) = history[0], history[-1]
        elapsed = t1 - t0
        if elapsed <= 0:
            return None, None
        vx = (x1 - x0) / elapsed / TICKS_PER_SECOND
        vy = (y1 - y0) / elapsed / TICKS_PER_SECOND
        if abs(vx) < FLING_MIN_SPEED and abs(vy) < FLING_MIN_SPEED:
            return None, None
        vx = max(-FLING_MAX_SPEED, min(FLING_MAX_SPEED, vx))
        # An upward throw rises; a downward throw just drops with intent. The
        # negative side is capped so nobody can launch the pet into orbit.
        vy = max(-FLING_MAX_SPEED, min(FLING_MAX_SPEED, vy))
        # A sideways release at floor level reads as nothing on the ground, so
        # give it a small hop: it visibly travels, then settles.
        near_floor = abs(self.y_pos - self.floor_y) <= self.sprite_h
        if near_floor and abs(vy) < FLING_MIN_SPEED:
            vy = -2.5
        return vx, vy

    # =================================================================
    # Context menu
    # =================================================================

    @guarded
    def contextMenuEvent(self, event):
        menu = self._build_menu()
        # The capture cannot happen from inside the menu action: the QMenu is
        # still on screen, so the model would describe the pet's own menu.
        # _menu_look only sets the flag; the capture waits for the menu to go.
        menu.exec_(event.globalPos())
        # Built fresh on every right-click with the pet as parent: without
        # this every menu (and its submenus) stayed alive until exit.
        menu.deleteLater()
        if self._pending_capture:
            self._pending_capture = False
            QTimer.singleShot(MENU_CLOSE_DELAY_MS, self._run_pending_capture)

    def _build_menu(self, menu=None):
        """The whole context menu.

        Built here rather than in contextMenuEvent so the tray icon can offer
        exactly the same menu -- which is the only way to reach the pet when
        click-through mode is on. Pass an existing `menu` to repopulate it in
        place; the tray does that on aboutToShow so the checkmarks are never
        stale.
        """
        if menu is None:
            menu = QMenu(self)
        else:
            menu.clear()

        skin_menu = menu.addMenu("Switch Companion")
        for name in discover_skins() or self.available_skins:
            action = QAction(name.capitalize(), self)
            action.setCheckable(True)
            action.setChecked(name == self.current_skin)
            action.triggered.connect(lambda _c=False, n=name: self._switch_skin(n))
            skin_menu.addAction(action)

        scale_menu = menu.addMenu("Scale")
        for label, factor in SCALE_PRESETS.items():
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(self.scale_factor == factor)
            action.triggered.connect(lambda _c=False, f=factor: self._set_scale(f))
            scale_menu.addAction(action)

        menu.addSeparator()

        self._add_toggle(menu, "Climb Windows", self.climb_enabled, self._set_climb)
        self._add_toggle(menu, "Allow Jumping", self.allow_jump,
                         lambda checked: self._set_movement_flag("allow_jump", checked))
        self._add_toggle(menu, "Allow Chasing", self.allow_chase,
                         lambda checked: self._set_movement_flag("allow_chase", checked))
        self._add_toggle(menu, "Click Through", self.click_through,
                         self._set_click_through)
        if self.focus.configured():
            focus_action = QAction("Focus Timer (%s)" % self.focus.describe(), self)
            focus_action.setEnabled(False)
            menu.addAction(focus_action)

        speak_action = QAction("Speak", self)
        speak_action.triggered.connect(self._menu_speak)
        menu.addAction(speak_action)

        treat_action = QAction("Give a Treat", self)
        treat_action.setToolTip("A snack: it perks the pet up")
        treat_action.triggered.connect(self._menu_treat)
        menu.addAction(treat_action)

        play_menu = menu.addMenu("Play")
        for label, handler, tip in (
                ("Dance!", lambda: self._start_dance(user=True),
                 "A little dance, with music"),
                ("Do a Trick", self._do_trick,
                 "Also: double-click the pet"),
                ("Laser Pointer", self._menu_laser,
                 "It chases your cursor for %d seconds" % LASER_SECONDS)):
            action = QAction(label, self)
            action.setToolTip(tip)
            action.triggered.connect(handler)
            play_menu.addAction(action)

        self._build_sound_menu(menu)

        shush_menu = menu.addMenu("Shush...")
        for label, minutes in (("15 minutes", 15), ("1 hour", 60),
                               ("2 hours", 120)):
            action = QAction(label, self)
            action.triggered.connect(
                lambda _c=False, m=minutes: self._menu_shush(m))
            shush_menu.addAction(action)
        unshush = QAction("Unshush Now", self)
        unshush.setEnabled(bool(self._snoozed_until))
        unshush.triggered.connect(lambda: self._menu_shush(0))
        shush_menu.addAction(unshush)

        self._build_privacy_menu(menu)

        personality_action = QAction("Edit Personality...", self)
        personality_action.triggered.connect(self._menu_edit_personality)
        menu.addAction(personality_action)

        ignore_action = QAction("Ignore Windows...", self)
        ignore_action.setToolTip("Windows the pet must not stand on")
        ignore_action.triggered.connect(self._menu_edit_ignores)
        menu.addAction(ignore_action)

        reset_action = QAction("Reset Position", self)
        reset_action.triggered.connect(self._reset_position)
        menu.addAction(reset_action)

        forget_action = QAction("Forget Me", self)
        forget_action.setToolTip("Delete saved position, skin and energy")
        forget_action.triggered.connect(self._menu_forget)
        menu.addAction(forget_action)

        memory_action = QAction("What Do You Remember?", self)
        memory_action.setToolTip("Read back the facts the pet has kept")
        memory_action.triggered.connect(self._menu_memory)
        menu.addAction(memory_action)

        menu.addSeparator()
        stats = QAction(self._stats_text(), self)
        stats.setEnabled(False)
        menu.addAction(stats)

        status = QAction(self._status_text(), self)
        status.setEnabled(False)
        menu.addAction(status)
        if self._ai_problem():
            retry = QAction("Retry AI", self)
            retry.triggered.connect(self._retry_ai)
            menu.addAction(retry)

        menu.addSeparator()
        quit_action = QAction("Exit", self)
        quit_action.triggered.connect(self.shutdown)
        menu.addAction(quit_action)
        return menu

    def _build_sound_menu(self, menu):
        audio = ai_companion.CONFIG.setdefault("audio", {})
        sound_menu = menu.addMenu("Sound")
        self._add_toggle(sound_menu, "Sounds", self.sounds.enabled,
                         lambda c: self._set_audio("enabled", bool(c)))
        self._add_toggle(sound_menu, "Talking Voice", self.sounds.voice,
                         lambda c: self._set_audio("voice", bool(c)))
        ambient = self._add_toggle(
            sound_menu, "Sounds On Its Own (rare)", self.sounds.ambient,
            lambda c: self._set_audio("ambient", bool(c)))
        ambient.setToolTip("Noises the pet makes without being touched. "
                           "At most one every %d seconds."
                           % int(self.sounds.ambient_gap))
        sound_menu.addSeparator()
        for label, level in (("Quiet", 0.15), ("Normal", 0.3), ("Loud", 0.55)):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(abs(float(audio.get("volume", 0.3)) - level) < 0.01)
            action.triggered.connect(
                lambda _c=False, v=level: self._set_audio("volume", v))
            sound_menu.addAction(action)

    def _set_audio(self, key, value):
        """Change one audio setting, persist it, and play a sample."""
        audio = ai_companion.CONFIG.setdefault("audio", {})
        audio[key] = value
        self.sounds.update(audio)
        try:
            ai_companion.save_config()
        except Exception:
            log_exception("save audio config")
        if self.sounds.enabled:
            self.sounds.play(self.sounds.voice_cue(self.current_skin))

    def _build_privacy_menu(self, menu):
        """Screens this pet is allowed to look at, and what happens to them."""
        privacy = menu.addMenu("Screenshots && Privacy")
        settings = ai_companion.screenshot_settings()

        self._add_toggle(privacy, "Look At My Screen", settings.get("enabled", False),
                         self._set_screenshots_enabled)

        look_action = QAction("Look At This", self)
        look_action.setToolTip("Capture the screen once and comment on it")
        look_action.triggered.connect(self._menu_look)
        privacy.addAction(look_action)

        privacy.addSeparator()

        block_action = QAction("Never Capture This App", self)
        block_action.setToolTip("Blocklist the focused application, by process name")
        block_action.triggered.connect(self._menu_block_current_app)
        privacy.addAction(block_action)

        edit_action = QAction("Edit Blocklist...", self)
        edit_action.setToolTip("Process and title rules that are never captured")
        edit_action.triggered.connect(self._menu_edit_blocklist)
        privacy.addAction(edit_action)

        ocr_cfg = ai_companion.ocr_settings()
        self._add_toggle(privacy, "OCR Content Filter", ocr_cfg.get("enabled", True),
                         self._set_ocr_enabled)

        filter_action = QAction("Filter: %s" % self._ocr_description(), self)
        filter_action.setEnabled(False)
        privacy.addAction(filter_action)

        status = QAction(ai_companion.screenshots_status_text(), self)
        status.setEnabled(False)
        privacy.addAction(status)

        log_action = QAction("What Happened To My Screens...", self)
        log_action.triggered.connect(self._menu_capture_log)
        privacy.addAction(log_action)

    def _ocr_description(self):
        if ocr_guard is None:
            return "unavailable"
        try:
            return ocr_guard.describe(ai_companion.ocr_settings())
        except Exception:
            return "unavailable"

    # ---- privacy actions -----------------------------------------------

    def _set_screenshots_enabled(self, checked):
        ai_companion.CONFIG.setdefault("screenshots", {})["enabled"] = bool(checked)
        ai_companion.save_config()
        if checked:
            self._speak_contextual("happy")
        else:
            self.speak("ok, no more looking.")

    def _set_ocr_enabled(self, checked):
        ocr = ai_companion.CONFIG.setdefault("screenshots", {}).setdefault("ocr", {})
        ocr["enabled"] = bool(checked)
        ai_companion.save_config()
        self.speak("filter %s." % ("on" if checked else "off"))

    def _menu_block_current_app(self):
        """One click: the app in front of me is never captured again."""
        window = self._focused_window()
        if window is None:
            self.speak("nothing focused to block.")
            return
        process = (window.process or "").strip()
        title = (window.title or "").strip()
        if process:
            entry = "proc:%s" % process
            added = ai_companion.add_blocklist_entry(entry)
            self.speak("%s %s." % ("blocked" if added else "already blocked",
                                   process))
            return
        # No process name (some X11 clients do not set _NET_WM_PID): fall back
        # to a title rule, which still keeps the window itself out.
        if title:
            entry = "title:%s" % title[:60]
            added = ai_companion.add_blocklist_entry(entry)
            self.speak("%s that window." % ("blocked" if added else "already blocked"))
            return
        self.speak("can't tell what that app is.")

    def _focused_window(self):
        """The focused window, refreshing the tracker if there is time."""
        if self.tracker is None:
            return None
        try:
            self.tracker.get_windows(force_refresh=True)
            return self.tracker.get_active_window()
        except Exception:
            return None

    def _menu_edit_blocklist(self):
        from PyQt5.QtWidgets import QInputDialog

        current = ", ".join(ai_companion.blocklist())
        text, ok = QInputDialog.getText(
            self, "Screenshot blocklist",
            "Comma-separated rules the pet must never send:\n"
            "  proc:vscode      exact process name (proc:foo* for a prefix)\n"
            "  pid:1234         one process id\n"
            "  title:inbox      window-title fragment\n"
            "  re:^secret\\s     window-title regex\n"
            "  fragment         any title or process containing it",
            text=current)
        if not ok:
            return
        entries = [part.strip() for part in text.split(",") if part.strip()]
        if ai_companion.set_blocklist(entries):
            self.speak("blocklist saved.")
        else:
            self.speak("i couldn't save that.")

    def _menu_capture_log(self):
        from PyQt5.QtWidgets import QMessageBox

        decisions = ai_companion.capture_decisions(12)
        if not decisions:
            body = "Nothing has been captured this session."
        else:
            lines = []
            for entry in decisions:
                when = time.strftime("%H:%M:%S", time.localtime(entry["at"]))
                reason = (" - %s" % entry["reason"]) if entry["reason"] else ""
                lines.append("%s  %s%s" % (when, entry["action"], reason))
            body = "\n".join(lines)
        box = QMessageBox(self)
        box.setWindowTitle("Screen capture log")
        box.setText("Recent captures (nothing here contains screen content)")
        box.setDetailedText(body)
        box.exec_()

    def _add_toggle(self, menu, label, checked, handler):
        action = QAction(label, self)
        action.setCheckable(True)
        action.setChecked(bool(checked))
        action.triggered.connect(handler)
        menu.addAction(action)
        return action

    def _stats_text(self):
        """One line of "what has this pet done", from the saved state."""
        data = self.state_store.data if self.state_store is not None else {}
        alive = 0.0
        if self.state_store is not None:
            try:
                alive = self.state_store.total_seconds_alive(
                    time.time() - self.session_start)
            except Exception:
                alive = 0.0
        hours, remainder = divmod(int(alive), 3600)
        minutes = remainder // 60
        age = ("%dh %02dm" % (hours, minutes)) if hours else "%dm" % minutes
        total_days, remainder = divmod(int(alive), 86400)
        total_hours = remainder // 3600
        lifetime = ""
        if total_days:
            # The lifetime counter is what anniversaries use; the menu should
            # not pretend the pet is new on every restart.
            lifetime = " (%dd %dh old)" % (total_days, total_hours)
        pets = 0
        if self.interaction is not None:
            pets = len(getattr(self.interaction, "pets", []) or [])
        return "alive %s%s - %d sessions - %d pets - %d treats" % (
            age, lifetime, data.get("pet_count", 0) or 0, pets,
            data.get("treats_given", 0) or 0)

    def _menu_treat(self):
        """A snack: energy, a heart, and a line."""
        if self.tick_count < self._treat_until_tick:
            self.speak("i'm still chewing.")
            return
        self._treat_until_tick = self.tick_count + 20 * 30   # 30s
        self.energy = min(100.0, self.energy + 25.0)
        self.heart_timer = 25
        self.petted_until_tick = self.tick_count
        if self.state_store is not None:
            current = self.state_store.data.get("treats_given", 0) or 0
            self.state_store.touch(treats_given=int(current) + 1)
            self.state_store.save(force=True)
        self._mark_user_moment()
        self.sounds.play("eat")
        if not self._start_one_shot("EAT"):
            self.sounds.play("chime")
        self._speak_contextual("treat")

    # ---- play ----------------------------------------------------------

    def _get_dizzy(self):
        self._shaken = False
        self._start_one_shot("DIZZY")
        self.sounds.play("dizzy")
        self._speak_contextual("dizzy")

    def _start_dance(self, user=True):
        if not self._start_one_shot("DANCE"):
            return
        if user:
            self._mark_user_moment()
        self.sounds.play("dance", ambient=not user)
        if user or random.random() < 0.5:
            self._speak_contextual("dance")

    def _do_trick(self):
        """A hop with a twirl: turns around mid-air and lands facing you."""
        self._mark_user_moment()
        if not self._start_one_shot("HAPPY"):
            return
        self.facing = -self.facing
        self._emit_fx("sparkles")
        self.sounds.play("jump")
        self._speak_contextual("trick")

    def _menu_laser(self):
        self._mark_user_moment()
        if self.flies:
            # flyers glide after it instead of sprinting
            self._enter_state("HAUNT")
        self._laser_until = self.tick_count + int(LASER_SECONDS * TICKS_PER_SECOND)
        self.sounds.play("alert")
        self._speak_contextual("laser")

    @guarded
    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._suppress_release = True
        self._do_trick()

    def _set_movement_flag(self, name, checked):
        """Menu toggles that now survive a restart (they used to be memory-only)."""
        setattr(self, name, bool(checked))
        self._save_state(force=True)

    def _set_climb(self, checked):
        self.climb_enabled = bool(checked)
        self._save_state(force=True)

    def _set_click_through(self, checked):
        """Let clicks fall through the pet to whatever is underneath."""
        self.click_through = bool(checked)
        self._apply_click_through()
        if self.click_through:
            self.speak("click-through on. use the tray icon." if self.tray
                       else "click-through on. use the menu again to undo.")
        self._save_state(force=True)

    def _apply_click_through(self):
        """Keep the window flag in sync with `self.click_through`."""
        try:
            flags = window_flags(self.click_through)
            if self.windowFlags() != flags:
                self.setWindowFlags(flags)
                self.show()
                self.raise_()
        except Exception:
            log_exception("click-through")

    def _menu_look(self):
        """Queue a one-off capture. See contextMenuEvent for the delay."""
        self.heart_timer = 20
        self._pending_capture = True

    def _run_pending_capture(self):
        """Fires once the context menu is gone."""
        if self.speech_timer == 0:
            self.speak("one sec, looking.")
        self.maybe_screenshot(force=True)

    def _menu_edit_personality(self):
        """Edit this companion's system prompt without touching the JSON."""
        from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QLabel,
                                     QTextEdit, QVBoxLayout)

        current = ai_companion.get_system_prompt(self.current_skin)
        dialog = QDialog(self)
        dialog.setWindowTitle("%s personality" % self.current_skin.capitalize())
        dialog.resize(520, 360)

        box = QVBoxLayout(dialog)
        hint = QLabel(
            "Instructions sent to the model with every line. One short sentence "
            "of instructions is plenty. Leave empty to use the default.")
        hint.setWordWrap(True)
        box.addWidget(hint)

        editor = QTextEdit(dialog)
        editor.setPlainText(current)
        box.addWidget(editor)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        box.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            text = editor.toPlainText().strip()
            if ai_companion.set_system_prompt(self.current_skin, text):
                self._speak_contextual("happy")
            else:
                self.speak("i couldn't save that. check companion_config.json")

    def _menu_edit_ignores(self):
        from PyQt5.QtWidgets import QInputDialog

        current = ", ".join(self._behaviour().get("ignore_windows", []))
        text, ok = QInputDialog.getText(
            self, "Ignore windows",
            "Comma-separated title or process fragments the pet must not\n"
            "perch on or climb (it can still look at them):", text=current)
        if not ok:
            return
        patterns = [part.strip() for part in text.split(",") if part.strip()]
        behaviour = dict(self._behaviour())
        behaviour["ignore_windows"] = patterns
        ai_companion.CONFIG.setdefault("behaviour", {}).update(behaviour)
        ai_companion.save_config()
        if self.tracker is not None:
            self.tracker.set_ignored_windows(patterns)
        self._speak_contextual("pet_first")

    def _menu_forget(self):
        self.forget_state()
        # "Forget Me" means start over: the stored click-through and position
        # are gone, so the pet should not keep pretending they aren't.
        self.click_through = False
        self._apply_click_through()
        self._reset_position()
        self._speak_contextual("greeting")

    def _menu_memory(self):
        """Read the pet's memory back to the user, read-only.

        Deliberately read-only display: the memory file gets written by what
        the model said in the MEMORY: line of a reply, and the only delete
        action is the explicit "forget everything" button here.
        """
        from PyQt5.QtWidgets import QMessageBox

        facts = ai_companion.memory_facts()
        if facts:
            body = "\n".join("- %s" % fact for fact in facts)
        else:
            body = ("Nothing yet. The pet keeps short facts about you as they "
                    "come up (memory must be enabled in the config).")
        box = QMessageBox(self)
        box.setWindowTitle("What do you remember?")
        box.setText("Facts I've kept (%s)" % len(facts))
        box.setDetailedText(body)
        box.setStandardButtons(QMessageBox.Ok)
        if facts:
            forget = box.addButton("Forget Everything", QMessageBox.DestructiveRole)
        box.exec_()
        if facts and box.clickedButton() == forget:
            confirm = QMessageBox(self)
            confirm.setWindowTitle("Forget everything")
            confirm.setText(
                "Erase the %d fact%s?\nThis does not touch anything else."
                % (len(facts), "" if len(facts) == 1 else "s"))
            confirm.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            confirm.setDefaultButton(QMessageBox.No)
            if confirm.exec_() == QMessageBox.Yes:
                ai_companion.forget_memory()
                self.speak("forgotten. clean slate.")
            else:
                self.speak("phew. keeping them.")

    def _status_text(self):
        status = ai_companion.ai_status()
        state = "AI on" if status["enabled"] else "AI off"
        text = "%s - %s" % (state, status["reason"])
        shushed = self._snooze_status_text()
        if shushed:
            text += " | %s" % shushed
        return text

    def _ai_problem(self):
        """Whether the model is configured but failing, so retry makes sense."""
        return ai_companion.ai_configured() and not ai_companion.ai_status()["enabled"]

    def _retry_ai(self):
        ai_companion.reset_failures()
        self._speak_contextual("idle")

    def _menu_speak(self):
        self.heart_timer = 20
        self.request_dialogue(user_initiated=True)

    def _switch_skin(self, name):
        self.load_skin(name)
        self.available_skins = discover_skins() or self.available_skins
        self.current_ledge = None
        self._reset_position()
        self._save_state(force=True)

    def _set_scale(self, factor):
        self.scale_factor = factor
        self.load_skin(self.current_skin)
        self.current_ledge = None
        self._reset_position()
        self._save_state(force=True)

    def _reset_position(self):
        self.x_pos = float(self.screen_left + self.screen_w // 2 - self.sprite_w // 2)
        self.y_pos = self.floor_y
        self.current_ledge = None
        self.climb_target_edge = None
        self.vy = 0.0
        self._enter_sit()
        self._sync_widget_geometry()

    def setup_tray(self):
        """A tray icon with the same menu.

        Two jobs: it is the way back out of click-through mode, and it is the
        way to reach the pet at all when it is hidden behind a fullscreen app.
        """
        if QSystemTrayIcon is None or self.tray is not None:
            return None
        try:
            if not QSystemTrayIcon.isSystemTrayAvailable():
                return None
            icon = self._tray_icon()
            tray = QSystemTrayIcon(icon, self)
            tray.setToolTip("Desktop Pet")
            tray_menu = QMenu(self)
            # Rebuild on open: the toggles in this menu reflect live state, and
            # a menu built once at startup would show the values it started
            # with.
            tray_menu.aboutToShow.connect(
                functools.partial(self._repopulate_tray_menu, tray_menu))
            self._repopulate_tray_menu(tray_menu)
            tray.setContextMenu(tray_menu)
            tray.activated.connect(self._on_tray_activated)
            tray.show()
            self.tray = tray
            self.tray_menu = tray_menu
            return tray
        except Exception:
            log_exception("tray")
            self.tray = None
            return None

    @guarded
    def _repopulate_tray_menu(self, menu=None):
        self._build_menu(menu if menu is not None else getattr(self, "tray_menu", None))

    def _tray_icon(self):
        """Use the pet's own first sprite as the tray icon if we can."""
        try:
            frames = self._frame_table.get("SIT") or []
            if frames:
                return QIcon(frames[0])
            for frames in self.sprites.values():
                if frames:
                    return QIcon(frames[0])
        except Exception:
            pass
        return QIcon()

    @guarded
    def _on_tray_activated(self, reason):
        if QSystemTrayIcon is not None and reason == QSystemTrayIcon.Trigger:
            self._restack_on_top(force=True)

    def hideEvent(self, event):
        # the particles go wherever the pet goes, including out of a capture
        if self.fx is not None:
            try:
                self.fx.hide()
            except Exception:
                pass
        super().hideEvent(event)

    def shutdown(self):
        """Stop the poller, save state, and quit cleanly."""
        try:
            self.timer.stop()
            self._save_state(force=True)
            if self.fx is not None:
                try:
                    self.fx.close()
                except Exception:
                    pass
            if self.tray is not None:
                try:
                    self.tray.hide()
                except Exception:
                    pass
                self.tray = None
                self.tray_menu = None
            if self.tracker is not None:
                self.tracker.stop_background()
        except Exception:
            log_exception("shutdown")
        QApplication.quit()

    def _install_signal_handlers(self):
        """Save state when the process is asked to stop.

        systemd and pm2 both stop the pet with SIGTERM, and a default handler
        kills the process outright, losing the position and energy.
        """
        import signal

        def handler(signum, _frame):
            try:
                self._save_state(force=True)
            except Exception:
                pass
            os._exit(0)

        for name in ("SIGTERM", "SIGINT", "SIGHUP"):
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                # Not the main thread, or the platform has no such signal.
                pass

    def closeEvent(self, event):
        try:
            self._save_state(force=True)
            if self.tracker is not None:
                self.tracker.stop_background()
        except Exception:
            pass
        super().closeEvent(event)


def _installed_fonts():
    try:
        return set(QFontDatabase().families())
    except Exception:
        return set()


def _pick_speech_font():
    """Fall back to a font that actually exists on this platform.

    Consolas is Windows-only; without a fallback the bubble silently used
    whatever Qt defaulted to, with different metrics and different wrapping.
    """
    global SPEECH_FONT_FAMILY
    installed = _installed_fonts()
    for candidate in (SPEECH_FONT_FAMILY, "Consolas", "DejaVu Sans Mono",
                      "Menlo", "Liberation Mono", "monospace"):
        if candidate in installed:
            SPEECH_FONT_FAMILY = candidate
            return


def main():
    global SPEECH_FONT_FAMILY

    # HiDPI must be requested before the QApplication exists. The pet then
    # works in logical pixels and the tracker is told the devicePixelRatio so
    # window rects line up with where the pet actually is.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    # A Qt.Tool window can be excluded from the "last window" heuristic, which
    # would quit the app when the context menu closed.
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName(PET_WINDOW_TITLE)
    _pick_speech_font()
    rotate_log()

    skins = discover_skins()
    start_skin = "cat" if "cat" in skins else (skins[0] if skins else "cat")
    pet = DesktopPet(skin=start_skin, scale=2)
    pet._install_signal_handlers()
    pet.setup_tray()

    try:
        return app.exec_()
    finally:
        if pet.tracker is not None:
            pet.tracker.stop_background()


if __name__ == "__main__":
    sys.exit(main())
