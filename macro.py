import pyautogui
import time
import json
import os
import re
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pynput import keyboard
import threading
from PIL import Image, ImageTk
import cv2
import numpy as np

# ===== 경로 설정 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
CONFIG_FILE = os.path.join(BASE_DIR, 'macro_config.json')
ICON_ICO = os.path.join(BASE_DIR, 'vertex-logo.ico')
ICON_PNG = os.path.join(BASE_DIR, 'vertex-logo.png')

os.makedirs(IMAGES_DIR, exist_ok=True)

pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0

# ===== 기본 설정 =====
DEFAULT_HOTKEYS = {
    'add_pos': 'f1',
    'toggle_detect': 'f2',
    'toggle_repeat': 'f3',
    'switch_mode': 'f8',
    'stop_all': 'f9',
    'exit': 'esc',
}

AVAILABLE_KEYS = [
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
    'esc', 'pause', 'insert', 'delete', 'home', 'end', 'page_up', 'page_down',
]

# ===== 상태 =====
detect_running = False
repeat_running = False
click_positions_repeat = []
scan_regions = []
input_mode = 'detect'
INTERVAL = 0.03
CONFIDENCE = 0.8
REPEAT_DELAY = 1.0
DETECT_COOLDOWN = 0.5
hotkeys = dict(DEFAULT_HOTKEYS)
always_on_top = True

# detect_rules: 각 감지 이미지별 규칙
# [{ 'trigger': 'btn.png', 'action': 'center'|'coord'|'image',
#    'coords': [(x,y),...], 'click_targets': ['target.png',...] }, ...]
detect_rules = []


def img_path(filename):
    return os.path.join(IMAGES_DIR, filename)


def safe_filename(name):
    base, ext = os.path.splitext(name)
    safe = re.sub(r'[^\x20-\x7E]', '', base).strip()
    safe = re.sub(r'[<>:"/\\|?*\s]+', '_', safe)
    if not safe:
        safe = f'capture_{int(time.time())}'
    return safe + ext


def new_rule(trigger, action='center', coords=None, click_targets=None):
    return {
        'trigger': trigger,
        'action': action,
        'coords': coords or [],
        'click_targets': click_targets or [],
    }


def save_config():
    data = {
        'detect_rules': detect_rules,
        'click_positions_repeat': click_positions_repeat,
        'scan_regions': scan_regions,
        'confidence': CONFIDENCE,
        'interval': INTERVAL,
        'repeat_delay': REPEAT_DELAY,
        'detect_cooldown': DETECT_COOLDOWN,
        'hotkeys': hotkeys,
        'always_on_top': always_on_top,
    }
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def load_config():
    global detect_rules, click_positions_repeat, scan_regions
    global hotkeys, always_on_top
    global CONFIDENCE, INTERVAL, REPEAT_DELAY, DETECT_COOLDOWN
    if not os.path.exists(CONFIG_FILE):
        return
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    detect_rules = data.get('detect_rules', [])
    # 구 형식 호환
    if not detect_rules and 'detect_images' in data:
        old_mode = data.get('detect_click_mode', 'coord')
        old_coords = [tuple(p) for p in data.get('click_positions_detect', [])]
        old_click_imgs = data.get('click_images', [])
        for fname in data['detect_images']:
            r = new_rule(fname)
            if old_mode == 'coord' and old_coords:
                r['action'] = 'coord'
                r['coords'] = list(old_coords)
            elif old_mode == 'image' and old_click_imgs:
                r['action'] = 'image'
                r['click_targets'] = list(old_click_imgs)
            detect_rules.append(r)

    detect_rules = [r for r in detect_rules if os.path.exists(img_path(r['trigger']))]
    for r in detect_rules:
        r['click_targets'] = [f for f in r.get('click_targets', []) if os.path.exists(img_path(f))]
        r['coords'] = [tuple(c) for c in r.get('coords', [])]

    click_positions_repeat = [tuple(p) for p in data.get('click_positions_repeat', [])]
    scan_regions = [tuple(r) for r in data.get('scan_regions', [])]
    CONFIDENCE = data.get('confidence', 0.8)
    INTERVAL = data.get('interval', 0.03)
    REPEAT_DELAY = data.get('repeat_delay', 1.0)
    DETECT_COOLDOWN = data.get('detect_cooldown', 0.5)
    hotkeys.update(data.get('hotkeys', {}))
    always_on_top = data.get('always_on_top', True)


# ===== 이미지 매칭 엔진 =====
_template_cache = {}
_edge_mask_cache = {}


def imread_safe(filepath):
    try:
        arr = np.fromfile(filepath, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def get_template(img_file):
    path = img_path(img_file)
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    cached = _template_cache.get(img_file)
    if cached and cached[0] == mtime:
        return cached[1]
    tpl = imread_safe(path)
    if tpl is not None:
        _template_cache[img_file] = (mtime, tpl)
        _edge_mask_cache.pop(img_file, None)
    return tpl


def get_edge_mask(img_file, template):
    cached = _edge_mask_cache.get(img_file)
    if cached is not None:
        return cached
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=2) > 0
    result = mask if np.count_nonzero(mask) > 10 else None
    _edge_mask_cache[img_file] = result
    return result


def grab_screen(region=None):
    screenshot = pyautogui.screenshot(region=region)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


def multi_scale_match(screen, template, confidence, scales=(1.0, 0.95, 1.05, 0.9, 1.1)):
    best_val, best_loc, best_scale = 0, None, 1.0
    h, w = template.shape[:2]
    sh, sw = screen.shape[:2]
    for scale in scales:
        if scale == 1.0:
            t = template
        else:
            nw, nh = int(w * scale), int(h * scale)
            if nw < 5 or nh < 5 or nw > sw or nh > sh:
                continue
            t = cv2.resize(template, (nw, nh), interpolation=cv2.INTER_AREA)
        if t.shape[0] > sh or t.shape[1] > sw:
            continue
        result = cv2.matchTemplate(screen, t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale
    if best_val >= confidence:
        fh, fw = int(h * best_scale), int(w * best_scale)
        return best_val, best_loc, fw, fh
    return None, None, 0, 0


def verify_color(img_file, template, screen, loc, w, h):
    sy, sx = loc[1], loc[0]
    matched = screen[sy:sy+h, sx:sx+w]
    if matched.shape[:2] != (h, w):
        return True
    # 스케일 달라진 경우 템플릿 리사이즈
    th, tw = template.shape[:2]
    if (h, w) != (th, tw):
        tpl = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
    else:
        tpl = template
    gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.dilate(edges, kernel, iterations=2) > 0
    if np.count_nonzero(fg_mask) <= 10:
        return True
    fg_t = tpl[fg_mask].astype(np.float32)
    fg_m = matched[fg_mask].astype(np.float32)
    diff = np.mean(np.abs(fg_t - fg_m))
    return diff <= 30


def locate_on_screen(img_file, confidence=0.8, region=None, screen=None):
    template = get_template(img_file)
    if template is None:
        return None
    if screen is None:
        screen = grab_screen(region)

    val, loc, fw, fh = multi_scale_match(screen, template, confidence)
    if val is None:
        return None

    if not verify_color(img_file, template, screen, loc, fw, fh):
        return None

    rx, ry = (region[0], region[1]) if region else (0, 0)
    return (rx + loc[0], ry + loc[1], fw, fh,
            rx + loc[0] + fw // 2, ry + loc[1] + fh // 2)


def get_key_obj(name):
    try:
        return getattr(keyboard.Key, name)
    except AttributeError:
        return None


# ===== 매크로 루프 =====
def detect_loop():
    global detect_running
    while detect_running:
        if not detect_rules:
            time.sleep(0.5)
            continue
        try:
            found = False
            regions = scan_regions if scan_regions else [None]
            for rule in detect_rules:
                if not detect_running:
                    break
                trigger = rule['trigger']
                if not os.path.exists(img_path(trigger)):
                    continue
                for region in regions:
                    if not detect_running:
                        break
                    try:
                        screen = grab_screen(region)
                        loc = locate_on_screen(trigger, CONFIDENCE, region, screen)
                        if not loc:
                            continue
                        found = True
                        action = rule.get('action', 'center')

                        if action == 'image' and rule.get('click_targets'):
                            all_ok = True
                            for ci, ct in enumerate(rule['click_targets']):
                                if not detect_running:
                                    all_ok = False
                                    break
                                ct_loc = locate_on_screen(ct, CONFIDENCE, screen=grab_screen())
                                if not ct_loc:
                                    for _ in range(int(5 / max(INTERVAL, 0.01))):
                                        if not detect_running:
                                            break
                                        ct_loc = locate_on_screen(ct, CONFIDENCE, screen=grab_screen())
                                        if ct_loc:
                                            break
                                        time.sleep(INTERVAL)
                                if not ct_loc:
                                    app.log(f'{trigger} | {ct} 못 찾음')
                                    all_ok = False
                                    break
                                pyautogui.click(ct_loc[4], ct_loc[5])
                                app.log(f'{trigger} → [{ci+1}/{len(rule["click_targets"])}] {ct} 클릭')
                            if all_ok:
                                app.log(f'{trigger} → 전체 {len(rule["click_targets"])}개 완료')

                        elif action == 'coord' and rule.get('coords'):
                            for ci, pos in enumerate(rule['coords']):
                                if not detect_running:
                                    break
                                pyautogui.click(*pos)
                                app.log(f'{trigger} → [{ci+1}/{len(rule["coords"])}] ({pos[0]}, {pos[1]}) 클릭')
                                if ci < len(rule['coords']) - 1:
                                    time.sleep(DETECT_COOLDOWN)
                        else:
                            pyautogui.click(loc[4], loc[5])
                            app.log(f'{trigger} → 중앙 클릭')

                        time.sleep(DETECT_COOLDOWN)
                        while detect_running:
                            if not locate_on_screen(trigger, CONFIDENCE, region):
                                break
                            time.sleep(INTERVAL)
                        break
                    except Exception:
                        pass
                if found:
                    break
        except Exception as e:
            app.log(f'에러: {e}')
        time.sleep(INTERVAL)


def repeat_loop():
    global repeat_running
    while repeat_running:
        for i, pos in enumerate(click_positions_repeat):
            if not repeat_running:
                break
            pyautogui.click(*pos)
            app.log(f'반복 {i+1}: ({pos[0]}, {pos[1]})')
            time.sleep(REPEAT_DELAY)


# ===== 테마 =====
THEME = {
    'bg': '#0d0d0d',
    'surface': '#161616',
    'card': '#1f1f1f',
    'border': '#2a2a2a',
    'accent': '#ffffff',
    'accent_dim': '#888888',
    'text': '#e0e0e0',
    'text_dim': '#666666',
    'green': '#4ade80',
    'red': '#f87171',
    'yellow': '#facc15',
}


# ===== GUI =====
class MacroApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Vertex Macro')
        self.root.geometry('700x850')
        self.root.minsize(680, 750)
        self.root.configure(bg=THEME['bg'])
        self.root.attributes('-topmost', always_on_top)
        self.capturing = False
        self._overlay = None
        self._selected_rule_idx = None
        self._region_highlight = None

        if os.path.exists(ICON_ICO):
            self.root.iconbitmap(ICON_ICO)

        self._setup_styles()
        self._build_ui()
        self.refresh_all()
        self.update_status()

        self.listener = keyboard.Listener(on_press=self.on_key)
        self.listener.daemon = True
        self.listener.start()

    def _setup_styles(self):
        t = THEME
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('TFrame', background=t['bg'])
        style.configure('Card.TFrame', background=t['card'])

        style.configure('TLabel', background=t['bg'], foreground=t['text'], font=('Segoe UI', 9))
        style.configure('H1.TLabel', background=t['bg'], foreground=t['accent'],
                         font=('Segoe UI', 13, 'bold'))
        style.configure('H2.TLabel', background=t['bg'], foreground=t['text'],
                         font=('Segoe UI', 10, 'bold'))
        style.configure('Dim.TLabel', background=t['bg'], foreground=t['text_dim'],
                         font=('Segoe UI', 8))
        style.configure('On.TLabel', background=t['bg'], foreground=t['green'],
                         font=('Segoe UI', 9, 'bold'))
        style.configure('Off.TLabel', background=t['bg'], foreground=t['red'],
                         font=('Segoe UI', 9, 'bold'))

        style.configure('TButton', font=('Segoe UI', 9), padding=[8, 4],
                         background=t['card'], foreground=t['text'],
                         bordercolor=t['border'], lightcolor=t['border'], darkcolor=t['border'])
        style.map('TButton',
                  background=[('active', t['border']), ('pressed', t['surface'])],
                  foreground=[('active', t['text']), ('pressed', t['text'])])

        style.configure('Accent.TButton', font=('Segoe UI', 9, 'bold'), padding=[10, 5],
                         background=t['accent'], foreground=t['bg'], bordercolor=t['accent'])
        style.map('Accent.TButton',
                  background=[('active', '#cccccc'), ('pressed', '#aaaaaa')],
                  foreground=[('active', t['bg']), ('pressed', t['bg'])])

        style.configure('Danger.TButton', font=('Segoe UI', 9), padding=[8, 4],
                         background=t['red'], foreground='white', bordercolor=t['red'])
        style.map('Danger.TButton',
                  background=[('active', '#dc2626'), ('pressed', '#b91c1c')],
                  foreground=[('active', 'white'), ('pressed', 'white')])

        style.configure('TNotebook', background=t['bg'], borderwidth=0, tabmargins=[0, 0, 0, 0])
        style.configure('TNotebook.Tab', background=t['surface'], foreground=t['text_dim'],
                         font=('Segoe UI', 9), padding=[8, 5], borderwidth=0)
        style.map('TNotebook.Tab',
                  background=[('selected', t['card'])], foreground=[('selected', t['accent'])])

        style.configure('TRadiobutton', background=t['bg'], foreground=t['text'], font=('Segoe UI', 9))
        style.map('TRadiobutton',
                  background=[('active', t['bg'])], foreground=[('active', t['text'])],
                  indicatorcolor=[('selected', t['accent']), ('!selected', t['text_dim'])])

        style.configure('TCheckbutton', background=t['bg'], foreground=t['text'], font=('Segoe UI', 9))
        style.map('TCheckbutton',
                  background=[('active', t['bg'])],
                  indicatorcolor=[('selected', t['accent']), ('!selected', t['text_dim'])])

        style.configure('TEntry', fieldbackground=t['surface'], foreground=t['text'],
                         bordercolor=t['border'], insertcolor=t['text'])
        style.configure('Vertical.TScrollbar', background=t['surface'],
                         troughcolor=t['bg'], bordercolor=t['bg'], arrowcolor=t['text_dim'])

    def _make_listbox(self, parent, height=5):
        t = THEME
        lb = tk.Listbox(parent, height=height, bg=t['surface'], fg=t['text'],
                          selectbackground='#444444', selectforeground='#ffffff',
                          font=('Consolas', 9), bd=0, highlightthickness=1,
                          highlightcolor=t['border'], highlightbackground=t['surface'],
                          activestyle='none', relief='flat')
        lb._hover_idx = None
        def on_motion(e):
            idx = lb.nearest(e.y)
            if idx < 0 or lb.size() == 0 or idx == lb._hover_idx:
                return
            if lb._hover_idx is not None and lb._hover_idx < lb.size() and lb._hover_idx not in lb.curselection():
                lb.itemconfig(lb._hover_idx, bg=t['surface'], fg=t['text'])
            lb._hover_idx = idx
            if idx not in lb.curselection():
                lb.itemconfig(idx, bg='#2a2a2a', fg='#ffffff')
        def on_leave(e):
            if lb._hover_idx is not None and lb._hover_idx < lb.size() and lb._hover_idx not in lb.curselection():
                lb.itemconfig(lb._hover_idx, bg=t['surface'], fg=t['text'])
            lb._hover_idx = None
        lb.bind('<Motion>', on_motion)
        lb.bind('<Leave>', on_leave)
        return lb

    def _build_ui(self):
        t = THEME

        # 헤더
        header = ttk.Frame(self.root)
        header.pack(fill='x', padx=16, pady=(12, 6))
        title_f = ttk.Frame(header)
        title_f.pack(side='left')
        if os.path.exists(ICON_PNG):
            logo = Image.open(ICON_PNG).resize((28, 28), Image.LANCZOS)
            self._logo_photo = ImageTk.PhotoImage(logo)
            tk.Label(title_f, image=self._logo_photo, bg=t['bg']).pack(side='left', padx=(0, 8))
        ttk.Label(title_f, text='Vertex Macro', style='H1.TLabel').pack(side='left')

        status_f = ttk.Frame(header)
        status_f.pack(side='right')
        self.detect_status = ttk.Label(status_f, text='감지 OFF', style='Off.TLabel')
        self.detect_status.pack(side='left', padx=(0, 12))
        self.repeat_status = ttk.Label(status_f, text='반복 OFF', style='Off.TLabel')
        self.repeat_status.pack(side='left')

        tk.Frame(self.root, height=1, bg=t['border']).pack(fill='x', padx=16)

        # 탭
        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=16, pady=(8, 4))
        self._build_tab_rules(nb)
        self._build_tab_repeat(nb)
        self._build_tab_region(nb)
        self._build_tab_settings(nb)

        # 로그
        log_f = ttk.Frame(self.root)
        log_f.pack(fill='x', padx=16, pady=(2, 4))
        log_top = ttk.Frame(log_f)
        log_top.pack(fill='x')
        ttk.Label(log_top, text='로그', style='H2.TLabel').pack(side='left')
        ttk.Button(log_top, text='지우기', command=self.clear_log).pack(side='right')
        self.log_text = tk.Text(log_f, height=5, bg=t['surface'], fg=t['green'],
                                 font=('Consolas', 8), state='disabled', bd=0,
                                 highlightthickness=1, highlightcolor=t['border'],
                                 highlightbackground=t['surface'], padx=6, pady=4)
        self.log_text.pack(fill='x', pady=(4, 0))

        # 하단
        bottom = ttk.Frame(self.root)
        bottom.pack(fill='x', padx=16, pady=(2, 8))
        self.hotkey_label = ttk.Label(bottom, text='', style='Dim.TLabel')
        self.hotkey_label.pack(side='left')
        self._update_hotkey_label()
        self.mode_label = ttk.Label(bottom, text='', style='Dim.TLabel')
        self.mode_label.pack(side='right')

    # ── 감지 규칙 탭 ──
    def _build_tab_rules(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text=' 감지 ')

        top = ttk.Frame(tab)
        top.pack(fill='x', padx=8, pady=(8, 4))
        self.detect_btn = ttk.Button(top, text='▶ 시작', style='Accent.TButton',
                                      command=self.toggle_detect)
        self.detect_btn.pack(side='left', padx=(0, 8))
        ttk.Button(top, text='캡처 추가', command=self.capture_trigger).pack(side='left', padx=2)
        ttk.Button(top, text='파일 추가', command=self.add_trigger_file).pack(side='left', padx=2)
        ttk.Button(top, text='삭제', command=self.del_rule).pack(side='left', padx=2)

        self.rule_list = self._make_listbox(tab, 4)
        self.rule_list.pack(fill='x', padx=8, pady=4)
        self.rule_list.bind('<<ListboxSelect>>', self._on_rule_select)

        sep_f = ttk.Frame(tab)
        sep_f.pack(fill='x', padx=8, pady=(4, 0))
        tk.Frame(sep_f, height=1, bg=THEME['border']).pack(fill='x', side='top')
        self.selected_rule_name = ttk.Label(sep_f, text='이미지를 선택하세요', style='Dim.TLabel')
        self.selected_rule_name.pack(anchor='w', pady=(4, 0))

        self.rule_detail = ttk.Frame(tab)
        self.rule_detail.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        self.rule_config = ttk.Frame(self.rule_detail)

        action_f = ttk.Frame(self.rule_config)
        action_f.pack(fill='x', pady=4)
        ttk.Label(action_f, text='감지 시 동작', style='H2.TLabel').pack(anchor='w')
        radio_f = ttk.Frame(action_f)
        radio_f.pack(fill='x', pady=(2, 0))
        self.rule_action_var = tk.StringVar(value='center')
        for txt, val in [('감지된 이미지 중앙 클릭', 'center'),
                         ('지정 좌표 클릭', 'coord'),
                         ('다른 이미지 찾아서 클릭', 'image')]:
            ttk.Radiobutton(radio_f, text=txt, variable=self.rule_action_var,
                             value=val, command=self._on_rule_action_change).pack(anchor='w', pady=1)

        # 좌표 패널
        self.rule_coord_panel = ttk.Frame(self.rule_config)
        rc_btn = ttk.Frame(self.rule_coord_panel)
        rc_btn.pack(fill='x', pady=2)
        ttk.Button(rc_btn, text='+ 마우스(F1)', command=self._rule_add_coord_mouse).pack(side='left', padx=(0, 4))
        ttk.Button(rc_btn, text='+ 직접입력', command=self._rule_add_coord_manual).pack(side='left', padx=2)
        ttk.Button(rc_btn, text='삭제', command=self._rule_del_coord).pack(side='left', padx=2)
        self.rule_coord_list = self._make_listbox(self.rule_coord_panel, 3)
        self.rule_coord_list.pack(fill='both', expand=True)

        # 이미지 클릭 패널
        self.rule_img_panel = ttk.Frame(self.rule_config)
        ri_btn = ttk.Frame(self.rule_img_panel)
        ri_btn.pack(fill='x', pady=2)
        ttk.Button(ri_btn, text='캡처', command=self._rule_capture_click_img).pack(side='left', padx=(0, 4))
        ttk.Button(ri_btn, text='파일 추가', command=self._rule_add_click_img_file).pack(side='left', padx=2)
        ttk.Button(ri_btn, text='삭제', command=self._rule_del_click_img).pack(side='left', padx=2)
        self.rule_img_list = self._make_listbox(self.rule_img_panel, 3)
        self.rule_img_list.pack(fill='both', expand=True)

    def _build_tab_repeat(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text=' 반복 ')
        ttk.Label(tab, text='순서대로 반복 클릭', style='H2.TLabel').pack(anchor='w', padx=8, pady=(10, 4))
        sf = ttk.Frame(tab)
        sf.pack(fill='x', padx=8, pady=4)
        self.repeat_btn = ttk.Button(sf, text='▶ 시작', style='Accent.TButton', command=self.toggle_repeat)
        self.repeat_btn.pack(side='left')
        btn = ttk.Frame(tab)
        btn.pack(fill='x', padx=8, pady=(6, 2))
        ttk.Button(btn, text='+ 마우스', command=self.add_pos_mouse).pack(side='left', padx=(0, 4))
        ttk.Button(btn, text='+ 직접입력', command=self.add_pos_manual).pack(side='left', padx=2)
        ttk.Button(btn, text='삭제', command=self.del_repeat_pos).pack(side='left', padx=2)
        ttk.Button(btn, text='초기화', command=self.clear_repeat_pos).pack(side='left', padx=2)
        self.repeat_list = self._make_listbox(tab, 6)
        self.repeat_list.pack(fill='both', expand=True, padx=8, pady=(2, 8))

    def _build_tab_region(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text=' 영역 ')
        ttk.Label(tab, text='스캔 범위 제한', style='H2.TLabel').pack(anchor='w', padx=8, pady=(10, 2))
        ttk.Label(tab, text='비어있으면 전체 화면', style='Dim.TLabel').pack(anchor='w', padx=8, pady=(0, 4))
        btn = ttk.Frame(tab)
        btn.pack(fill='x', padx=8, pady=4)
        ttk.Button(btn, text='화면 선택', style='Accent.TButton', command=self.start_region).pack(side='left', padx=(0, 4))
        ttk.Button(btn, text='+ 직접입력', command=self.add_region_manual).pack(side='left', padx=2)
        ttk.Button(btn, text='삭제', command=self.del_region).pack(side='left', padx=2)
        ttk.Button(btn, text='초기화', command=self.clear_regions).pack(side='left', padx=2)
        self.region_list = self._make_listbox(tab, 6)
        self.region_list.pack(fill='both', expand=True, padx=8, pady=(2, 8))
        self.region_list.bind('<<ListboxSelect>>', self._on_region_select)

    def _on_region_select(self, event=None):
        sel = self.region_list.curselection()
        if not sel or sel[0] >= len(scan_regions):
            return
        r = scan_regions[sel[0]]
        self._show_region_highlight(r[0], r[1], r[2], r[3])

    def _show_region_highlight(self, x, y, w, h):
        if self._region_highlight:
            try:
                self._region_highlight.destroy()
            except Exception:
                pass
        ov = tk.Toplevel()
        ov.overrideredirect(True)
        ov.attributes('-topmost', True)
        ov.attributes('-alpha', 0.35)
        ov.geometry(f'{w}x{h}+{x}+{y}')
        ov.configure(bg='#4ade80')
        c = tk.Canvas(ov, highlightthickness=0, bg='#4ade80')
        c.pack(fill='both', expand=True)
        c.create_rectangle(2, 2, w - 2, h - 2, outline='#ffffff', width=2, fill='')
        c.create_text(w // 2, h // 2, text=f'{w} x {h}', fill='#ffffff',
                       font=('Segoe UI', 14, 'bold'))
        self._region_highlight = ov
        ov.after(1500, ov.destroy)

    def _build_tab_settings(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text=' 설정 ')
        canvas = tk.Canvas(tab, bg=THEME['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient='vertical', command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        canvas.bind_all('<MouseWheel>', lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units'))
        px = 10

        ttk.Label(inner, text='타이밍', style='H2.TLabel').grid(row=0, column=0, columnspan=2, sticky='w', padx=px, pady=(10, 6))
        fields = [('스캔 간격 (초)', 'interval_var', str(INTERVAL)),
                  ('반복 간격 (초)', 'repeat_delay_var', str(REPEAT_DELAY)),
                  ('감지 후 대기 (초)', 'cooldown_var', str(DETECT_COOLDOWN)),
                  ('정확도 (0~1)', 'confidence_var', str(CONFIDENCE))]
        for i, (label, var_name, default) in enumerate(fields):
            ttk.Label(inner, text=label).grid(row=1+i, column=0, sticky='w', padx=px, pady=3)
            var = tk.StringVar(value=default)
            setattr(self, var_name, var)
            ttk.Entry(inner, textvariable=var, width=8).grid(row=1+i, column=1, sticky='w', padx=px, pady=3)
        row = len(fields) + 1

        ttk.Label(inner, text='단축키', style='H2.TLabel').grid(row=row, column=0, columnspan=2, sticky='w', padx=px, pady=(14, 6))
        row += 1
        hk_labels = {'add_pos': '좌표 추가', 'toggle_detect': '감지 시작/중지',
                     'toggle_repeat': '반복 시작/중지', 'switch_mode': '입력 대상 전환',
                     'stop_all': '전체 중지', 'exit': '종료'}
        self.hotkey_vars = {}
        for k, label in hk_labels.items():
            ttk.Label(inner, text=label).grid(row=row, column=0, sticky='w', padx=px, pady=3)
            var = tk.StringVar(value=hotkeys.get(k, ''))
            self.hotkey_vars[k] = var
            ttk.Combobox(inner, textvariable=var, values=AVAILABLE_KEYS, width=10,
                          state='readonly').grid(row=row, column=1, sticky='w', padx=px, pady=3)
            row += 1

        ttk.Label(inner, text='기타', style='H2.TLabel').grid(row=row, column=0, columnspan=2, sticky='w', padx=px, pady=(14, 6))
        row += 1
        self.topmost_var = tk.BooleanVar(value=always_on_top)
        ttk.Checkbutton(inner, text='항상 위에', variable=self.topmost_var).grid(row=row, column=0, columnspan=2, sticky='w', padx=px, pady=3)
        row += 1
        bf = ttk.Frame(inner)
        bf.grid(row=row, column=0, columnspan=2, sticky='w', padx=px, pady=(10, 10))
        ttk.Button(bf, text='설정 적용', style='Accent.TButton', command=self.apply_settings).pack(side='left', padx=(0, 8))
        ttk.Button(bf, text='설정 초기화', style='Danger.TButton', command=self.reset_settings).pack(side='left')

    # === 로그 ===
    def log(self, msg):
        def _log():
            self.log_text.config(state='normal')
            self.log_text.insert('end', f'[{time.strftime("%H:%M:%S")}] {msg}\n')
            self.log_text.see('end')
            self.log_text.config(state='disabled')
        self.root.after(0, _log)

    def clear_log(self):
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.config(state='disabled')

    # === 상태 ===
    def refresh_all(self):
        self.rule_list.delete(0, 'end')
        for i, r in enumerate(detect_rules):
            self.rule_list.insert('end', f"  {i+1}.  {r['trigger']}")
        if self._selected_rule_idx is not None and self._selected_rule_idx < len(detect_rules):
            self.rule_list.selection_set(self._selected_rule_idx)
            self.rule_list.itemconfig(self._selected_rule_idx, bg='#444444', fg='#ffffff')

        self.repeat_list.delete(0, 'end')
        for i, p in enumerate(click_positions_repeat):
            self.repeat_list.insert('end', f'  {i+1}.  ({p[0]}, {p[1]})')

        self.region_list.delete(0, 'end')
        for i, r in enumerate(scan_regions):
            self.region_list.insert('end', f'  {i+1}.  X={r[0]}  Y={r[1]}  W={r[2]}  H={r[3]}')

        self._refresh_rule_detail()

    def _refresh_rule_detail(self):
        if self._selected_rule_idx is not None and self._selected_rule_idx < len(detect_rules):
            rule = detect_rules[self._selected_rule_idx]
            self.rule_coord_list.delete(0, 'end')
            for i, c in enumerate(rule.get('coords', [])):
                self.rule_coord_list.insert('end', f'  {i+1}.  ({c[0]}, {c[1]})')
            self.rule_img_list.delete(0, 'end')
            for i, f in enumerate(rule.get('click_targets', [])):
                self.rule_img_list.insert('end', f'  {i+1}.  {f}')

    def update_status(self):
        if detect_running:
            self.detect_status.config(text='감지 ON', style='On.TLabel')
            self.detect_btn.config(text='■ 중지')
        else:
            self.detect_status.config(text='감지 OFF', style='Off.TLabel')
            self.detect_btn.config(text='▶ 시작')
        if repeat_running:
            self.repeat_status.config(text='반복 ON', style='On.TLabel')
            self.repeat_btn.config(text='■ 중지')
        else:
            self.repeat_status.config(text='반복 OFF', style='Off.TLabel')
            self.repeat_btn.config(text='▶ 시작')
        self.mode_label.config(text=f'F1 대상: {input_mode}')

    def _update_hotkey_label(self):
        h = hotkeys
        parts = [f"{h[k].upper()}:{v}" for k, v in
                 [('add_pos','좌표'),('toggle_detect','감지'),('toggle_repeat','반복'),
                  ('switch_mode','전환'),('stop_all','중지'),('exit','종료')]]
        self.hotkey_label.config(text='  '.join(parts))

    # === 규칙 선택 ===
    def _on_rule_select(self, event=None):
        sel = self.rule_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(detect_rules):
            return
        if self._selected_rule_idx is not None and self._selected_rule_idx < self.rule_list.size():
            self.rule_list.itemconfig(self._selected_rule_idx,
                                       bg=THEME['surface'], fg=THEME['text'])
        self._selected_rule_idx = idx
        rule = detect_rules[idx]
        self.rule_list.itemconfig(idx, bg='#444444', fg='#ffffff')
        self.selected_rule_name.config(text=f'▸ {rule["trigger"]}', style='On.TLabel')
        self.rule_config.pack(fill='both', expand=True)
        self.rule_action_var.set(rule.get('action', 'center'))
        self._on_rule_action_change()
        self._refresh_rule_detail()

    def _on_rule_action_change(self):
        if self._selected_rule_idx is None:
            return
        action = self.rule_action_var.get()
        detect_rules[self._selected_rule_idx]['action'] = action
        save_config()
        self.refresh_all()
        self.rule_coord_panel.pack_forget()
        self.rule_img_panel.pack_forget()
        if action == 'coord':
            self.rule_coord_panel.pack(fill='both', expand=True, pady=(4, 0))
        elif action == 'image':
            self.rule_img_panel.pack(fill='both', expand=True, pady=(4, 0))

    # === 규칙 이미지 관리 ===
    def _import_image(self, path):
        name = os.path.basename(path)
        norm = os.path.normpath(path)
        if norm.startswith(os.path.normpath(IMAGES_DIR) + os.sep):
            return name
        safe_name = safe_filename(name)
        dest = img_path(safe_name)
        if os.path.exists(dest):
            base, ext = os.path.splitext(safe_name)
            c = 1
            while os.path.exists(img_path(f'{base}_{c}{ext}')):
                c += 1
            safe_name = f'{base}_{c}{ext}'
            dest = img_path(safe_name)
        shutil.copy2(path, dest)
        return safe_name

    def capture_trigger(self):
        def on_done(screenshot, x, y, w, h):
            cropped = screenshot.crop((x, y, x + w, y + h))
            ts = time.strftime('%Y%m%d_%H%M%S')
            filename = f'capture_{ts}.png'
            cropped.save(img_path(filename))
            detect_rules.append(new_rule(filename))
            save_config()
            self.refresh_all()
            self.log(f'감지 이미지 캡처: {filename}')
        self._open_overlay('감지할 이미지 캡처\nESC: 취소', '#ffffff', on_done)

    def add_trigger_file(self):
        paths = filedialog.askopenfilenames(title='감지 이미지', initialdir=IMAGES_DIR,
                                             filetypes=[('이미지', '*.png *.jpg *.bmp')])
        added = 0
        for p in paths:
            fname = self._import_image(p)
            if not any(r['trigger'] == fname for r in detect_rules):
                detect_rules.append(new_rule(fname))
                added += 1
        if added:
            save_config()
            self.refresh_all()
            self.log(f'감지 이미지 {added}개 추가')

    def del_rule(self):
        sel = self.rule_list.curselection()
        if sel:
            detect_rules.pop(sel[0])
            self._selected_rule_idx = None
            self.rule_config.pack_forget()
            self.selected_rule_name.config(text='이미지를 선택하세요', style='Dim.TLabel')
            save_config()
            self.refresh_all()

    # === 규칙별 좌표 ===
    def _get_selected_rule(self):
        if self._selected_rule_idx is not None and self._selected_rule_idx < len(detect_rules):
            return detect_rules[self._selected_rule_idx]
        return None

    def _rule_add_coord_mouse(self):
        rule = self._get_selected_rule()
        if not rule:
            return
        x, y = pyautogui.position()
        rule['coords'].append((x, y))
        save_config()
        self._refresh_rule_detail()
        self.log(f'{rule["trigger"]} 좌표: ({x}, {y})')

    def _rule_add_coord_manual(self):
        rule = self._get_selected_rule()
        if not rule:
            return
        self._show_coord_dialog(lambda x, y: self._rule_add_coord_cb(rule, x, y))

    def _rule_add_coord_cb(self, rule, x, y):
        rule['coords'].append((x, y))
        save_config()
        self._refresh_rule_detail()

    def _rule_del_coord(self):
        rule = self._get_selected_rule()
        if not rule:
            return
        sel = self.rule_coord_list.curselection()
        if sel:
            rule['coords'].pop(sel[0])
            save_config()
            self._refresh_rule_detail()

    # === 규칙별 클릭 이미지 ===
    def _rule_capture_click_img(self):
        rule = self._get_selected_rule()
        if not rule:
            return
        def on_done(screenshot, x, y, w, h):
            cropped = screenshot.crop((x, y, x + w, y + h))
            ts = time.strftime('%Y%m%d_%H%M%S')
            filename = f'click_{ts}.png'
            cropped.save(img_path(filename))
            rule['click_targets'].append(filename)
            save_config()
            self._refresh_rule_detail()
            self.log(f'클릭 이미지 캡처: {filename}')
        self._open_overlay('클릭할 이미지 캡처\nESC: 취소', THEME['yellow'], on_done)

    def _rule_add_click_img_file(self):
        rule = self._get_selected_rule()
        if not rule:
            return
        paths = filedialog.askopenfilenames(title='클릭 이미지', initialdir=IMAGES_DIR,
                                             filetypes=[('이미지', '*.png *.jpg *.bmp')])
        for p in paths:
            fname = self._import_image(p)
            if fname not in rule['click_targets']:
                rule['click_targets'].append(fname)
        save_config()
        self._refresh_rule_detail()

    def _rule_del_click_img(self):
        rule = self._get_selected_rule()
        if not rule:
            return
        sel = self.rule_img_list.curselection()
        if sel:
            rule['click_targets'].pop(sel[0])
            save_config()
            self._refresh_rule_detail()

    # === 오버레이 ===
    def _open_overlay(self, guide_text, outline_color, on_done):
        self.capturing = True
        self.root.withdraw()
        time.sleep(0.3)
        screenshot = pyautogui.screenshot()
        overlay = tk.Toplevel()
        overlay.attributes('-fullscreen', True)
        overlay.attributes('-topmost', True)
        overlay.attributes('-alpha', 0.4)
        overlay.overrideredirect(True)
        self._overlay = overlay
        sw, sh = overlay.winfo_screenwidth(), overlay.winfo_screenheight()
        canvas = tk.Canvas(overlay, highlightthickness=0, cursor='crosshair', bg='black')
        canvas.pack(fill='both', expand=True)
        canvas.create_text(sw // 2, sh // 2, text=guide_text, fill='white',
                           font=('Segoe UI', 18, 'bold'), justify='center')
        start = [0, 0]
        rect_id = [None]
        def on_press(e):
            start[0], start[1] = e.x, e.y
            rect_id[0] = canvas.create_rectangle(start[0], start[1], start[0], start[1],
                                                   outline=outline_color, width=2)
        def on_drag(e):
            if rect_id[0]:
                canvas.coords(rect_id[0], start[0], start[1], e.x, e.y)
        def on_release(e):
            x1, y1 = min(start[0], e.x), min(start[1], e.y)
            x2, y2 = max(start[0], e.x), max(start[1], e.y)
            overlay.destroy()
            self._overlay = None
            self.capturing = False
            self.root.deiconify()
            if x2 - x1 >= 5 and y2 - y1 >= 5:
                on_done(screenshot, x1, y1, x2 - x1, y2 - y1)
        def on_esc(e):
            self._cancel_overlay()
        canvas.bind('<ButtonPress-1>', on_press)
        canvas.bind('<B1-Motion>', on_drag)
        canvas.bind('<ButtonRelease-1>', on_release)
        overlay.bind('<Escape>', on_esc)

    def _cancel_overlay(self):
        if self._overlay:
            self._overlay.destroy()
            self._overlay = None
        self.capturing = False
        self.root.deiconify()

    # === 공통 좌표 입력 다이얼로그 ===
    def _show_coord_dialog(self, callback):
        dialog = tk.Toplevel(self.root)
        dialog.title('좌표 입력')
        dialog.geometry('260x60')
        dialog.configure(bg=THEME['bg'])
        dialog.attributes('-topmost', True)
        dialog.resizable(False, False)
        f = ttk.Frame(dialog)
        f.pack(fill='both', expand=True, padx=8, pady=8)
        ttk.Label(f, text='X:').pack(side='left')
        xe = ttk.Entry(f, width=7); xe.pack(side='left', padx=(2, 6)); xe.focus()
        ttk.Label(f, text='Y:').pack(side='left')
        ye = ttk.Entry(f, width=7); ye.pack(side='left', padx=2)
        def ok(event=None):
            try:
                x, y = int(xe.get()), int(ye.get())
                callback(x, y)
                dialog.destroy()
            except ValueError:
                messagebox.showerror('오류', '숫자를 입력하세요', parent=dialog)
        ttk.Button(f, text='확인', command=ok).pack(side='left', padx=8)
        dialog.bind('<Return>', ok)

    # === 반복 좌표 ===
    def add_pos_mouse(self):
        x, y = pyautogui.position()
        click_positions_repeat.append((x, y))
        save_config()
        self.refresh_all()
        self.log(f'반복 좌표: ({x}, {y})')

    def add_pos_manual(self):
        def cb(x, y):
            click_positions_repeat.append((x, y))
            save_config()
            self.refresh_all()
            self.log(f'반복 좌표: ({x}, {y})')
        self._show_coord_dialog(cb)

    def del_repeat_pos(self):
        sel = self.repeat_list.curselection()
        if sel:
            click_positions_repeat.pop(sel[0])
            save_config()
            self.refresh_all()

    def clear_repeat_pos(self):
        click_positions_repeat.clear()
        save_config()
        self.refresh_all()
        self.log('반복 좌표 초기화')

    # === 영역 ===
    def start_region(self):
        def on_done(_ss, x, y, w, h):
            scan_regions.append((x, y, w, h))
            save_config()
            self.refresh_all()
            self.log(f'영역: X={x} Y={y} W={w} H={h}')
        self._open_overlay('스캔 영역 선택\nESC: 취소', THEME['green'], on_done)

    def add_region_manual(self):
        dialog = tk.Toplevel(self.root)
        dialog.title('영역 입력')
        dialog.geometry('340x60')
        dialog.configure(bg=THEME['bg'])
        dialog.attributes('-topmost', True)
        dialog.resizable(False, False)
        f = ttk.Frame(dialog)
        f.pack(fill='both', expand=True, padx=8, pady=8)
        entries = []
        for lb in ['X:', 'Y:', 'W:', 'H:']:
            ttk.Label(f, text=lb).pack(side='left')
            e = ttk.Entry(f, width=5); e.pack(side='left', padx=2); entries.append(e)
        entries[0].focus()
        def ok(event=None):
            try:
                vals = tuple(int(e.get()) for e in entries)
                scan_regions.append(vals)
                save_config()
                self.refresh_all()
                dialog.destroy()
            except ValueError:
                messagebox.showerror('오류', '숫자를 입력하세요', parent=dialog)
        ttk.Button(f, text='확인', command=ok).pack(side='left', padx=6)
        dialog.bind('<Return>', ok)

    def del_region(self):
        sel = self.region_list.curselection()
        if sel:
            scan_regions.pop(sel[0])
            save_config()
            self.refresh_all()

    def clear_regions(self):
        scan_regions.clear()
        save_config()
        self.refresh_all()
        self.log('영역 초기화')

    # === 설정 ===
    def apply_settings(self):
        global INTERVAL, REPEAT_DELAY, CONFIDENCE, DETECT_COOLDOWN, hotkeys, always_on_top
        try:
            INTERVAL = float(self.interval_var.get())
            REPEAT_DELAY = float(self.repeat_delay_var.get())
            DETECT_COOLDOWN = float(self.cooldown_var.get())
            CONFIDENCE = float(self.confidence_var.get())
        except ValueError:
            messagebox.showerror('오류', '숫자를 입력하세요')
            return
        new_hk = {}
        for k, var in self.hotkey_vars.items():
            v = var.get()
            if v in new_hk.values():
                messagebox.showerror('오류', f'단축키 중복: {v.upper()}')
                return
            new_hk[k] = v
        hotkeys = new_hk
        always_on_top = self.topmost_var.get()
        self.root.attributes('-topmost', always_on_top)
        save_config()
        self._update_hotkey_label()
        self.log('설정 적용')

    def reset_settings(self):
        if not messagebox.askyesno('확인', '설정 초기화?'):
            return
        global hotkeys, always_on_top, INTERVAL, REPEAT_DELAY, DETECT_COOLDOWN, CONFIDENCE
        INTERVAL, REPEAT_DELAY, DETECT_COOLDOWN, CONFIDENCE = 0.03, 1.0, 0.5, 0.8
        hotkeys = dict(DEFAULT_HOTKEYS)
        always_on_top = True
        self.interval_var.set('0.03'); self.repeat_delay_var.set('1.0')
        self.cooldown_var.set('0.5'); self.confidence_var.set('0.8')
        for k, v in hotkeys.items():
            if k in self.hotkey_vars:
                self.hotkey_vars[k].set(v)
        self.topmost_var.set(True)
        self.root.attributes('-topmost', True)
        save_config()
        self._update_hotkey_label()
        self.log('설정 초기화')

    # === 시작/중지 ===
    def toggle_detect(self):
        global detect_running
        if not detect_running:
            if not detect_rules:
                messagebox.showwarning('경고', '감지 이미지를 먼저 추가하세요')
                return
            detect_running = True
            threading.Thread(target=detect_loop, daemon=True).start()
            self.log('감지 시작')
        else:
            detect_running = False
            self.log('감지 중지')
        self.update_status()

    def toggle_repeat(self):
        global repeat_running
        if not repeat_running:
            if not click_positions_repeat:
                messagebox.showwarning('경고', '반복 좌표를 먼저 추가하세요')
                return
            repeat_running = True
            threading.Thread(target=repeat_loop, daemon=True).start()
            self.log('반복 시작')
        else:
            repeat_running = False
            self.log('반복 중지')
        self.update_status()

    # === 키보드 ===
    def on_key(self, key):
        global input_mode, detect_running, repeat_running
        matched = None
        for action, key_name in hotkeys.items():
            if key == get_key_obj(key_name):
                matched = action
                break
        if matched is None:
            return
        try:
            if matched == 'add_pos':
                if input_mode == 'detect':
                    rule = self._get_selected_rule()
                    if rule and rule['action'] == 'coord':
                        self.root.after(0, self._rule_add_coord_mouse)
                    else:
                        self.log('감지 탭에서 좌표 클릭 모드 규칙을 선택하세요')
                else:
                    self.root.after(0, self.add_pos_mouse)
            elif matched == 'toggle_detect':
                self.root.after(0, self.toggle_detect)
            elif matched == 'toggle_repeat':
                self.root.after(0, self.toggle_repeat)
            elif matched == 'switch_mode':
                input_mode = 'repeat' if input_mode == 'detect' else 'detect'
                self.root.after(0, self.update_status)
                self.log(f'입력 대상: {input_mode}')
            elif matched == 'stop_all':
                detect_running = False
                repeat_running = False
                self.root.after(0, self.update_status)
                self.log('전체 중지')
            elif matched == 'exit':
                if self.capturing:
                    self.root.after(0, self._cancel_overlay)
                    return
                detect_running = False
                repeat_running = False
                self.root.after(0, self.root.destroy)
        except Exception as e:
            self.log(f'에러: {e}')


if __name__ == '__main__':
    load_config()
    root = tk.Tk()
    app = MacroApp(root)
    root.mainloop()
