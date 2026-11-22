"""
Chess screen overlay: reads the board on your screen, asks 3 UCI engines for the best move,
draws arrows on top, and flags "brilliant candidates" (sound piece sacrifices).

SETUP
  pip install mss opencv-python numpy python-chess
  Install 3 UCI engines (see README/engines.json).
  Windows is the supported platform (click-through overlay + hidden-from-capture window).

USE
  1. Put the STARTING position on screen (e.g. a new game vs a bot, or an analysis board).
  2. python chess_overlay.py --color white      (or --color black if your side is at the bottom)
  3. Drag a box around the board, press ENTER. Templates are learned from the start position.
  4. Play. Arrows: green=Stockfish, blue=Lc0, orange=3rd engine, thick=2+ engines agree,
     gold=brilliant candidate. Esc closes it.

Use it for bots, analysis boards and reviewing your own games. Using engine help in live games
against humans is cheating and gets accounts banned.
"""
import argparse, ctypes, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import cv2, mss, numpy as np
import chess, chess.engine
import tkinter as tk

# ---- engines are configured in engines.json next to the exe (created on first run) ----
import os, json
BASE = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
DEFAULT_CFG = {
    "Stockfish": {"cmd": ["engines/stockfish.exe"], "options": {"Threads": 2}},
    "Lc0":       {"cmd": ["engines/lc0.exe", "--weights=engines/network.pb.gz"], "options": {}},
    "Berserk":   {"cmd": ["engines/berserk.exe"], "options": {}},
}
def load_engines():
    p = os.path.join(BASE, "engines.json")
    if not os.path.exists(p):
        json.dump(DEFAULT_CFG, open(p, "w"), indent=2)
        input(f"Created {p}. Drop your 3 engine .exe files in the 'engines' folder, edit paths if needed, then rerun. Press Enter.")
        sys.exit(0)
    cfg = json.load(open(p))
    out = {}
    for n, c in cfg.items():
        cmd = [os.path.join(BASE, c["cmd"][0])] + [
            a.replace("=engines", "=" + os.path.join(BASE, "engines")) for a in c["cmd"][1:]]
        out[n] = (cmd, c.get("options", {}))
    return out
ENGINES = load_engines()
_PAL = ["#2ecc40", "#3b9dff", "#ff8c1a"]
COLORS = {n: _PAL[i % 3] for i, n in enumerate(ENGINES)}
THINK = 0.4          # seconds per engine per position
BRILLIANT_TOL = 25   # centipawns worse than best still allowed
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
S = 32

# ---------------- screen reading ----------------
def cell(sq, flip):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return (7 - f, r) if flip else (f, 7 - r)          # (col, row)

def feat(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    g = g[int(h*.08):int(h*.92), int(w*.08):int(w*.92)]
    g = cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32)
    corners = np.concatenate([g[:4, :4].ravel(), g[:4, -4:].ravel(), g[-4:, :4].ravel(), g[-4:, -4:].ravel()])
    bg = np.median(corners)
    mask = np.abs(g - bg) > 20
    return np.where(mask, g, 128.0), mask.mean()

def squares(frame, flip):
    h, w = frame.shape[:2]
    ch, cw = h / 8, w / 8
    out = {}
    for sq in chess.SQUARES:
        c, r = cell(sq, flip)
        out[sq] = feat(frame[int(r*ch):int((r+1)*ch), int(c*cw):int((c+1)*cw)])
    return out

def learn(frame, flip):
    start = chess.Board()
    t = []
    for sq, f in squares(frame, flip).items():
        p = start.piece_at(sq)
        if p: t.append((p.symbol(), f[0]))
    return t

def read(frame, flip, templates):
    pl = {}
    for sq, (f, cov) in squares(frame, flip).items():
        if cov < 0.07: continue
        best = min(templates, key=lambda t: np.abs(t[1] - f).mean())
        pl[sq] = best[0]
    return pl

# ---------------- board tracking ----------------
def placement(board): return {s: p.symbol() for s, p in board.piece_map().items()}

def track(board, pl, my_color):
    if placement(board) == pl: return board
    for m in list(board.legal_moves):
        board.push(m)
        if placement(board) == pl: return board
        board.pop()
    b = chess.Board(None)
    for s, sym in pl.items(): b.set_piece_at(s, chess.Piece.from_symbol(sym))
    b.turn = my_color
    b.set_castling_fen("KQkq")
    return b

# ---------------- analysis ----------------
def mat(b, color):
    return sum(VAL[p.piece_type] for p in b.piece_map().values() if p.color == color) - \
           sum(VAL[p.piece_type] for p in b.piece_map().values() if p.color != color)

def brilliant(board, infos):
    side = board.turn
    sc = lambda i: i["score"].pov(side).score(mate_score=10000)
    best = sc(infos[0])
    if best > 600 or best < -100: return []          # already winning big / already lost: no brilliants
    base, out = mat(board, side), []
    for i in infos:
        pv = i.get("pv", [])
        if not pv or best - sc(i) > BRILLIANT_TOL or sc(i) < -50: continue
        n = min(len(pv), 7); n -= (n % 2 == 0)
        if n < 1: continue
        b = board.copy()
        for m in pv[:n]: b.push(m)
        if mat(b, side) - base <= -2: out.append(pv[0])
    return out

def run_engine(name, eng, board):
    multi = 6 if name == list(ENGINES)[0] else 1
    infos = eng.analyse(board, chess.engine.Limit(time=THINK), multipv=multi)
    if isinstance(infos, dict): infos = [infos]
    return name, infos

# ---------------- overlay ----------------
class Overlay:
    def __init__(self, roi):
        self.roi = roi
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True, "-transparentcolor", "#010101")
        self.root.geometry(f"{roi['width']}x{roi['height']}+{roi['left']}+{roi['top']}")
        self.cv = tk.Canvas(self.root, bg="#010101", highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.panel = tk.Toplevel(self.root)
        self.panel.overrideredirect(True); self.panel.attributes("-topmost", True)
        self.panel.geometry(f"+{roi['left']+roi['width']+10}+{roi['top']}")
        self.label = tk.Label(self.panel, text="", justify="left", bg="#111", fg="#eee",
                              font=("Consolas", 11), padx=8, pady=6)
        self.label.pack()
        self.root.bind("<Escape>", lambda e: sys.exit(0))
        self.root.update()
        try:   # click-through + invisible to screen capture (Windows)
            for w in (self.root, self.panel):
                hwnd = ctypes.windll.user32.GetParent(w.winfo_id())
                st = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
                ctypes.windll.user32.SetWindowLongW(hwnd, -20, st | 0x80000 | 0x20)
                ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x11)
        except Exception as e:
            print("Overlay capture-hiding unavailable (arrows may confuse reading):", e)

    def draw(self, arrows, text, flip):
        self.cv.delete("all")
        cs = self.roi["width"] / 8
        def xy(sq):
            c, r = cell(sq, flip); return (c + .5) * cs, (r + .5) * cs
        for m, color, width in arrows:
            (x1, y1), (x2, y2) = xy(m.from_square), xy(m.to_square)
            self.cv.create_line(x1, y1, x2, y2, fill=color, width=width, arrow=tk.LAST,
                                arrowshape=(18, 22, 8), capstyle="round")
        self.label.config(text=text)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--color", choices=["white", "black"], default="white")
    a = ap.parse_args()
    flip = a.color == "black"; mine = chess.WHITE if not flip else chess.BLACK
    try: ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception: pass

    sct = mss.mss()
    shot = np.array(sct.grab(sct.monitors[1]))[:, :, :3]
    k = min(1.0, 1400 / shot.shape[1])
    x, y, w, h = cv2.selectROI("Drag around the board, press ENTER", cv2.resize(shot, None, fx=k, fy=k))
    cv2.destroyAllWindows()
    side = int(min(w, h) / k)
    roi = {"left": int(x / k), "top": int(y / k), "width": side, "height": side}
    frame = np.array(sct.grab(roi))[:, :, :3]
    templates = learn(frame, flip)
    print("Learned", len(templates), "templates. Engines starting...")

    engines = {}
    for n, (cmd, opts) in ENGINES.items():
        try:
            e = chess.engine.SimpleEngine.popen_uci(cmd); e.configure(opts); engines[n] = e
        except Exception as ex:
            print(f"[skip] {n}: {ex}")
    if not engines: sys.exit("No engines loaded. Edit ENGINES.")

    state = {"fen": None, "result": None, "board": chess.Board(), "stable": 0, "last": None}
    lock = threading.Lock()

    def worker():
        done = None
        while True:
            with lock: fen = state["fen"]
            if fen is None or fen == done: time.sleep(0.05); continue
            board = chess.Board(fen)
            if not board.is_valid() or board.is_game_over(): done = fen; continue
            with ThreadPoolExecutor(len(engines)) as ex:
                res = dict(ex.map(lambda kv: run_engine(kv[0], kv[1], board), engines.items()))
            with lock:
                if state["fen"] == fen: state["result"] = (fen, board, res)
            done = fen
    threading.Thread(target=worker, daemon=True).start()

    ov = Overlay(roi)

    def tick():
        try:
            pl = read(np.array(sct.grab(roi))[:, :, :3], flip, templates)
            state["stable"] = state["stable"] + 1 if pl == state["last"] else 0
            state["last"] = pl
            if state["stable"] == 2:
                state["board"] = track(state["board"], pl, mine)
                with lock: state["fen"] = state["board"].fen()
            with lock: r = state["result"]
            if r and r[0] == state["board"].fen():
                _, board, res = r
                votes, arrows, lines = {}, [], []
                for n, infos in res.items():
                    m = infos[0]["pv"][0]; votes.setdefault(m, []).append(n)
                    s = infos[0]["score"].pov(board.turn)
                    lines.append(f"{n:<10}{board.san(m):<7}{s}")
                for m, ns in votes.items():
                    arrows.append((m, COLORS[ns[0]], 10 if len(ns) > 1 else 5))
                top = max(votes, key=lambda m: len(votes[m]))
                lines.insert(0, f"CONSENSUS: {board.san(top)} ({len(votes[top])}/{len(res)})")
                first = list(ENGINES)[0]
                if first in res:
                    bril = brilliant(board, res[first])
                    for m in bril[:2]: arrows.append((m, "#ffd700", 12))
                    lines.append("BRILLIANT: " + (", ".join(board.san(m) for m in bril) or "none"))
                ov.draw(arrows, "\n".join(lines), flip)
        except Exception as ex:
            print("tick error:", ex)
        ov.root.after(150, tick)

    tick(); ov.root.mainloop()

if __name__ == "__main__":
    main()
