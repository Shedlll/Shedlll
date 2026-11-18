"""Shed's GitHub profile hero: an animated terminal card.

Intro: A/T/G/C letters stream out of the centre, a lit 3D DNA double helix rises
out of the depth, the window and welcome panel draw themselves in, the helix
settles into its slot and keeps turning, and the prompt types a command.
Palette: dark green + burgundy on near-black.
"""
import base64, math, os, random
import numpy as np

FONT_DIR = "/home/claude/gh/fontdl/node_modules/@fontsource/jetbrains-mono/files"
def font_b64(weight):
    with open(f"{FONT_DIR}/jetbrains-mono-latin-{weight}-normal.woff2", "rb") as fh:
        return base64.b64encode(fh.read()).decode()

# ---------------------------------------------------------------- palette
C = dict(screen="#0b0b0a", bar="#141413", ink="#ececea", soft="#a3a39f", dim="#666662",
         line="#34342f", box="#4a4a44")
TONES = {  # strand A = dark green, strand B = burgundy; 0 far/shadow .. 3 lit
    "g": ["#16301f", "#245238", "#357a52", "#5a9e74"],
    "b": ["#33111a", "#5e1b2b", "#8c2a3f", "#b44a5e"],
}
GREEN, BURG = "#245238", "#5e1b2b"

# ---------------------------------------------------------------- helix
ART = dict(cols=32, rows=20, size=10.4, frames=60, loop=6.0)
RAMP = ".,:;i+*o#%@"

def helix_frames():
    cols, rows, NF = ART["cols"], ART["rows"], ART["frames"]
    aspect = 0.6
    R, tube, pitch = 1.0, 0.17, 0.50           # helix radius, strand thickness, rise per radian
    groove = 0.78 * math.pi                    # strands offset < pi: major/minor groove
    S, K2 = 5.0, 7.0                           # screen scale (in rows), camera distance
    tilt = 0.14
    light = np.array([-0.45, 0.65, -1.0]); light /= np.linalg.norm(light)

    # strands as lit tubes: points + normals, tagged by strand
    ts = np.linspace(-2.8 * math.pi, 2.8 * math.pi, 1600)
    vs = np.linspace(0, 2 * math.pi, 26, endpoint=False)
    P, N, TAG, LV = [], [], [], []
    for tag, ph in ((0, 0.0), (1, groove)):
        c = np.stack([R * np.cos(ts + ph), ts * pitch, R * np.sin(ts + ph)], 1)
        d = np.stack([-R * np.sin(ts + ph), np.full_like(ts, pitch), R * np.cos(ts + ph)], 1)
        T = d / np.linalg.norm(d, axis=1, keepdims=True)
        n1 = np.stack([np.cos(ts + ph), np.zeros_like(ts), np.sin(ts + ph)], 1)  # radial, ⟂ T
        b1 = np.cross(T, n1)
        for v in vs:
            n = n1 * math.cos(v) + b1 * math.sin(v)
            P.append(c + tube * n); N.append(n)
            TAG.append(np.full(len(ts), tag)); LV.append(np.full(len(ts), 3))
    # base pairs: rungs between the strands, each half in its strand's colour, dimmer
    for t in np.arange(-2.8 * math.pi, 2.8 * math.pi, 0.78):
        a = np.array([R * math.cos(t), t * pitch, R * math.sin(t)])
        b = np.array([R * math.cos(t + groove), t * pitch, R * math.sin(t + groove)])
        ax = (b - a) / np.linalg.norm(b - a)
        u = np.cross(ax, [0, 1, 0]); u /= np.linalg.norm(u); w = np.cross(ax, u)
        for s in np.linspace(0.08, 0.92, 26):
            for v in np.linspace(0, 2 * math.pi, 10, endpoint=False):
                n = u * math.cos(v) + w * math.sin(v)
                P.append((a + (b - a) * s + 0.05 * n)[None]); N.append(n[None])
                TAG.append(np.array([0 if s < 0.5 else 1])); LV.append(np.array([-1]))
    P = np.concatenate(P); N = np.concatenate(N); TAG = np.concatenate(TAG); LV = np.concatenate(LV)

    ct, st = math.cos(tilt), math.sin(tilt)
    out = []
    for f in range(NF):
        A = f / NF * 2 * math.pi                 # one full turn per loop: seamless
        ca, sa = math.cos(A), math.sin(A)
        def rot(V):
            x = V[:, 0] * ca + V[:, 2] * sa
            z = -V[:, 0] * sa + V[:, 2] * ca
            y = V[:, 1]
            return x, y * ct - z * st, y * st + z * ct
        X, Y, Z = rot(P)
        nx, ny, nz = rot(N)
        persp = K2 / (Z + K2)
        col = np.rint(cols / 2 + S * X * persp / aspect).astype(int)
        row = np.rint(rows / 2 - S * Y * persp).astype(int)
        ok = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
        lum = nx * light[0] + ny * light[1] + nz * light[2]
        depth = np.clip((Z + 1.2) / 2.4, 0, 1)           # 0 near .. 1 far
        val = np.clip(((lum + 1) / 2) ** 1.4 * (1 - 0.6 * depth), 0, 0.999)
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(-Z[idx])]                  # far first, near overwrites
        chars = [[" "] * cols for _ in range(rows)]
        cls = [[None] * cols for _ in range(rows)]
        for k in order:
            r, c = row[k], col[k]
            if LV[k] < 0:   # base-pair rung: bar glyphs
                chars[r][c] = "-=≡"[min(2, int(val[k] * 3))] if False else "-="[int(val[k] * 1.999)]
                tone = 1 + int(val[k] * 1.999)
            else:
                chars[r][c] = RAMP[int(val[k] * len(RAMP))]
                tone = min(3, int(val[k] * 4))
            cls[r][c] = ("g" if TAG[k] == 0 else "b") + str(tone)
        out.append((chars, cls))
    return out

def frame_text(chars, cls, klass, style):
    size = ART["size"]; adv = size * 0.6
    body = []
    for r, line in enumerate(chars):
        s = "".join(line)
        if not s.strip():
            continue
        lead = len(s) - len(s.lstrip()); end = len(s.rstrip())
        runs, cur, buf = [], None, ""
        for c in range(lead, end):
            k = cls[r][c] if line[c] != " " else cur
            if k != cur and buf:
                runs.append(f'<tspan class="{cur}">{buf}</tspan>'); buf = ""
            cur = k; buf += line[c]
        if buf:
            runs.append(f'<tspan class="{cur}">{buf}</tspan>')
        body.append(f'<tspan x="{lead*adv:.1f}" y="{(r+1)*size-2:.1f}">{"".join(runs)}</tspan>')
    return f'<text class="{klass}" style="{style}">{"".join(body)}</text>'

# ---------------------------------------------------------------- timeline helper
class Clock:
    def __init__(self, T): self.T, self.rules, self.n = T, [], 0
    def anim(self, stops, extra=""):
        """stops: [(sec, {'o':..,'t':..}, ease?)]; holds first/last stop."""
        i = f"k{self.n}"; self.n += 1
        def body(p, ease=None):
            parts = []
            if "o" in p: parts.append(f"opacity:{p['o']}")
            if "t" in p: parts.append(f"transform:{p['t']}")
            if ease: parts.append(f"animation-timing-function:{ease}")
            return ";".join(parts)
        fr = []
        if stops[0][0] > 0: fr.append(f"0%{{{body(stops[0][1])}}}")
        for st in stops:
            fr.append(f"{st[0]/self.T*100:.3f}%{{{body(st[1], st[2] if len(st) > 2 else None)}}}")
        fr.append(f"100%{{{body(stops[-1][1])}}}")
        self.rules.append(f"@keyframes {i}{{{''.join(fr)}}}.{i}{{animation:{i} {self.T}s linear both;{extra}}}")
        return i
    def css(self): return "".join(self.rules)

def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ---------------------------------------------------------------- card
def hero():
    W, H, L, R, TOP = 946, 428, 5, 941, 32
    BOX = dict(x=24, y=50, w=898, h=284); DIV = 352
    clk = Clock(5.4)
    OUT = "cubic-bezier(.2,.8,.2,1)"
    def show(at, frm="translateY(5px)"):
        return clk.anim([(at, {"o": 0, "t": frm}, OUT), (at + 0.25, {"o": 1, "t": "none"})])
    def t(x, y, s, fill, size=13, extra=""):
        return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}"{extra}>{esc(s)}</text>'

    # helix frames, looping from load
    frames = helix_frames(); NF, loop = ART["frames"], ART["loop"]; dt = loop / NF
    art = "".join(frame_text(ch, cl, "f f0" if i == 0 else "f", f"animation-delay:{-(loop - i*dt):.3f}s")
                  for i, (ch, cl) in enumerate(frames))
    artW, artH = ART["cols"] * ART["size"] * 0.6, ART["rows"] * ART["size"]
    leftMid = BOX["x"] + (DIV - BOX["x"]) / 2
    slot = (leftMid - artW / 2 + 10, BOX["y"] + 40)
    at = lambda x, y, s: f"translate({x:.1f}px,{y:.1f}px) scale({s})"
    big, cx, cy = 1.5, W / 2, H / 2 + 4
    centre = at(cx - artW * big / 2, cy - artH * big / 2, big)
    deep = at(cx - artW * 0.05 / 2, cy - artH * 0.05 / 2, 0.05)
    land = at(*slot, 1)
    helix = clk.anim([(0.35, {"o": 0, "t": deep}, "cubic-bezier(.16,1,.3,1)"),
                      (1.75, {"o": 1, "t": centre}),
                      (3.15, {"o": 1, "t": centre}, "cubic-bezier(.7,0,.25,1)"),
                      (3.95, {"o": 1, "t": land})], "transform-origin:0 0")

    # warp: nucleotide letters + streaks flying out of the centre
    rng = random.Random(42)
    letters, lcss = [], []
    for i in range(96):
        ang = rng.uniform(0, 2 * math.pi); reach = rng.uniform(380, 640)
        dx, dy = math.cos(ang) * reach, math.sin(ang) * reach * 0.6
        deg = math.degrees(math.atan2(dy, dx))
        dur = rng.uniform(0.55, 1.1); delay = rng.uniform(0.05, 1.0)
        streak = i % 3 == 0
        ch = "-" if streak else "ATGC"[i % 4]
        end = "scale(5.5,1.1)" if streak else "scale(2.2)"
        rot = f" rotate({deg:.0f}deg)" if streak else ""
        fill = [TONES["g"][3], TONES["b"][3], C["soft"], TONES["g"][2], TONES["b"][2]][i % 5]
        reps = max(1, math.ceil((3.1 - delay) / dur))
        lcss.append(f"@keyframes w{i}{{0%{{opacity:0;transform:translate({cx:.0f}px,{cy:.0f}px){rot} scale(.25)}}"
                    f"25%{{opacity:1}}100%{{opacity:0;transform:translate({cx+dx:.0f}px,{cy+dy:.0f}px){rot} {end}}}}}"
                    f".w{i}{{animation:w{i} {dur:.2f}s cubic-bezier(.55,0,1,.45) {delay:.2f}s {reps} both}}")
        letters.append(f'<text class="w{i}" font-size="13" text-anchor="middle" fill="{fill}">{ch}</text>')
    warp = clk.anim([(0, {"o": 1}), (2.75, {"o": 1}), (3.2, {"o": 0})])

    # window chrome
    chrome = clk.anim([(2.8, {"o": 0}), (3.1, {"o": 1})])
    dots = "".join(f'<circle cx="{L+20+i*18}" cy="16" r="5.5" fill="{c}"/>'
                   for i, c in enumerate([TONES["b"][2], "#4a4536", TONES["g"][2]]))
    titlebar = (f'<g class="{chrome}"><path d="M{L} 12a12 12 0 0 1 12-12h{R-L-24}a12 12 0 0 1 12 12v{TOP-12}H{L}z" '
                f'fill="{C["bar"]}"/>{dots}{t(W/2, 20.5, "~/shedlll — oh-my-posh", C["dim"], 12, " text-anchor=\"middle\"")}</g>')

    # panel
    label = "shedlll"; labelW = len(label) * 13 * 0.6 + 14
    panel = (f'<rect x="{BOX["x"]}" y="{BOX["y"]}" width="{BOX["w"]}" height="{BOX["h"]}" rx="8" fill="none" '
             f'stroke="{C["box"]}" stroke-width="1.2" pathLength="1" class="dr"/>'
             f'<g class="{show(3.55, "none")}"><rect x="{BOX["x"]+14}" y="{BOX["y"]-8}" width="{labelW}" height="16" '
             f'fill="{C["screen"]}"/>{t(BOX["x"]+21, BOX["y"]+4.5, label, TONES["g"][3], 13)}</g>'
             f'<path d="M{DIV} {BOX["y"]+22}V{BOX["y"]+BOX["h"]-22}" stroke="{C["line"]}" stroke-width="1.2" '
             f'class="{show(3.6, "scaleY(0)")}" style="transform-origin:{DIV}px {BOX["y"]+BOX["h"]/2}px"/>')

    left = (f'<g class="{show(3.9)}">{t(leftMid, BOX["y"]+30, "Shed", C["ink"], 15, " text-anchor=\"middle\" font-weight=\"700\"")}</g>'
            f'<g class="{show(4.0)}">{t(leftMid, BOX["y"]+BOX["h"]-16, "Montréal, Québec", C["dim"], 12.5, " text-anchor=\"middle\"")}</g>')

    RX = DIV + 26
    rows = [("About", "h"),
            ("Student heading toward AI engineering or bioengineering.", "b"),
            ("Wrote my own antivirus in C.", "b"),
            (None, "rule"),
            ("Stack", "h"),
            ("C · Python · Three.js · Claude API", "b"),
            (None, "gap"),
            ("Builds on", "h"),
            ('Windows · macOS <tspan fill="%s">→ Omarchy, someday</tspan>' % C["dim"], "raw"),
            (None, "gap"),
            ("Speaks", "h"),
            ("English · Français · العربية", "b")]
    y, right = BOX["y"] + 36, []
    for i, (s, kind) in enumerate(rows):
        when = 3.95 + i * 0.055
        if kind == "rule":
            right.append(f'<path d="M{RX} {y-6}H{BOX["x"]+BOX["w"]-26}" stroke="{C["line"]}" stroke-width="1" '
                         f'class="{show(when, "scaleX(0)")}" style="transform-origin:{RX}px 0"/>'); y += 22; continue
        if kind == "gap": y += 8; continue
        if kind == "raw":
            right.append(f'<g class="{show(when)}"><text x="{RX}" y="{y}" font-size="13" fill="{C["soft"]}">{s}</text></g>'); y += 20; continue
        if kind == "h":
            right.append(f'<g class="{show(when)}"><rect x="{RX}" y="{y-8.5}" width="3" height="9" rx="1" fill="{TONES["b"][3] if i % 2 else TONES["g"][3]}"/>'
                         f'{t(RX+10, y, s, C["ink"], 13, " font-weight=\"700\"")}</g>'); y += 22
        else:
            right.append(f'<g class="{show(when)}">{t(RX, y, s, C["soft"], 13)}</g>'); y += 20

    # prompt: oh-my-posh powerline segments, then a typed command
    PY, PH = BOX["y"] + BOX["h"] + 14, 40
    x0, segY, segH = BOX["x"] + 12, PY + 9, 22
    s1, s2 = "shedlll", "~/projects"
    w1, w2 = len(s1) * 7.5 + 22, len(s2) * 7.5 + 26
    seg = (f'<path d="M{x0+4} {segY}h{w1-4}l10 {segH/2}-10 {segH/2}H{x0+4}a4 4 0 0 1-4-4v-{segH-8}a4 4 0 0 1 4-4z" fill="{GREEN}"/>'
           f'{t(x0+11, segY+15.5, s1, "#d6eadc", 12.5, " font-weight=\"700\"")}'
           f'<path d="M{x0+w1} {segY}h{w2}l10 {segH/2}-10 {segH/2}H{x0+w1}l10-{segH/2}z" fill="{BURG}"/>'
           f'{t(x0+w1+18, segY+15.5, s2, "#f0d3d9", 12.5)}')
    cmd = "brew install curiosity"; adv = 13.5 * 0.6
    tx = x0 + w1 + w2 + 26
    caret = t(tx, PY + 25, "❯", TONES["g"][3], 14)
    cx0 = tx + 18
    t0, step = 4.85, 0.06
    total = t0 + len(cmd) * step + 0.01
    vals = ";".join(["0"] + [f"{(k+1)*adv:.1f}" for k in range(len(cmd))])
    keys = ";".join(["0"] + [f"{(t0 + k*step)/total:.4f}" for k in range(len(cmd))])
    cvals = ";".join([f"{cx0:.1f}"] + [f"{cx0+(k+1)*adv:.1f}" for k in range(len(cmd))])
    typed = (f'<clipPath id="ty"><rect x="{cx0}" y="{PY}" width="0" height="{PH}">'
             f'<animate attributeName="width" values="{vals}" keyTimes="{keys}" dur="{total:.2f}s" calcMode="discrete" fill="freeze"/></rect></clipPath>'
             f'<g clip-path="url(#ty)">{t(cx0, PY+25, cmd, C["ink"], 13.5)}</g>'
             f'<rect x="{cx0}" y="{PY+12}" width="8.1" height="16" fill="{C["ink"]}" class="bl">'
             f'<animate attributeName="x" values="{cvals}" keyTimes="{keys}" dur="{total:.2f}s" calcMode="discrete" fill="freeze"/></rect>')
    prompt = (f'<g class="{show(4.45)}"><rect x="{BOX["x"]}" y="{PY}" width="{BOX["w"]}" height="{PH}" rx="7" fill="none" '
              f'stroke="{C["box"]}" stroke-width="1.2"/>{seg}{caret}{typed}</g>'
              f'<g class="{show(4.6)}">{t(BOX["x"]+12, PY+PH+19, "? for shortcuts", C["dim"], 12)}'
              f'{t(BOX["x"]+BOX["w"]-12, PY+PH+19, "brew · omp · win + mac", C["dim"], 12, " text-anchor=\"end\"")}</g>')

    tone_css = "".join(f".{k}{i}{{fill:{c}}}" for k, cs in TONES.items() for i, c in enumerate(cs))
    css = (f"@font-face{{font-family:JBM;font-weight:500;src:url(data:font/woff2;base64,{font_b64(500)}) format('woff2')}}"
           f"@font-face{{font-family:JBM;font-weight:700;src:url(data:font/woff2;base64,{font_b64(700)}) format('woff2')}}"
           "text{font-family:JBM,ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-weight:500;white-space:pre}"
           + tone_css +
           f".f{{font-size:{ART['size']}px;opacity:0;animation:fr {loop}s step-end infinite}}.f0{{opacity:1}}"
           f"@keyframes fr{{0%{{opacity:1}}{100/NF:.3f}%{{opacity:0}}100%{{opacity:0}}}}"
           ".dr{stroke-dasharray:1;animation:dr .85s cubic-bezier(.65,0,.35,1) 3.0s both}"
           "@keyframes dr{from{stroke-dashoffset:1}to{stroke-dashoffset:0}}"
           ".bl{animation:bl 1.05s steps(1) infinite}@keyframes bl{50%{opacity:0}}"
           + "".join(lcss) + clk.css() +
           "@media (prefers-reduced-motion:reduce){*{animation:none!important}.intro{display:none}}")

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Animated terminal card for Shed, Montréal: a spinning ASCII DNA helix in dark green and burgundy. Heading toward AI engineering or bioengineering; wrote an antivirus in C. Stack: C, Python, Three.js, Claude API. Builds on Windows and macOS. Speaks English, French and Arabic.">
<title>Shed</title>
<defs><clipPath id="win"><rect x="{L}" y="0" width="{R-L}" height="{H-4}" rx="12"/></clipPath></defs>
<style>{css}</style>
<rect x="{L}" y="0" width="{R-L}" height="{H-4}" rx="12" fill="{C["screen"]}" stroke="#1e1e1c"/>
<g clip-path="url(#win)">
<g class="intro {warp}">{"".join(letters)}</g>
{panel}
<g class="{helix}" style="transform:{land};transform-origin:0 0">{art}</g>
{titlebar}{left}{"".join(right)}{prompt}
</g>
</svg>'''

if __name__ == "__main__":
    svg = hero()
    open(os.path.join(os.path.dirname(__file__), "..", "assets", "hero.svg"), "w").write(svg)
    print(f"{len(svg)/1024:.0f} KB")
