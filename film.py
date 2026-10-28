# Safe to forget (Seal, 2026). Olkiluoto in log time, 2026 -> ~1,000,000 years.
# One program makes the picture and the score.
# python3 film.py stills 10,40  |  python3 film.py full out.mp4   (env W, FPS, PROCS)
import os, sys, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W = int(os.environ.get('W', 1920)); H = W * 9 // 16
FPS = int(os.environ.get('FPS', 30))
S = W / 1920.0
DUR, T0, T1 = 100.0, 5.0, 90.0          # film seconds; the clock runs T0..T1
EMAX = 1.0e6                            # years after 2026 at the end
LOGE = math.log10(EMAX + 1)
NC, NROW, NCOL = 3300, 50, 66           # canisters on a 50 x 66 grid at 430 m
CLOSE = 95.0                            # closure ~2121

def elapsed(t):
    u = min(max((t - T0) / (T1 - T0), 0.0), 1.0)
    return 10 ** (u * LOGE) - 1

def film_t(E):
    return T0 + (T1 - T0) * np.log10(np.asarray(E, float) + 1) / LOGE

def rate(t):                            # years per film second
    return (elapsed(t + .02) - elapsed(t - .02)) / .04

def ss(a, b, x):
    x = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)

def fss(a, b, x):
    return float(ss(a, b, x))

rng = np.random.default_rng(430)
col_order = np.argsort(np.arange(NCOL) + rng.normal(0, 5, NCOL))
CCOL = np.repeat(col_order, NROW)
CROW = np.tile(np.arange(NROW - 1, -1, -1), NCOL)   # each tunnel fills far end first
q = (np.arange(NC) + rng.random(NC)) / NC
EMPL = 0.3 + (CLOSE - 3.3) * q ** 0.7               # years after 2026; ~51/yr at the end
TEMPL = film_t(EMPL)
PH = rng.random(NC) * 6.2832
T_CLOSE = float(film_t(CLOSE))

# the long climate: a warm interval, then 100 kyr glacial cycles
E0, PER = 52000.0, 100000.0
def _cyc(E):
    p = ((E - E0) / PER) % 1.0
    k = int((E - E0) // PER)
    return p, 0.8 + 0.2 * math.sin(k * 1.9 + 0.5)

def ice(E):
    if E < E0: return 0.0
    p, amp = _cyc(E)
    if p < 0.82: return amp * fss(0.0, 0.82, p) ** 1.4
    return amp * (1 - fss(0.82, 0.9, p))

def cold(E):
    if E < E0: return 0.5 * fss(15000, E0, E)
    p, amp = _cyc(E)
    if p < 0.82: return min(1.0, 0.5 + 0.6 * fss(0.0, 0.5, p))
    return 0.5 * (1 - fss(0.82, 0.88, p)) + 0.5 * fss(0.94, 1.0, p)

def veg(E):
    return (1 - fss(0.35, 0.7, cold(E))) * (1 - fss(0.02, 0.2, ice(E)))
def plant(E): return 1 - fss(CLOSE, CLOSE + 110, E)
def shaft(E): return 1 - fss(CLOSE - 3, CLOSE + 6, E)
def counter_a(E): return 1 - fss(300, 700, E)

_EG = np.concatenate([[0.0], np.logspace(-2, LOGE, 16000)])
_IC = np.array([ice(e) for e in _EG]); _DP = np.zeros_like(_EG)
for _i in range(1, len(_EG)):
    _a = 1 - math.exp(-(_EG[_i] - _EG[_i - 1]) / 7000.0)
    _DP[_i] = _DP[_i - 1] + (_IC[_i] - _DP[_i - 1]) * _a

def rsl(E):   # relative sea level at the site in metres (+ = sea higher)
    dep = float(np.interp(E, _EG, _DP))
    return -38 * (1 - math.exp(-E / 6000.0)) + 120 * dep

# ---------------- picture: static geometry ----------------
from scipy.ndimage import gaussian_filter
YS = 0.30 * H                       # surface datum (y px)
VEX = 3.0 * S                       # px per metre above ground (exaggerated)
VS = (0.70 * H - YS) / 430.0        # px per metre below ground
XN = np.linspace(0, 1, W)
HT = 6 + 24 * (0.5 - XN) + 1.6 * np.sin(XN * 23) + 0.8 * np.sin(XN * 61 + 1)  # terrain m
GY = YS - HT * VEX                  # ground line per column
YY = np.arange(H, dtype=np.float32)[:, None]
UNDER = YY >= GY[None, :]

def grad(top, bot):
    a = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    return (np.array(top, np.float32) * (1 - a) + np.array(bot, np.float32) * a) * np.ones((1, W, 1), np.float32)
SKY_W = grad((6, 9, 20), (34, 40, 58))
SKY_C = grad((26, 32, 44), (120, 132, 146))
_n = gaussian_filter(rng.standard_normal((H // 4, W // 4)).astype(np.float32), (1.0, 7.0))
_n = np.array(Image.fromarray(_n).resize((W, H), Image.BILINEAR))
_d = np.clip((YY - GY[None, :]) / (H - YS), 0, 1)
ROCK = np.stack([24 + 10 * _n - 10 * _d, 23 + 9 * _n - 9 * _d, 27 + 9 * _n - 6 * _d], -1).astype(np.float32)
ROCK += np.clip(np.where((YY - GY[None, :]) < 6 * S, 18, 0), 0, 18)[..., None] * np.array([0.9, 0.8, 0.6], np.float32)

# repository plane: tunnels recede into the screen
YHZ, YNEAR = 0.595 * H, 0.83 * H
def proj(xn, z):
    s = 1.0 / (1.0 + 2.0 * z)
    return W / 2 + (xn - 0.5) * 0.80 * W * s, YHZ + (YNEAR - YHZ) * s, s
CX, CY, CS = proj(CCOL / (NCOL - 1), CROW / (NROW - 1))
CXI = np.clip(CX.astype(int), 0, W - 2); CYI = np.clip(CY.astype(int), 0, H - 2)
HUE = np.stack([np.full(NC, 1.0), 0.50 + 0.18 * rng.random(NC), 0.22 + 0.16 * rng.random(NC)], 1)

# the plant: Olkiluoto halls, encapsulation plant, headframe (xn, width, height px@1080)
BLD = [(0.12, .07, 70), (0.20, .05, 120), (0.255, .03, 160), (0.29, .06, 95), (0.36, .05, 60),
       (0.405, .012, 140), (0.45, .04, 55), (0.53, .06, 40)]
WIN = [(rng.random(), rng.random(), rng.random()) for _ in range(160)]
SHAFTS = [0.405, 0.43, 0.455]
TREES = sorted([(rng.random() * 0.98 + 0.01, 0.6 + 0.8 * rng.random(), rng.random()) for _ in range(260)])

# ---------------- picture: one frame ----------------
try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', int(30 * S))
    FONTB = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', int(54 * S))
    FONTS = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', int(16 * S))
except Exception:
    FONT = FONTB = FONTS = ImageFont.load_default()
YREP = 0.70 * H

def heat(E):                         # spent-fuel glow, never quite zero
    d = np.maximum(E - EMPL, 0.0)
    return 0.05 + 0.95 * (1 + d / 40.0) ** -0.55

def render(t):
    E = elapsed(t); c = cold(E); ic = ice(E); v = veg(E); sl = rsl(E)
    img = SKY_W * (1 - c) + SKY_C * c
    # ice sheet above the ground
    if ic > 0.002:
        top = GY - ic * 0.22 * H * (0.75 + 0.25 * np.sin(XN * 3 + 1))
        m = (YY >= top[None, :]) & ~UNDER
        shade = (0.75 + 0.25 * (YY - top[None, :]) / (0.22 * H + 1))
        img = np.where(m[..., None], np.array([196, 210, 224], np.float32) * np.clip(shade, .6, 1)[..., None], img)
    # sea: fills wherever ground is below sea level
    sy = YS - sl * VEX
    sea = (YY >= sy) & ~UNDER
    if ic > 0.002: sea &= ~m
    img = np.where(sea[..., None], np.array([18, 34, 52], np.float32) * (1 - .5 * c) + np.array([60, 70, 80], np.float32) * c, img)
    # rock, with permafrost reaching down from the surface
    rock = ROCK.copy()
    pf = fss(0.45, 1.0, c) * (1 - 0.6 * ic) * 260.0
    if pf > 1:
        dz = (YY - GY[None, :]) / VS
        a = np.clip((pf - dz) / 30.0, 0, 1)[..., None] * 0.55
        rock = rock * (1 - a) + np.array([90, 104, 118], np.float32) * a
    img = np.where(UNDER[..., None], rock, img)
    # tunnels at 430 m: open and faintly lit, then backfilled
    tun = shaft(E)
    glow = np.zeros((H // 4, W // 4, 3), np.float32)
    on = E >= EMPL
    fl = np.exp(-np.maximum(t - TEMPL, 0) / 0.18) * on
    tw = 1 + 0.35 * np.sin(PH + t * (9 + 4 * HUE[:, 1])) * fss(T0, T_CLOSE, t) * (1 - fss(T_CLOSE, T_CLOSE + 4, t))
    I = (heat(E) * tw + 2.5 * fl) * on * CS ** 0.6
    col = HUE * I[:, None] * np.array([255, 150, 70], np.float32)
    for k in range(3):
        np.add.at(glow[..., k], (CYI // 4, CXI // 4), col[:, k] * 0.9)
    glow = gaussian_filter(glow, (2.2, 2.2, 0)) * 3.0 + gaussian_filter(glow, (9, 9, 0)) * 8.0
    big = np.array(Image.fromarray(np.clip(glow, 0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), np.float32)
    img = img + big
    sharp = np.zeros((H, W, 3), np.float32)
    np.add.at(sharp, (CYI, CXI), col * 0.9)
    img = img + sharp
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out, 'RGBA')
    if tun > 0.01:
        for r in range(0, NROW, 7):
            x0, y0, _ = proj(0, r / (NROW - 1)); x1, _, _ = proj(1, r / (NROW - 1))
            d.line([(x0, y0), (x1, y0)], fill=(150, 110, 70, int(30 * tun)), width=1)
        for cc in range(0, NCOL, 6):
            a0 = proj(cc / (NCOL - 1), 0); a1 = proj(cc / (NCOL - 1), 1)
            d.line([a0[:2], a1[:2]], fill=(150, 110, 70, int(22 * tun)), width=1)
        for sx in SHAFTS:
            px = sx * W; gy = GY[int(px)]
            d.line([(px, gy), (px, YHZ)], fill=(210, 150, 80, int(120 * tun)), width=max(1, int(2 * S)))
    # the plant above ground
    pa = plant(E)
    if pa > 0.01:
        for i, (bx, bw, bh) in enumerate(BLD):
            x0 = int((bx - bw / 2) * W); x1 = int((bx + bw / 2) * W); gy = float(GY[int(bx * W)])
            hh = bh * S * (0.3 + 0.7 * pa)
            d.rectangle([x0, gy - hh, x1, gy + 2], fill=(14, 15, 20, int(255 * pa)))
            for (wx, wy, wp) in WIN[i * 20:(i + 1) * 20]:
                if math.sin(wp * 40 + t * 0.7) > 0.2 - pa:
                    px = x0 + 3 + wx * (x1 - x0 - 6); py = gy - hh + 4 + wy * (hh - 8)
                    d.rectangle([px, py, px + 2 * S, py + 2 * S], fill=(255, 200, 120, int(200 * pa)))
    # forest: only on exposed land
    nt = int(len(TREES) * v)
    for (tx, ts, tp) in TREES:
        if tp > v: continue
        px = int(tx * W); gy = float(GY[px])
        if sy <= gy and sl > HT[px]: continue
        if pa > 0.3 and 0.08 < tx < 0.58: continue
        hh = 14 * S * ts * fss(tp, tp + 0.15, v + 0.1)
        d.polygon([(px - hh * .3, gy), (px + hh * .3, gy), (px, gy - hh)], fill=(16, 30, 22, 235))
    return out, d, E

# ---------------- picture: words and finish ----------------
def year_text(E):
    return 'anno %d' % int(2026 + E)

def frame(t):
    out, d, E = render(t)
    a = counter_a(E) * fss(T0 - 2.5, T0 - 0.5, t)
    if a > 0.01:
        d.text((0.06 * W, 0.08 * H), year_text(E), font=FONT, fill=(230, 214, 190, int(220 * a)))
    da = fss(T0 - 3, T0, t) * (1 - fss(T_CLOSE + 2, T_CLOSE + 8, t))
    if da > 0.01:
        d.text((0.06 * W, YREP - 8 * S), '-430 m', font=FONTS, fill=(200, 170, 130, int(150 * da)))
    ta = fss(0.3, 2.0, t) * (1 - fss(3.6, 5.0, t))
    if ta > 0.01:
        tw = d.textlength('Safe to forget', font=FONTB)
        d.text(((W - tw) / 2, 0.10 * H), 'Safe to forget', font=FONTB, fill=(236, 224, 204, int(235 * ta)))
    ea = fss(DUR - 7, DUR - 4, t)
    if ea > 0.01:
        s1 = 'Olkiluoto. Licensed 22 October 2026.'
        s2 = 'Built so that no one has to remember.'
        for i, s in enumerate((s1, s2)):
            tw = d.textlength(s, font=FONTS)
            d.text(((W - tw) / 2, 0.10 * H + i * 26 * S), s, font=FONTS, fill=(220, 214, 204, int(200 * ea)))
    f = np.asarray(out, np.float32)
    f *= VIG
    f *= fss(0.0, 1.2, t) * (1 - fss(DUR - 1.5, DUR, t)) * 0.999 + 0.001
    f += rng.normal(0, 2.2, (H, W, 1)).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)

_yy, _xx = np.mgrid[0:H, 0:W]
VIG = (1 - 0.35 * (((_xx - W / 2) / (W / 2)) ** 2 + ((_yy - H / 2) / (H / 2)) ** 2) ** 1.4)[..., None].astype(np.float32)
VIG = np.clip(VIG, 0.4, 1)
