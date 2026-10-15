"""Cricket Scout AI: context-adjusted IPL batting analytics (Streamlit dashboard)."""
import html
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Cricket Scout AI", page_icon="🏏", layout="wide")

GITHUB = "https://github.com/anuu-yd9/cricket-scout-ai"
BLUE, VIOLET, PINK, GREEN, AMBER, RED = "#3b82f6", "#8b5cf6", "#ec4899", "#2dd4a0", "#f5a524", "#ff5d73"
STYLE_COLORS = {
    "Aggressive powerplay hitter": RED,
    "Power finisher": AMBER,
    "Anchor": BLUE,
    "Traditional opener": GREEN,
    "Middle-order accelerator": VIOLET,
}
FEATS = ["rel_sr", "four_pct", "six_pct", "dot_pct", "balls_per_out", "pp_share", "death_share"]
FEAT_NAMES = {
    "rel_sr": "scoring rate", "four_pct": "fours", "six_pct": "sixes", "dot_pct": "dot balls",
    "balls_per_out": "survival", "pp_share": "powerplay overs", "death_share": "death overs",
}

# ----------------------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{--ease:cubic-bezier(.22,1,.36,1);--panel:#07080c;--panel2:#0c0e14;--line:#1b1e2b;--line-hi:#2c3350;
--text:#f4f5f7;--muted:#9aa1b2;--dim:#7d8497;--blue:#3b82f6;--violet:#8b5cf6;--pink:#ec4899;--green:#2dd4a0;
--mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
html,body,.stApp{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif}
.stApp{background:radial-gradient(1100px 460px at 50% -140px,rgba(59,130,246,.17),transparent 62%),
radial-gradient(700px 380px at 92% 8%,rgba(139,92,246,.10),transparent 60%),#000}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{visibility:hidden;height:0}
.block-container{max-width:1180px;padding-top:1.2rem;padding-bottom:3.5rem}
h1,h2,h3,p,li,label{letter-spacing:-.01em}
a{color:#8fb4ff;text-decoration:none} a:hover{color:#fff}

@keyframes up{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes grow{to{width:var(--w)}}
@keyframes blink{0%,49%{opacity:1}50%,100%{opacity:0}}
@keyframes typing{from{width:0}to{width:calc(var(--n)*1ch)}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(45,212,160,.5)}50%{box-shadow:0 0 0 6px rgba(45,212,160,0)}}
.fade{opacity:0;animation:up .75s var(--ease) forwards;animation-delay:calc(var(--i,0)*70ms)}

/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:.2rem;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:50;
background:rgba(0,0,0,.84);backdrop-filter:blur(14px);padding-top:.35rem;margin-bottom:.6rem}
.stTabs [data-baseweb="tab"]{height:46px;padding:0 16px;font-weight:600;font-size:.95rem;color:var(--muted);border-radius:10px 10px 0 0;
transition:color .25s var(--ease),background .25s var(--ease)}
.stTabs [data-baseweb="tab"]:hover{color:#fff;background:rgba(255,255,255,.05)}
.stTabs [aria-selected="true"]{color:#fff}
.stTabs [data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--blue),var(--violet));height:2px}
.stTabs [data-baseweb="tab-border"]{display:none}

/* brand bar */
.brand{display:flex;justify-content:space-between;align-items:center;padding:.2rem 0 .9rem}
.logo{font-size:1.05rem;font-weight:700;letter-spacing:-.02em;display:flex;align-items:center;gap:.55rem}
.logo i{display:grid;place-items:center;width:32px;height:32px;border-radius:9px;font-style:normal;
background:linear-gradient(135deg,var(--blue),var(--violet));box-shadow:0 6px 20px rgba(59,130,246,.35)}
.brand-r{display:flex;gap:.8rem;align-items:center;font-size:.82rem;color:var(--muted)}
.live{display:inline-flex;gap:.5rem;align-items:center;padding:.32rem .75rem;border:1px solid var(--line);border-radius:999px;background:var(--panel)}
.live b{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 2.2s infinite}
.ghlink{padding:.32rem .8rem;border:1px solid var(--line);border-radius:999px;transition:all .3s var(--ease);color:#cfd5e4}
.ghlink:hover{border-color:var(--blue);color:#fff;transform:translateY(-1px)}

/* hero */
.hero{display:grid;grid-template-columns:1.1fr 1fr;gap:2.2rem;align-items:center;padding:1.6rem 0 1.2rem}
.eyebrow{font-family:var(--mono);font-size:.74rem;font-weight:500;letter-spacing:.14em;color:var(--blue);margin-bottom:.9rem}
.hero h1{font-size:clamp(2.3rem,4.6vw,3.7rem);font-weight:800;line-height:1.04;letter-spacing:-.035em;margin:0;padding:0;color:var(--text)}
.grad{background:linear-gradient(92deg,#3b82f6 0%,#8b5cf6 55%,#ec4899 100%);-webkit-background-clip:text;background-clip:text;color:transparent}
.lead{color:var(--muted);font-size:1.06rem;line-height:1.65;margin:1.1rem 0 1.2rem;max-width:34rem}
.chips{display:flex;flex-wrap:wrap;gap:.5rem}
.chips span{font-size:.8rem;font-weight:500;color:#cfd5e4;padding:.34rem .8rem;border:1px solid var(--line);border-radius:999px;background:var(--panel);
transition:all .3s var(--ease)}
.chips span:hover{border-color:var(--blue);transform:translateY(-2px);color:#fff}

/* terminal */
.term{background:#0a0b10;border:1px solid var(--line);border-radius:16px;overflow:hidden;box-shadow:0 30px 80px rgba(59,130,246,.10);
transition:border-color .4s var(--ease),transform .4s var(--ease)}
.term:hover{border-color:var(--line-hi);transform:translateY(-3px)}
.term .bar{display:flex;align-items:center;gap:.4rem;padding:.65rem .9rem;background:#12141c;border-bottom:1px solid var(--line)}
.term .bar i{width:11px;height:11px;border-radius:50%;display:inline-block}
.term .bar span{margin-left:.6rem;color:var(--dim);font-family:var(--mono);font-size:.74rem}
.term .body{padding:1rem 1.15rem 1.2rem;font-family:var(--mono);font-size:.76rem;line-height:1.85;min-height:250px}
.tl{white-space:nowrap;overflow:hidden;text-overflow:clip}
.ps{color:var(--green)} .pm{color:var(--dim)}
.typed{display:inline-block;overflow:hidden;white-space:nowrap;vertical-align:bottom;width:0;box-sizing:content-box;padding-right:.6ch;color:#7fb2ff;
animation:typing var(--t) steps(var(--n),end) var(--d) forwards}
.tl.out{opacity:0;animation:up .45s var(--ease) var(--d) forwards;color:#e6e9f2;white-space:normal}
.tl.out b{color:var(--green);font-weight:600} .tl.out em{color:var(--dim);font-style:normal}
.cur{display:inline-block;width:.6ch;height:1.05em;background:#e5e7eb;vertical-align:text-bottom;opacity:0;animation:blink 1s steps(1) infinite;animation-delay:var(--d)}

/* cards, bento */
.bento{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin:.4rem 0 1.2rem}
.card{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:1.15rem 1.3rem;
transition:transform .35s var(--ease),border-color .35s var(--ease),box-shadow .35s var(--ease)}
.card:hover{transform:translateY(-4px);border-color:var(--line-hi);box-shadow:0 18px 50px rgba(59,130,246,.09)}
.s2{grid-column:span 2} .s3{grid-column:span 3} .s6{grid-column:span 6}
.num{font-size:2.15rem;font-weight:800;letter-spacing:-.035em;line-height:1.1;background:linear-gradient(95deg,#fff,#a9c3ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.lab{color:var(--muted);font-size:.84rem;margin-top:.25rem}
.kick{font-family:var(--mono);font-size:.7rem;letter-spacing:.12em;color:var(--dim);text-transform:uppercase;margin-bottom:.5rem}
.card p{color:var(--muted);font-size:.9rem;line-height:1.6;margin:.5rem 0 0}
.card b{color:var(--text)}
.dots{display:grid;grid-template-columns:repeat(20,1fr);gap:4px;margin-top:.8rem}
.dots i{aspect-ratio:1;border-radius:3px;background:#161926}
.dots i.on{background:linear-gradient(135deg,var(--blue),var(--violet));opacity:0;animation:up .5s var(--ease) forwards;animation-delay:calc(var(--k)*45ms + .4s)}
.cmp{margin-top:.8rem;display:grid;gap:.5rem}
.cmp div{display:grid;grid-template-columns:110px 1fr 64px;gap:.7rem;align-items:center;font-size:.8rem;color:var(--muted)}
.cmp span.t{height:8px;border-radius:99px;background:#161926;overflow:hidden}
.cmp span.t i{display:block;height:100%;width:0;border-radius:99px;animation:grow 1.1s var(--ease) .5s forwards}
.cmp code{font-family:var(--mono);color:var(--text);font-size:.8rem;text-align:right;white-space:nowrap}
.bug{display:flex;gap:1.2rem;align-items:center;flex-wrap:wrap}
.bug .big{font-family:var(--mono);font-size:1.5rem;font-weight:600}
.bug .old{color:var(--dim);text-decoration:line-through;text-decoration-color:var(--red)}
.bug .arrow{color:var(--blue)}
.explore{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-top:.2rem}
.explore .card{padding:1rem 1.1rem}
.explore h4{margin:0 0 .3rem;font-size:.95rem;font-weight:700;color:var(--text)}
.explore p{margin:0;font-size:.82rem}

/* headings */
.sec{font-size:1.75rem;font-weight:800;letter-spacing:-.03em;margin:.8rem 0 .15rem;color:var(--text)}
.sub{color:var(--muted);font-size:.95rem;margin-bottom:1rem;max-width:46rem;line-height:1.6}

/* leaderboard */
.lb{display:grid;gap:8px;margin:.6rem 0 .8rem}
.lb-row{display:grid;grid-template-columns:34px minmax(150px,1.1fr) 2.4fr 76px 110px;gap:14px;align-items:center;padding:.62rem .9rem;
background:var(--panel);border:1px solid var(--line);border-radius:12px;opacity:0;animation:up .55s var(--ease) forwards;animation-delay:calc(var(--i)*45ms);
transition:border-color .3s var(--ease),transform .3s var(--ease),background .3s var(--ease)}
.lb-row:hover{border-color:var(--line-hi);transform:translateX(4px);background:var(--panel2)}
.rk{font-family:var(--mono);color:var(--dim);font-size:.85rem;text-align:center}
.lb-row:nth-child(1) .rk{color:#ffd166;font-weight:700}.lb-row:nth-child(2) .rk{color:#cfd5e4;font-weight:700}.lb-row:nth-child(3) .rk{color:#e09f6e;font-weight:700}
.nm{font-weight:600;font-size:.95rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.nm em{font-style:normal;font-family:var(--mono);font-size:.76rem;color:var(--dim);margin-left:.5rem;font-weight:400}
.track{height:10px;border-radius:99px;background:#141725;overflow:hidden}
.fill{display:block;height:100%;width:0;border-radius:99px;background:linear-gradient(90deg,var(--blue),var(--violet));animation:grow 1s var(--ease) forwards;animation-delay:calc(var(--i)*45ms + .15s)}
.val{white-space:nowrap;min-width:4.4em;font-family:var(--mono);font-weight:600;font-size:.92rem;text-align:right;color:#cfe0ff}
.meta{font-size:.76rem;color:var(--dim);text-align:right;white-space:nowrap}
.tag{display:inline-block;font-size:.68rem;padding:.1rem .45rem;border:1px solid #4a3a1a;color:#e6b455;border-radius:6px;margin-left:.4rem}

/* scouting */
.hdr{display:flex;gap:1.1rem;align-items:center;padding:1.2rem 1.4rem}
.av{width:62px;height:62px;border-radius:18px;display:grid;place-items:center;font-size:1.35rem;font-weight:800;
background:linear-gradient(135deg,var(--blue),var(--violet));box-shadow:0 10px 30px rgba(59,130,246,.3);flex:none}
.hdr h2{margin:0;font-size:1.65rem;font-weight:800;letter-spacing:-.03em;display:flex;gap:.7rem;align-items:center;flex-wrap:wrap}
.pill{display:inline-block;font-size:.74rem;font-weight:600;letter-spacing:0;padding:.22rem .7rem;border-radius:999px;border:1px solid;white-space:nowrap}
.verdict{color:#c9cfdd;font-size:.95rem;margin-top:.35rem;line-height:1.6}
.verdict b{color:#fff}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:14px 0}
.kpi .num{font-size:1.9rem}
.pbar{height:6px;border-radius:99px;background:#161926;margin-top:.7rem;overflow:hidden}
.pbar i{display:block;height:100%;width:0;border-radius:99px;background:linear-gradient(90deg,var(--blue),var(--violet));animation:grow 1s var(--ease) .3s forwards}
.pt{font-size:.74rem;color:var(--dim);margin-top:.35rem}

/* tables */
.tbl{width:100%;border-collapse:separate;border-spacing:0 6px}
.tbl th{font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);text-align:left;padding:.2rem .9rem}
.tbl td{padding:.7rem .9rem;background:var(--panel);border-top:1px solid var(--line);border-bottom:1px solid var(--line);font-size:.9rem;transition:background .25s}
.tbl td:first-child{border-left:1px solid var(--line);border-radius:12px 0 0 12px;font-weight:600}
.tbl td:last-child{border-right:1px solid var(--line);border-radius:0 12px 12px 0}
.tbl tr:hover td{background:var(--panel2)}
.tbl td.n,.tbl th.n{text-align:right;font-family:var(--mono)}

/* tale of the tape */
.tape{display:grid;gap:8px;margin-top:.5rem}
.tp{display:grid;grid-template-columns:74px 1fr 170px 1fr 74px;gap:12px;align-items:center;padding:.62rem .9rem;background:var(--panel);border:1px solid var(--line);border-radius:12px;transition:border-color .3s var(--ease)}
.tp:hover{border-color:var(--line-hi)}
.tp .v{font-family:var(--mono);font-weight:600;font-size:.92rem;color:var(--dim)} .tp .v.w{color:#fff}
.tp .v.l{text-align:left} .tp .v.r{text-align:right}
.tp .lbl{text-align:center;font-size:.8rem;color:var(--muted)}
.tp .b{height:10px;border-radius:99px;background:#141725;display:flex;overflow:hidden}
.tp .b.l{justify-content:flex-end}
.tp .b i{display:block;height:100%;width:0;border-radius:99px;animation:grow 1s var(--ease) .2s forwards}
.tph{display:grid;grid-template-columns:1fr 1fr;gap:12px;font-weight:700;font-size:1.05rem;margin-bottom:.2rem}
.tph div:last-child{text-align:right}

/* style cards */
.scards{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-top:1rem}
.sc{border-top:3px solid var(--c)}
.sc h4{margin:.1rem 0 .15rem;font-size:1rem;font-weight:700;line-height:1.25}
.sc .cnt{font-family:var(--mono);font-size:.74rem;color:var(--dim)}
.trait{display:inline-block;font-size:.74rem;padding:.2rem .55rem;margin:.25rem .25rem 0 0;border-radius:7px;background:#11141d;border:1px solid var(--line);color:#cfd5e4}
.trait.up{color:#9be7c7}.trait.dn{color:#ffa3b1}
.sc .ex{margin-top:.8rem;font-size:.78rem;color:var(--muted);line-height:1.5}

/* method */
.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:.6rem 0 1rem;align-items:stretch}
.step{position:relative}
.step .no{font-family:var(--mono);font-size:.72rem;color:var(--blue)}
.step h4{margin:.25rem 0 .3rem;font-size:.95rem}
.step p{font-size:.8rem !important}
.step:not(:last-child)::after{content:"→";position:absolute;right:-13px;top:50%;transform:translateY(-50%);color:var(--blue);z-index:2;font-size:1rem}
.good{color:#9be7c7}.bad{color:#ffa3b1}
.foot{margin-top:2.6rem;padding-top:1.2rem;border-top:1px solid var(--line);color:var(--dim);font-size:.82rem;text-align:center;line-height:1.7}

/* streamlit widgets and charts */
[data-testid="stPlotlyChart"]{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:6px 6px 0;transition:border-color .35s var(--ease)}
[data-testid="stPlotlyChart"]:hover{border-color:var(--line-hi)}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:var(--panel);margin-bottom:.7rem}
[data-testid="stExpander"] summary{font-weight:600}
[data-testid="stExpander"] summary:hover{color:#fff}
label p{color:var(--muted) !important;font-size:.84rem !important;font-weight:500 !important}
[data-baseweb="select"]>div{background:var(--panel) !important;border-color:var(--line) !important;border-radius:12px !important;transition:border-color .25s}
[data-baseweb="select"]>div:hover{border-color:var(--line-hi) !important}

@media (max-width:900px){
.hero{grid-template-columns:1fr}.bento{grid-template-columns:repeat(2,1fr)}.s2,.s3,.s6{grid-column:span 2}
.explore,.scards,.flow{grid-template-columns:1fr 1fr}.kpis{grid-template-columns:1fr 1fr}.step:not(:last-child)::after{display:none}
.lb-row{grid-template-columns:30px 1fr 70px}.lb-row .track,.lb-row .meta{display:none}
.tp{grid-template-columns:56px 1fr 56px}.tp .b,.tp .lbl{display:none}.tp .lbl{display:block;grid-column:1/-1;order:-1}
}
@media (prefers-reduced-motion:reduce){
*{animation:none !important;transition:none !important}
.fade,.lb-row,.tl.out,.dots i.on{opacity:1 !important}
.typed{width:calc(var(--n)*1ch) !important}
.fill,.cmp span.t i,.pbar i,.tp .b i{width:var(--w) !important}
.cur{opacity:1 !important}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
def H(markup: str) -> None:
    """Render HTML. Whitespace between lines is removed so Markdown never treats it as a code block."""
    st.markdown("".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def ordinal(n: float) -> str:
    n = int(round(n))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def initials(name: str) -> str:
    parts = [p for p in re.split(r"[ .]+", name) if p]
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else name[:2].upper()


def pill(style: str) -> str:
    c = STYLE_COLORS.get(style, "#9aa1b2")
    return f'<span class="pill" style="color:{c};border-color:{c}55;background:{c}14">{esc(style)}</span>'


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
        margin=dict(l=20, r=20, t=24, b=20),
        font=dict(family="Inter, sans-serif", size=13, color="#cfd5e4"),
        hoverlabel=dict(bgcolor="#12141c", bordercolor="#2c3350", font=dict(family="Inter, sans-serif", size=13, color="#f4f5f7")),
    )
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


@st.cache_data
def load():
    rae = pd.read_parquet("data/app/batter_rae.parquet")
    cl = pd.read_parquet("data/app/batter_clusters.parquet").reset_index(drop=True)
    cl["boundary_pct"] = cl["four_pct"] + cl["six_pct"]
    return rae, cl


rae, cl = load()
rae = rae.copy()
rae["year"] = rae["year"].astype(int)
PLAYERS_WITH_PROFILE = set(cl["player"])

AXES = {  # label -> (column, higher_is_better)
    "Scoring": ("rel_sr", True), "Boundaries": ("boundary_pct", True), "Sixes": ("six_pct", True),
    "Dot control": ("dot_pct", False), "Survival": ("balls_per_out", True),
    "Powerplay": ("pp_share", True), "Death overs": ("death_share", True),
}
SKILLS = ["Scoring", "Boundaries", "Sixes", "Dot control", "Survival"]
pct = pd.DataFrame(index=cl.index)
for label, (col, hib) in AXES.items():
    r = cl[col].rank(pct=True) * 100
    pct[label] = r if hib else 100 - r


def radar(names, height=420):
    fig = go.Figure()
    for name, color in zip(names, [BLUE, PINK]):
        idx = cl.index[cl["player"] == name][0]
        vals = pct.loc[idx].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=list(AXES) + [list(AXES)[0]], fill="toself", name=name,
            line=dict(color=color, width=2.5), fillcolor=color.replace("#", "#") + "30",
            hovertemplate="%{theta}: %{r:.0f}th percentile<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(range=[0, 100], showticklabels=False, gridcolor="#1f2333", linecolor="#1f2333"),
                   angularaxis=dict(tickfont=dict(size=13, color="#cfd5e4"), gridcolor="#1f2333", linecolor="#1f2333")),
        showlegend=len(names) > 1, legend=dict(orientation="h", y=-0.06, x=0.5, xanchor="center"),
    )
    style_fig(fig, height)
    fig.update_layout(margin=dict(l=70, r=70, t=30, b=40))
    return fig


# ----------------------------------------------------------------------------------------
# Brand bar
# ----------------------------------------------------------------------------------------
y0, y1 = int(rae["year"].min()), int(rae["year"].max())
H(f"""
<div class="brand fade">
<div class="logo"><i>🏏</i>Cricket Scout AI</div>
<div class="brand-r"><span class="live"><b></b>Real data · Cricsheet IPL {y0}–{y1}</span>
<a class="ghlink" href="{GITHUB}" target="_blank" rel="noopener">GitHub ↗</a></div>
</div>
""")

tab_over, tab_lb, tab_scout, tab_cmp, tab_map, tab_how = st.tabs(
    ["Overview", "Leaderboard", "Scouting report", "Compare", "Style map", "How it works"]
)

# ========================================================================================
# Overview
# ========================================================================================
with tab_over:
    top = rae[rae["balls"] >= 250].sort_values("rae_per100", ascending=False).iloc[0]
    script = [
        ("cmd", "scout leaderboard --top 1 --min-balls 250"),
        ("out", f"<b>{esc(top['player'])} ({int(top['year'])})</b> +{top['rae_per100']:.1f} <em>runs per 100 balls above expected</em>"),
        ("cmd", "scout check --duplicates"),
        ("out", "<b>0</b> duplicate rows · 237.9 balls per match <b>✓</b>"),
        ("cmd", "scout reliability --season-to-season"),
        ("out", "r = <b>0.35</b> <em>→ only ~12% of a batter's form carries over</em>"),
    ]
    t, rows = 0.5, []
    for kind, text in script:
        if kind == "cmd":
            n = len(text)
            dur = round(n * 0.03, 2)
            rows.append(
                f'<div class="tl"><span class="ps">scout@ipl</span> <span class="pm">~ %</span> '
                f'<span class="typed" style="--n:{n};--t:{dur}s;--d:{t:.2f}s">{esc(text)}</span></div>')
            t += dur + 0.3
        else:
            rows.append(f'<div class="tl out" style="--d:{t:.2f}s">{text}</div>')
            t += 0.55
    rows.append(f'<div class="tl"><span class="ps">scout@ipl</span> <span class="pm">~ %</span> <span class="cur" style="--d:{t:.2f}s"></span></div>')

    H(f"""
<div class="hero">
<div class="fade" style="--i:1">
<div class="eyebrow">IPL BATTING ANALYTICS</div>
<h1>Find real talent,<br><span class="grad">not lucky seasons.</span></h1>
<p class="lead">A scouting system built on every IPL delivery since {y0}. It adjusts for match situation and era, and it is honest about how noisy cricket really is.</p>
<div class="chips"><span>Expected-runs model</span><span>K-Means batting styles</span><span>Walk-forward validation</span></div>
</div>
<div class="term fade" style="--i:3"><div class="bar"><i style="background:#ff5f57"></i><i style="background:#febc2e"></i><i style="background:#28c840"></i><span>scout: zsh</span></div>
<div class="body">{''.join(rows)}</div></div>
</div>
""")

    cells = "".join('<i class="on" style="--k:%d"></i>' % i if i < 25 else "<i></i>" for i in range(200))

    dots_html = '<div class="dots">' + cells + "</div>"
    H(f"""
<div class="bento">
<div class="card s2 fade" style="--i:2"><div class="num">1,243</div><div class="lab">IPL matches analysed</div></div>
<div class="card s2 fade" style="--i:3"><div class="num">295,732</div><div class="lab">Deliveries modelled</div></div>
<div class="card s2 fade" style="--i:4"><div class="num">{len(cl)}</div><div class="lab">Batters profiled into {cl['style'].nunique()} styles</div></div>
<div class="card s3 fade" style="--i:5"><div class="kick">Reliability</div>
<div class="num">r = 0.35</div><div class="lab">Season-to-season correlation of batting performance</div>
{dots_html}<p>Each square is 0.5% of variance. About <b>12–13%</b> of a batter's form carries into the next year; the rest is luck.</p></div>
<div class="card s3 fade" style="--i:6"><div class="kick">Honest evaluation</div>
<div class="num">1%</div><div class="lab">Gain over a "guess the league average" baseline</div>
<div class="cmp"><div>Gradient boosting<span class="t"><i style="--w:99%;background:linear-gradient(90deg,#3b82f6,#8b5cf6)"></i></span><code>0.1046</code></div>
<div>League average<span class="t"><i style="--w:100%;background:#3a4057"></i></span><code>0.1057</code></div></div>
<p>Mean absolute error across 13 walk-forward seasons (lower is better). The model barely beats the baseline, and that is reported as it is.</p></div>
<div class="card s6 fade" style="--i:7"><div class="kick">Data quality</div>
<div class="bug"><span class="big old">591,464 rows</span><span class="big arrow">→</span><span class="big">295,732 rows</span>
<span class="lab" style="flex:1;min-width:260px;margin:0">A combined file in the download silently doubled every delivery. Caught by checking one player's career total against reality, then fixed with an automated validation step.</span></div></div>
</div>
""")

    H('<div class="sec fade" style="--i:1">Explore the dashboard</div><div class="sub">Use the tabs above. Everything here is computed from the real ball-by-ball data.</div>')
    H("""
<div class="explore">
<div class="card fade" style="--i:1"><h4>Leaderboard</h4><p>Who beat the match situation, by season.</p></div>
<div class="card fade" style="--i:2"><h4>Scouting report</h4><p>Skill profile, season history and similar batters.</p></div>
<div class="card fade" style="--i:3"><h4>Compare</h4><p>Two batters, head to head.</p></div>
<div class="card fade" style="--i:4"><h4>Style map</h4><p>Five batting archetypes in one picture.</p></div>
<div class="card fade" style="--i:5"><h4>How it works</h4><p>Method, validation and what did not work.</p></div>
</div>
""")

# ========================================================================================
# Leaderboard
# ========================================================================================
with tab_lb:
    H('<div class="sec">Who beat the situation?</div><div class="sub"><b>Runs above expected</b> compares what a batter scored with what a typical batter would score in the same situation: over, wickets lost and chase pressure. Higher means more value created per 100 balls.</div>')
    c1, c2, c3 = st.columns([1, 1.4, 1])
    season = c1.selectbox("Season", ["All seasons"] + sorted(rae["year"].unique().tolist(), reverse=True))
    min_balls = c2.slider("Minimum balls faced", 60, 500, 250, step=20)
    top_n = c3.selectbox("Show top", [10, 15, 20], index=0)

    view = rae[rae["balls"] >= min_balls]
    if season != "All seasons":
        view = view[view["year"] == season]
    top_rows = view.sort_values("rae_per100", ascending=False).head(top_n).reset_index(drop=True)
    if top_rows.empty:
        st.info("No batter meets that minimum for this season. Try lowering the minimum balls.")
    else:
        mx = float(top_rows["rae_per100"].max())
        items = []
        for i, r in top_rows.iterrows():
            w = max(4.0, r["rae_per100"] / mx * 100) if mx > 0 else 4
            tag = '<span class="tag">small sample</span>' if r["balls"] < 350 else ""
            items.append(
                f'<div class="lb-row" style="--i:{i};--w:{w:.1f}%"><span class="rk">{i + 1}</span>'
                f'<span class="nm">{esc(r["player"])}<em>{int(r["year"])}</em>{tag}</span>'
                f'<span class="track"><span class="fill"></span></span>'
                f'<span class="val">+{r["rae_per100"]:.1f}</span><span class="meta">{int(r["balls"])} balls · {int(r["runs"])} runs</span></div>')
        H(f'<div class="lb">{"".join(items)}</div>')
        st.caption("Values are runs above expected per 100 balls. \"Small sample\" marks fewer than 350 balls, where luck can dominate.")

# ========================================================================================
# Scouting report
# ========================================================================================
with tab_scout:
    names = sorted(cl["player"])
    player = st.selectbox("Pick a batter", names, index=names.index("V Kohli") if "V Kohli" in names else 0)
    row = cl[cl["player"] == player].iloc[0]
    p = pct.loc[row.name]
    hist = rae[(rae["player"] == player) & (rae["balls"] >= 60)].sort_values("year")
    skills = p[SKILLS].sort_values(ascending=False)

    H(f"""
<div class="card hdr fade"><div class="av">{esc(initials(player))}</div><div>
<h2>{esc(player)} {pill(row['style'])}</h2>
<div class="verdict">Strongest skill: <b>{skills.index[0].lower()}</b> ({ordinal(skills.iloc[0])} percentile). Weakest: <b>{skills.index[-1].lower()}</b> ({ordinal(skills.iloc[-1])} percentile). Based on {int(row['balls']):,} balls faced.</div>
</div></div>
""")

    kp = [
        (f"{row['rel_sr']:.2f}x", "Strike rate vs league", p["Scoring"], "scoring"),
        (f"{row['six_pct']:.1f}%", "Balls hit for six", p["Sixes"], "six-hitting"),
        (f"{row['dot_pct']:.1f}%", "Dot balls", p["Dot control"], "dot-ball control"),
        (f"{row['balls_per_out']:.0f}", "Balls per dismissal", p["Survival"], "survival"),
    ]
    cards = "".join(
        f'<div class="card kpi fade" style="--i:{i + 1}"><div class="num">{v}</div><div class="lab">{l}</div>'
        f'<div class="pbar"><i style="--w:{max(pc, 3):.0f}%"></i></div><div class="pt">{ordinal(pc)} percentile for {w}</div></div>'
        for i, (v, l, pc, w) in enumerate(kp))
    H(f'<div class="kpis">{cards}</div>')

    cl_, cr_ = st.columns([1, 1.15])
    with cl_:
        H('<div class="sec" style="font-size:1.15rem">Skill profile</div><div class="sub" style="margin-bottom:.4rem">Percentile among all profiled IPL batters. Further out is better.</div>')
        show(radar([player]))
    with cr_:
        H('<div class="sec" style="font-size:1.15rem">Season by season</div><div class="sub" style="margin-bottom:.4rem">Runs above expected per 100 balls. Single seasons are noisy: look for patterns, not one big bar.</div>')
        fig = go.Figure(go.Bar(
            x=hist["year"], y=hist["rae_per100"].round(1),
            marker=dict(color=[BLUE if v > 0 else PINK for v in hist["rae_per100"]], cornerradius=4),
            customdata=np.stack([hist["balls"], hist["runs"]], axis=-1),
            hovertemplate="<b>%{x}</b><br>%{y:+.1f} per 100 balls<br>%{customdata[0]} balls · %{customdata[1]} runs<extra></extra>",
        ))
        fig.add_hline(y=0, line_color="#3a4057", line_width=1)
        fig.update_xaxes(showgrid=False, tickmode="linear", dtick=2, linecolor="#1f2333")
        fig.update_yaxes(gridcolor="#141725", zeroline=False)
        show(style_fig(fig, 420))

    cols = [v[0] for v in AXES.values()]
    z = (cl[cols] - cl[cols].mean()) / cl[cols].std()
    dist = ((z - z.loc[row.name]) ** 2).sum(axis=1) ** 0.5
    sim = cl.assign(distance=dist).drop(index=row.name).nsmallest(5, "distance")
    H('<div class="sec" style="font-size:1.15rem;margin-top:1.4rem">Most similar batters</div><div class="sub" style="margin-bottom:.3rem">Nearest neighbours across all seven style features.</div>')
    body = "".join(
        f'<tr><td>{esc(r["player"])}</td><td>{pill(r["style"])}</td><td class="n">{r["rel_sr"]:.2f}x</td>'
        f'<td class="n">{r["six_pct"]:.1f}</td><td class="n">{r["dot_pct"]:.1f}</td><td class="n">{r["balls_per_out"]:.0f}</td></tr>'
        for _, r in sim.iterrows())
    H(f'<table class="tbl"><thead><tr><th>Batter</th><th>Style</th><th class="n">SR vs league</th><th class="n">Six %</th><th class="n">Dot %</th><th class="n">Balls/out</th></tr></thead><tbody>{body}</tbody></table>')

# ========================================================================================
# Compare
# ========================================================================================
with tab_cmp:
    H('<div class="sec">Head to head</div><div class="sub">Two batters across every feature the model uses. The brighter value wins that row.</div>')
    names = sorted(cl["player"])
    a, b = st.columns(2)
    p1 = a.selectbox("Batter A", names, index=names.index("V Kohli") if "V Kohli" in names else 0)
    p2 = b.selectbox("Batter B", names, index=names.index("MS Dhoni") if "MS Dhoni" in names else 1)
    ra, rb = cl[cl["player"] == p1].iloc[0], cl[cl["player"] == p2].iloc[0]

    l, r = st.columns([1, 1.1])
    with l:
        show(radar([p1, p2], 440))
    with r:
        metrics = [
            ("Strike rate vs league", "rel_sr", True, "{:.2f}x"), ("Fours %", "four_pct", True, "{:.1f}"),
            ("Sixes %", "six_pct", True, "{:.1f}"), ("Dot balls % (lower wins)", "dot_pct", False, "{:.1f}"),
            ("Balls per dismissal", "balls_per_out", True, "{:.1f}"),
            ("Powerplay share", "pp_share", None, "{:.0%}"), ("Death-over share", "death_share", None, "{:.0%}"),
        ]
        rows = []
        for label, col, hib, fmt in metrics:
            va, vb = float(ra[col]), float(rb[col])
            wa = wb = False
            if hib is True:
                wa, wb = va > vb, vb > va
            elif hib is False:
                wa, wb = va < vb, vb < va
            m = max(va, vb) or 1
            rows.append(
                f'<div class="tp"><span class="v l {"w" if wa else ""}">{fmt.format(va)}</span>'
                f'<span class="b l"><i style="--w:{va / m * 100:.0f}%;background:{BLUE}"></i></span>'
                f'<span class="lbl">{label}</span>'
                f'<span class="b"><i style="--w:{vb / m * 100:.0f}%;background:{PINK}"></i></span>'
                f'<span class="v r {"w" if wb else ""}">{fmt.format(vb)}</span></div>')
        H(f'<div class="tph"><div style="color:{BLUE}">{esc(p1)}</div><div style="color:{PINK}">{esc(p2)}</div></div><div class="tape">{"".join(rows)}</div>')
    H(f'<div style="margin-top:.8rem">{pill(ra["style"])} &nbsp;vs&nbsp; {pill(rb["style"])}</div>')

# ========================================================================================
# Style map
# ========================================================================================
with tab_map:
    H('<div class="sec">The batting style map</div><div class="sub">Every dot is a batter. <b>Closer together means a more similar playing style.</b> Dot size shows balls faced. Hover for details, and use the filter to isolate styles.</div>')
    all_styles = [s for s in STYLE_COLORS if s in set(cl["style"])]
    chosen = st.multiselect("Show styles", all_styles, default=all_styles)
    d = cl[cl["style"].isin(chosen)] if chosen else cl.iloc[0:0]

    fig = go.Figure()
    for s in chosen:
        g = d[d["style"] == s]
        fig.add_trace(go.Scatter(
            x=g["pc1"], y=g["pc2"], mode="markers", name=s,
            marker=dict(size=g["balls"], sizemode="area", sizeref=2.0 * cl["balls"].max() / (30.0 ** 2), sizemin=6,
                        color=STYLE_COLORS[s], opacity=0.85, line=dict(width=1, color="#000")),
            customdata=np.stack([g["player"], g["rel_sr"], g["six_pct"], g["balls"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>" + s + "<br>SR vs league %{customdata[1]:.2f}x · six % %{customdata[2]:.1f}<br>%{customdata[3]:,} balls<extra></extra>",
        ))
    if len(d):
        cand = list(d.nlargest(9, "balls").index)
        for s in chosen:
            g = d[d["style"] == s]
            cand += list((g["pc1"] ** 2 + g["pc2"] ** 2).nlargest(1).index)
        picked = []
        for i in dict.fromkeys(cand):
            x, y = d.loc[i, "pc1"], d.loc[i, "pc2"]
            if all(abs(x - d.loc[j, "pc1"]) > 0.55 or abs(y - d.loc[j, "pc2"]) > 0.32 for j in picked):
                picked.append(i)
        fig.add_trace(go.Scatter(
            x=d.loc[picked, "pc1"], y=d.loc[picked, "pc2"], mode="text", text=d.loc[picked, "player"],
            textposition="top center", textfont=dict(size=12, color="#e5e7eb"), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(legend=dict(orientation="h", y=-0.04, x=0.5, xanchor="center", font=dict(size=12)))
    show(style_fig(fig, 580))
    st.caption("The styles overlap at the edges: batting is a spectrum, so these are useful descriptions, not hard categories (silhouette score 0.22).")

    overall_mean, overall_std = cl[FEATS].mean(), cl[FEATS].std()
    cards = []
    for i, s in enumerate(all_styles):
        g = cl[cl["style"] == s]
        zs = ((g[FEATS].mean() - overall_mean) / overall_std)
        order = zs.abs().sort_values(ascending=False).index[:3]
        traits = "".join(
            f'<span class="trait {"up" if zs[f] > 0 else "dn"}">{"↑" if zs[f] > 0 else "↓"} {FEAT_NAMES[f]}</span>' for f in order)
        ex = ", ".join(esc(n) for n in g.nlargest(3, "balls")["player"])
        cards.append(
            f'<div class="card sc fade" style="--c:{STYLE_COLORS[s]};--i:{i}"><h4>{esc(s)}</h4><div class="cnt">{len(g)} batters</div>'
            f'<div>{traits}</div><div class="ex">Most experienced: {ex}</div></div>')
    H(f'<div class="scards">{"".join(cards)}</div>')

# ========================================================================================
# How it works
# ========================================================================================
with tab_how:
    H("""
<div class="sec">How it works</div>
<div class="sub">Scouts judge batters on strike rate and average, but one season is mostly luck, and a 130 strike rate in 2012 is not the same as a 130 today. This project separates skill from situation and tests honestly.</div>
<div class="flow">
<div class="card step fade" style="--i:1"><div class="no">01</div><h4>Ball-by-ball data</h4><p>1,243 IPL matches from Cricsheet.</p></div>
<div class="card step fade" style="--i:2"><div class="no">02</div><h4>Validate and clean</h4><p>Automated checks stop bad data before modelling.</p></div>
<div class="card step fade" style="--i:3"><div class="no">03</div><h4>Expected-runs model</h4><p>Gradient boosting on match situation only.</p></div>
<div class="card step fade" style="--i:4"><div class="no">04</div><h4>Style clustering</h4><p>K-Means on seven era-adjusted features.</p></div>
<div class="card step fade" style="--i:5"><div class="no">05</div><h4>This dashboard</h4><p>Streamlit and Plotly, on real outputs.</p></div>
</div>
<div class="bento">
<div class="card s3 fade" style="--i:1"><div class="kick">Expected-runs model</div><div class="num">3.1%</div><div class="lab">Lower error than predicting the average on later, unseen seasons</div></div>
<div class="card s3 fade" style="--i:2"><div class="kick">Walk-forward validation</div><div class="num">6 of 13</div><div class="lab">Seasons where the next-season model beat the best simple baseline</div></div>
</div>
""")

    with st.expander("Walk-forward evaluation, 2014 to 2026", expanded=True):
        years = list(range(2014, 2027))
        model = [.084, .099, .102, .118, .101, .103, .122, .111, .105, .111, .097, .097, .111]
        league = [.092, .091, .107, .104, .101, .110, .118, .111, .105, .108, .110, .101, .120]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=league, name="Guess the league average", mode="lines+markers",
                                 line=dict(color="#6b7285", dash="dash", width=2), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=years, y=model, name="Gradient-boosting model", mode="lines+markers",
                                 line=dict(color=BLUE, width=3), marker=dict(size=8)))
        fig.update_xaxes(showgrid=False, dtick=1, linecolor="#1f2333")
        fig.update_yaxes(gridcolor="#141725", title="Mean absolute error (lower is better)", range=[0.075, 0.13])
        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0))
        show(style_fig(fig, 380))
        st.caption("Each year the model trains only on earlier seasons and predicts that year. Average error: model 0.1046 vs league-average 0.1057. The gap is about 1%, within noise.")

    with st.expander("Accuracy depends on how many balls a batter faced"):
        H("""
<table class="tbl"><thead><tr><th>Balls in target season</th><th class="n">Player-seasons</th><th class="n">Model error</th><th class="n">League-average error</th></tr></thead><tbody>
<tr><td>60–100</td><td class="n">133</td><td class="n">0.145</td><td class="n">0.144</td></tr>
<tr><td>100–200</td><td class="n">265</td><td class="n">0.111</td><td class="n">0.110</td></tr>
<tr><td>200–350</td><td class="n">265</td><td class="n">0.088</td><td class="n">0.094</td></tr>
<tr><td>350+</td><td class="n">79</td><td class="n">0.073</td><td class="n">0.069</td></tr></tbody></table>
""")
        st.caption("Short seasons are mostly luck for everyone. The model only edges ahead in the 200–350 range.")

    with st.expander("The data bug I caught"):
        st.markdown(
            "Career totals looked wrong: one batter had roughly double his real runs, and matches averaged about 475 deliveries instead of about 240. "
            "The cause was a combined file in the download that was loaded alongside the individual match files, so every delivery appeared twice. "
            "I now exclude it and run automated checks (no duplicate rows, balls per match between 200 and 300) before any modelling."
        )

    H("""
<div class="sec" style="font-size:1.3rem;margin-top:1.2rem">What I found, including what did not work</div>
<div class="bento">
<div class="card s2"><div class="kick good">Worked</div><p><b>Expected runs</b> separates situation from skill, so a death-over finisher and a powerplay opener are compared fairly. Top results match known explosive batters.</p></div>
<div class="card s2"><div class="kick bad">Did not work</div><p>Predicting next season's strike rate beat the baseline by only <b>1%</b>. Season-to-season persistence is just <b>r = 0.35</b>.</p></div>
<div class="card s2"><div class="kick">Neutral</div><p>Runs above expected (r = 0.355) was about as persistent as strike rate (r = 0.343). It makes comparisons fairer, not predictions easier.</p></div>
</div>
<div class="chips" style="margin-top:.8rem"><span>Python</span><span>pandas</span><span>scikit-learn</span><span>Plotly</span><span>Streamlit</span></div>
""")

H(f'<div class="foot">Built by <b>Anoushka Yadav</b> · <a href="{GITHUB}" target="_blank" rel="noopener">github.com/anuu-yd9/cricket-scout-ai</a><br>Ball-by-ball data from Cricsheet. Independent analysis, not affiliated with the IPL.</div>')
