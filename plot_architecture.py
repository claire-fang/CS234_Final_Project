"""Generate a publication-quality model architecture figure (CS top-venue style).

Shows:
  (a) MDP formulation: state → policy → action → environment → next state
  (b) RetrievalSelector dual-path architecture

Usage:
    python plot_architecture.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'STIX'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.linewidth': 0.6,
    'figure.dpi': 200,
})

# Colors (muted academic palette)
C_BLUE   = '#3C78A8'
C_ORANGE = '#D4782F'
C_GREEN  = '#4A8C3F'
C_RED    = '#B83B3B'
C_PURPLE = '#7B5EA7'
C_GRAY   = '#888888'
C_LGRAY  = '#F2F2F2'
C_BG     = '#FAFAFA'
C_WHITE  = '#FFFFFF'
C_DARK   = '#2C2C2C'
C_LBLUE  = '#D6E8F5'
C_LORANGE = '#FDEBD0'
C_LGREEN = '#D5E8D4'
C_LPURPLE = '#E8DFF0'

# ── Helpers ────────────────────────────────────────────────────────────

def _rounded_box(ax, xy, w, h, text, fc=C_WHITE, ec=C_DARK, lw=0.8,
                 fontsize=8, text_color=C_DARK, bold=False, alpha=1.0,
                 ha='center', va='center', radius=0.02, zorder=3,
                 subtext=None, subsize=6.5):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw,
                         alpha=alpha, zorder=zorder,
                         transform=ax.transData)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    if subtext:
        cy += 0.012
    weight = 'bold' if bold else 'normal'
    ax.text(cx, cy, text, ha=ha, va=va, fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=zorder + 1,
            transform=ax.transData)
    if subtext:
        ax.text(cx, cy - 0.028, subtext, ha='center', va='center',
                fontsize=subsize, color=C_GRAY, style='italic',
                zorder=zorder + 1, transform=ax.transData)
    return box


def _arrow(ax, x1, y1, x2, y2, color=C_DARK, lw=0.8, style='-|>',
           connectionstyle='arc3,rad=0', zorder=2, linestyle='-'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, linewidth=lw,
        connectionstyle=connectionstyle, zorder=zorder,
        linestyle=linestyle, mutation_scale=10,
        transform=ax.transData,
    )
    ax.add_patch(arrow)
    return arrow


def _bracket_text(ax, x, y, text, fontsize=7, color=C_GRAY):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, transform=ax.transData)


# ── Main Figure ────────────────────────────────────────────────────────

fig = plt.figure(figsize=(10, 5.5))

# Two panels: left = MDP, right = architecture
ax_left = fig.add_axes([0.01, 0.02, 0.42, 0.90])
ax_right = fig.add_axes([0.47, 0.02, 0.52, 0.90])

for ax in [ax_left, ax_right]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

# Panel labels
ax_left.text(0.02, 0.97, '(a) Sequential Retrieval as MDP',
             fontsize=10.5, fontweight='bold', color=C_DARK, va='top')
ax_right.text(0.02, 0.97, '(b) Policy Network: RetrievalSelector',
              fontsize=10.5, fontweight='bold', color=C_DARK, va='top')

# ══════════════════════════════════════════════════════════════════════
#  Panel (a): MDP diagram
# ══════════════════════════════════════════════════════════════════════

# State box
_rounded_box(ax_left, (0.05, 0.72), 0.42, 0.12,
             r'State $s_t$', fc=C_LBLUE, ec=C_BLUE, fontsize=9, bold=True,
             subtext='question, paragraphs, read set, context', subsize=6)

# Feature extraction
_rounded_box(ax_left, (0.55, 0.72), 0.38, 0.12,
             'Feature\nExtraction', fc=C_LGRAY, ec=C_GRAY, fontsize=8,
             subtext=r'$\phi(s_t) \in \mathbb{R}^{614}$', subsize=6.5)

# Arrow: state → feature
_arrow(ax_left, 0.47, 0.78, 0.55, 0.78, color=C_BLUE)

# Policy box
_rounded_box(ax_left, (0.25, 0.50), 0.50, 0.12,
             r'Policy $\pi_\theta(a_t | s_t)$', fc=C_LPURPLE, ec=C_PURPLE,
             fontsize=9, bold=True,
             subtext='RetrievalSelector (~200K params)', subsize=6)

# Arrow: feature → policy
_arrow(ax_left, 0.74, 0.72, 0.60, 0.62, color=C_PURPLE,
       connectionstyle='arc3,rad=-0.15')

# Action box
_rounded_box(ax_left, (0.05, 0.28), 0.42, 0.12,
             r'Action $a_t$', fc=C_LORANGE, ec=C_ORANGE, fontsize=9, bold=True,
             subtext=r'read$_i\;(i{=}0..9)$ or answer', subsize=6.5)

# Arrow: policy → action
_arrow(ax_left, 0.40, 0.50, 0.26, 0.40, color=C_PURPLE,
       connectionstyle='arc3,rad=0.15')

# Reward box
_rounded_box(ax_left, (0.55, 0.28), 0.38, 0.12,
             r'Reward $r_t$', fc=C_LGREEN, ec=C_GREEN, fontsize=9, bold=True,
             subtext='supporting/distractor/stop', subsize=6)

# Arrow: action → reward
_arrow(ax_left, 0.47, 0.34, 0.55, 0.34, color=C_ORANGE)

# Next state / loop arrow
_rounded_box(ax_left, (0.25, 0.06), 0.50, 0.10,
             r'Environment: $s_{t+1} = \mathcal{T}(s_t, a_t)$',
             fc=C_LGRAY, ec=C_GRAY, fontsize=7.5,
             subtext='update read set, context (budget K=5)', subsize=5.5)

# Arrow: action → environment
_arrow(ax_left, 0.26, 0.28, 0.40, 0.16, color=C_DARK,
       connectionstyle='arc3,rad=0.15')

# Arrow: environment → state (loop back)
_arrow(ax_left, 0.25, 0.11, 0.05, 0.11, color=C_DARK,
       connectionstyle='arc3,rad=0', style='-')
_arrow(ax_left, 0.05, 0.11, 0.05, 0.72, color=C_DARK,
       connectionstyle='arc3,rad=0', style='-')
_arrow(ax_left, 0.05, 0.70, 0.06, 0.72, color=C_DARK,
       style='-|>')

# Budget label
ax_left.text(0.02, 0.42, r'$t \leq K{=}5$', fontsize=7, color=C_GRAY,
             rotation=90, ha='center', va='center')


# ══════════════════════════════════════════════════════════════════════
#  Panel (b): RetrievalSelector Architecture
# ══════════════════════════════════════════════════════════════════════

# ── Input features ──
_rounded_box(ax_right, (0.10, 0.82), 0.80, 0.10,
             r'Input Features $\phi(s_t) \in \mathbb{R}^{614}$',
             fc=C_LBLUE, ec=C_BLUE, fontsize=8.5, bold=True)

# Split labels
ax_right.text(0.30, 0.78, r'ST embedding (512-d)', fontsize=6,
              color=C_GRAY, ha='center')
ax_right.text(0.50, 0.78, '+', fontsize=7, color=C_GRAY, ha='center')
ax_right.text(0.70, 0.78, r'structured (102-d)', fontsize=6,
              color=C_GRAY, ha='center')

# Split arrows
_arrow(ax_right, 0.30, 0.82, 0.30, 0.74, color=C_BLUE)
_arrow(ax_right, 0.70, 0.82, 0.70, 0.74, color=C_BLUE)

# ── Path A: Per-paragraph scoring ──
_rounded_box(ax_right, (0.02, 0.58), 0.40, 0.15,
             'Path A: Para Scorer', fc=C_LORANGE, ec=C_ORANGE,
             fontsize=8, bold=True)
ax_right.text(0.22, 0.67, r'per-para feats $(B, 10, 10)$', fontsize=5.5,
              color=C_GRAY, ha='center')
ax_right.text(0.22, 0.635, r'MLP: $10 \!\to\! 8 \!\to\! 1$', fontsize=6.5,
              color=C_ORANGE, ha='center', fontweight='bold')
ax_right.text(0.22, 0.605, r'→ para scores $(B, 10)$', fontsize=5.5,
              color=C_GRAY, ha='center')

# Path A input arrow
_arrow(ax_right, 0.30, 0.78, 0.22, 0.73, color=C_ORANGE,
       connectionstyle='arc3,rad=0.1')

# Frozen label
ax_right.text(0.22, 0.555, 'FROZEN in PPO', fontsize=5.5,
              color=C_RED, ha='center', style='italic')

# ── Path B: Context pathway ──
_rounded_box(ax_right, (0.52, 0.53), 0.45, 0.20,
             'Path B: Context Pathway', fc=C_LPURPLE, ec=C_PURPLE,
             fontsize=8, bold=True)
ax_right.text(0.745, 0.685, r'global feats → Linear → LN → ReLU', fontsize=5.5,
              color=C_GRAY, ha='center')
ax_right.text(0.745, 0.655, r'→ ResBlock → LN → ReLU → Dropout(0.15)',
              fontsize=5.5, color=C_GRAY, ha='center')
ax_right.text(0.745, 0.62, r'ctx $\to$ para mod $(B, 10)$', fontsize=6,
              color=C_PURPLE, ha='center', fontweight='bold')
ax_right.text(0.745, 0.59, r'ctx $\to$ answer logit $(B, 1)$', fontsize=6,
              color=C_PURPLE, ha='center', fontweight='bold')
ax_right.text(0.745, 0.56, r'ctx $\to$ value $\hat{V}(s)$ $(B, 1)$', fontsize=6,
              color=C_GREEN, ha='center', fontweight='bold')

# Path B input arrow
_arrow(ax_right, 0.70, 0.78, 0.745, 0.73, color=C_PURPLE,
       connectionstyle='arc3,rad=-0.1')

# ── Combination ──
_rounded_box(ax_right, (0.10, 0.38), 0.80, 0.10,
             '', fc=C_LGREEN, ec=C_GREEN, fontsize=8.5, bold=True)
ax_right.text(0.50, 0.44,
              r'logits $= [\;\alpha \cdot$ para_scores $+ (1{-}\alpha) \cdot$'
              r' ctx_mod$,\;$ answer_logit $]$',
              fontsize=7, ha='center', va='center', color=C_DARK)
ax_right.text(0.50, 0.395,
              r'$\alpha = \sigma(\mathrm{learnable}),\; \mathrm{init} \approx 0.62$',
              fontsize=6, ha='center', va='center', color=C_GRAY)

# Arrows into combination
_arrow(ax_right, 0.22, 0.555, 0.30, 0.48, color=C_ORANGE,
       connectionstyle='arc3,rad=0.1')
_arrow(ax_right, 0.745, 0.53, 0.65, 0.48, color=C_PURPLE,
       connectionstyle='arc3,rad=-0.1')

# ── Output: Softmax → Action ──
_rounded_box(ax_right, (0.15, 0.22), 0.30, 0.10,
             r'$\pi_\theta(a|s) = \mathrm{softmax}(\mathrm{logits})$',
             fc=C_LORANGE, ec=C_ORANGE, fontsize=7, bold=False)
_arrow(ax_right, 0.35, 0.38, 0.30, 0.32, color=C_DARK)

_rounded_box(ax_right, (0.55, 0.22), 0.30, 0.10,
             r'$\hat{V}(s_t)$', fc=C_LGREEN, ec=C_GREEN,
             fontsize=8, bold=True, subtext='value baseline', subsize=5.5)
_arrow(ax_right, 0.65, 0.38, 0.70, 0.32, color=C_GREEN)

# ── Action outputs ──
actions_x = [0.15, 0.23, 0.31, 0.39]
action_labels = [r'read$_0$', r'read$_1$', '...', r'read$_9$']
for i, (xp, lab) in enumerate(zip(actions_x, action_labels)):
    ax_right.text(xp + 0.03, 0.17, lab, fontsize=6, ha='center',
                  va='center', color=C_ORANGE,
                  bbox=dict(boxstyle='round,pad=0.15', facecolor=C_WHITE,
                            edgecolor=C_ORANGE, linewidth=0.5, alpha=0.9))

ax_right.text(0.50, 0.17, r'answer', fontsize=6, ha='center', va='center',
              color=C_RED,
              bbox=dict(boxstyle='round,pad=0.15', facecolor=C_WHITE,
                        edgecolor=C_RED, linewidth=0.5, alpha=0.9))

# Arrows from softmax to actions
for xp in [0.18, 0.26, 0.34, 0.42]:
    _arrow(ax_right, 0.30, 0.22, xp, 0.19, color=C_ORANGE, lw=0.5)
_arrow(ax_right, 0.30, 0.22, 0.50, 0.19, color=C_RED, lw=0.5)

# ── Training label ──
ax_right.text(0.50, 0.06,
              r'Training: BC (cross-entropy) $\to$ PPO (clipped surrogate + adaptive KL)',
              fontsize=7, ha='center', va='center', color=C_DARK,
              bbox=dict(boxstyle='round,pad=0.3', facecolor=C_LGRAY,
                        edgecolor=C_GRAY, linewidth=0.5))

# ── Save ──
fig.savefig('checkpoints_blind/model_architecture.png', dpi=200,
            bbox_inches='tight', facecolor=C_WHITE, edgecolor='none')
fig.savefig('checkpoints_blind/model_architecture.pdf',
            bbox_inches='tight', facecolor=C_WHITE, edgecolor='none')
plt.close()
print('Saved model_architecture.png + .pdf')
