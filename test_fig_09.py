import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set global font to standard sans-serif
plt.rcParams['font.family'] = 'sans-serif'

# Colors
BG_COLOR = "#080c16"          
TEXT_MAIN = "#f8fafc"         
WARNING_ORANGE = "#f59e0b"    
CYAN_GLOW = "#22d3ee"         
EMERALD = "#10b981"           

def create_figure():
    fig, ax = plt.subplots(figsize=(24, 13.5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax

def draw_title(ax, text):
    ax.text(0.5, 0.92, text, ha='center', va='top', fontsize=44, weight='bold', color=TEXT_MAIN, 
            bbox=dict(facecolor=BG_COLOR, edgecolor='none', pad=0, alpha=0.9))

fig, ax = create_figure()
draw_title(ax, "ERROR ANALYSIS: CONSERVATIVE DOWNGRADES")

labels = ["Severe -> Mild\n(Downgrade)", "Mild -> Severe\n(Upgrade)", "Severe -> Healthy\n(Missed Danger)"]
values = [2, 1, 0]
colors = [WARNING_ORANGE, CYAN_GLOW, EMERALD]

ax_bar = fig.add_axes([0.2, 0.32, 0.6, 0.48])
ax_bar.set_facecolor(BG_COLOR)
bars = ax_bar.bar(labels, values, color=colors, width=0.4)
ax_bar.set_ylabel("Number of Occurrences", color=TEXT_MAIN, fontsize=22)
ax_bar.tick_params(colors=TEXT_MAIN, labelsize=20)
for spine in ax_bar.spines.values():
    spine.set_color("#334155")
    spine.set_linewidth(2)
    
ax.text(0.5, 0.10, "Most errors are conservative downgrades in visually ambiguous cases.\nZero 'Severe -> Healthy' errors demonstrate safe decision-support behavior.", 
        ha='center', va='center', color=CYAN_GLOW, fontsize=26, style='italic', linespacing=1.6)

plt.savefig("test_fig_09.png")
print("Test figure saved.")
