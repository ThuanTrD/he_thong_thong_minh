import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Set global font to standard sans-serif
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'Helvetica', 'sans-serif']

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# PREMIUM COLOR PALETTE & AESTHETICS
# ---------------------------------------------------------
BG_COLOR = "#080c16"          # Deep navy blue/black
GRID_COLOR = "#1e293b"        # Subtle grid
BOX_BG = "#111827"            # Dark panel with glass feel
BOX_BORDER = "#0ea5e9"        # Cyan edge
TEXT_MAIN = "#f8fafc"         # White text
TEXT_MUTED = "#94a3b8"        # Gray text
CYAN_GLOW = "#22d3ee"         # Cyan highlight
EMERALD = "#10b981"           # Soft green
SOFT_MINT = "#6ee7b7"         # Accent green
WARNING_ORANGE = "#f59e0b"    # Transition orange
DANGER_RED = "#f43f5e"        # Red alert

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def create_figure():
    # Massive 24x13.5 figure for glorious 16:9 4K-like rendering
    fig, ax = plt.subplots(figsize=(24, 13.5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Subtle grid background
    for i in np.linspace(0, 1, 25):
        ax.axhline(i, color="#ffffff", alpha=0.015, lw=1.5)
        ax.axvline(i, color="#ffffff", alpha=0.015, lw=1.5)
        
    return fig, ax

def draw_title(ax, text):
    # Main slide title: 48px equivalent
    ax.text(0.5, 0.92, text, ha='center', va='top', fontsize=44, weight='bold', color=TEXT_MAIN, 
            bbox=dict(facecolor=BG_COLOR, edgecolor='none', pad=0, alpha=0.9))

def draw_glass_box(ax, x, y, width, height, text="", title="", bg=BOX_BG, border=BOX_BORDER, alpha=0.9, fontsize=24, title_color=CYAN_GLOW, lw=3):
    # Generous padding inside box via pad=0.03 (approx 40-50px equivalent)
    shadow = patches.FancyBboxPatch((x - width/2 + 0.008, y - height/2 - 0.01), width, height, 
                                 boxstyle="round,pad=0.03,rounding_size=0.03", 
                                 ec="none", fc="#000000", alpha=0.5)
    ax.add_patch(shadow)
    
    box = patches.FancyBboxPatch((x - width/2, y - height/2), width, height, 
                                 boxstyle="round,pad=0.03,rounding_size=0.03", 
                                 ec=border, fc=bg, lw=lw, alpha=alpha)
    ax.add_patch(box)
    
    if title:
        # Title size: 30-36px equivalent
        ax.text(x, y + height/2 - 0.015, title, ha='center', va='top', color=title_color, 
                fontsize=fontsize+8, weight='bold')
        if text:
            # Body size: 24-28px equivalent, generous line spacing
            ax.text(x, y - 0.04, text, ha='center', va='center', color=TEXT_MAIN, 
                    fontsize=fontsize, linespacing=1.8)
    elif text:
        ax.text(x, y, text, ha='center', va='center', color=TEXT_MAIN, 
                fontsize=fontsize, linespacing=1.8, weight='bold')

def draw_glow_arrow(ax, x1, y1, x2, y2, color=CYAN_GLOW, lw_core=3, lw_glow=10, mutation_scale=40):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw_glow, alpha=0.15, mutation_scale=mutation_scale))
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw_core, mutation_scale=mutation_scale))

def draw_bar(ax, x, y, width, height, value, color, label):
    ax.add_patch(patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=0.01", fc="#1e293b", ec="none"))
    fill_w = width * value
    ax.add_patch(patches.FancyBboxPatch((x, y), fill_w, height, boxstyle="round,pad=0,rounding_size=0.01", fc=color, ec="none"))
    ax.text(x, y + height + 0.015, label, color=TEXT_MUTED, fontsize=20)
    ax.text(x + fill_w + 0.015, y + height/2, f"{value*100:.0f}%", color=color, va='center', weight='bold', fontsize=22)

# ---------------------------------------------------------
# FIGURES GENERATION
# ---------------------------------------------------------

def fig_01_system_overview():
    fig, ax = create_figure()
    draw_title(ax, "HYBRID INTELLIGENT DECISION SUPPORT PIPELINE")
    
    # 5 massive, well-spaced blocks
    steps = [
        ("CNN Visual Perception\n(EfficientNet-B0)", 0.78, "#0284c7"),
        ("Uncertainty Handling\n(Entropy & OOD Detection)", 0.62, WARNING_ORANGE),
        ("Fuzzy Severity Reasoning\n(Sugeno Inference)", 0.46, CYAN_GLOW),
        ("Contextual Alert Escalation\n(Environmental Risk)", 0.30, EMERALD),
        ("Final Decision & XAI\n(Symbolic Traceability)", 0.14, SOFT_MINT)
    ]
    
    for i, (text, y, color) in enumerate(steps):
        draw_glass_box(ax, 0.5, y, 0.55, 0.04, text=text, border=color, title_color=color, fontsize=26)
        if i < len(steps) - 1:
            next_y = steps[i+1][1]
            draw_glow_arrow(ax, 0.5, y - 0.065, 0.5, next_y + 0.065, color=color)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_01_system_overview.png"), bbox_inches='tight')
    plt.close()

def fig_02_severity_pipeline():
    fig, ax = create_figure()
    draw_title(ax, "SEVERITY REASONING & CONTEXTUAL ESCALATION")
    
    draw_glass_box(ax, 0.5, 0.71, 0.55, 0.18, title="1. CNN Softmax Distribution", border="#0284c7")
    draw_bar(ax, 0.32, 0.69, 0.25, 0.02, 0.49, "#3b82f6", "Mild Confidence")
    draw_bar(ax, 0.32, 0.62, 0.25, 0.02, 0.51, WARNING_ORANGE, "Severe Confidence")
    
    draw_glow_arrow(ax, 0.5, 0.57, 0.5, 0.51)
    
    draw_glass_box(ax, 0.5, 0.40, 0.65, 0.16, title="2. Fuzzy Reasoning Zone", 
                   text="Transition region detected.\nModerate intermediate state activated via fuzzy interpolation.", border=CYAN_GLOW, fontsize=24)
    
    draw_glow_arrow(ax, 0.5, 0.28, 0.5, 0.19)
    
    draw_glass_box(ax, 0.25, 0.10, 0.35, 0.14, title="3. Environment", text="Humidity: 90%   |   Temp: 35°C", border=EMERALD, fontsize=24)
    draw_glow_arrow(ax, 0.47, 0.10, 0.65, 0.10, color=EMERALD)
    draw_glass_box(ax, 0.80, 0.10, 0.15, 0.14, title="Final Alert", text="DANGER", border=DANGER_RED, fontsize=36, title_color=DANGER_RED)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_02_severity_pipeline.png"), bbox_inches='tight')
    plt.close()

def fig_03_membership_functions():
    fig, ax = plt.subplots(figsize=(24, 13.5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BOX_BG)
    
    x = np.linspace(0, 100, 500)
    def trapmf(x, a, b, c, d):
        return np.maximum(0, np.minimum(np.minimum((x-a)/(b-a + 1e-9), 1), (d-x)/(d-c + 1e-9)))
    def trimf(x, a, b, c):
        return np.maximum(0, np.minimum((x-a)/(b-a + 1e-9), (c-x)/(c-b + 1e-9)))
        
    y_mild = trimf(x, 10, 30, 50)
    y_mod = trimf(x, 30, 55, 80)
    y_sev = trapmf(x, 60, 85, 100, 100)
    
    ax.plot(x, y_mild, color="#3b82f6", lw=5, label="Mild")
    ax.fill_between(x, 0, y_mild, color="#3b82f6", alpha=0.15)
    
    ax.plot(x, y_mod, color=WARNING_ORANGE, lw=6, label="Moderate (Fuzzy Transition State)")
    ax.fill_between(x, 0, y_mod, color=WARNING_ORANGE, alpha=0.25)
    
    ax.plot(x, y_sev, color=DANGER_RED, lw=5, label="Severe")
    ax.fill_between(x, 0, y_sev, color=DANGER_RED, alpha=0.15)
    
    ax.set_title("CONTINUOUS SEVERITY INTERPOLATION (VSI)", color=TEXT_MAIN, fontsize=40, weight='bold', pad=30)
    ax.set_xlabel("Visual Severity Index (VSI)", color=TEXT_MUTED, fontsize=24, labelpad=20)
    ax.set_ylabel("Membership Degree (μ)", color=TEXT_MUTED, fontsize=24, labelpad=20)
    ax.tick_params(colors=TEXT_MUTED, labelsize=20)
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(2)
        
    ax.legend(loc='upper right', facecolor=BOX_BG, edgecolor=BOX_BORDER, labelcolor=TEXT_MAIN, fontsize=22, borderpad=1)
    ax.grid(color="#ffffff", alpha=0.05, lw=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_03_membership_functions.png"), facecolor=BG_COLOR)
    plt.close()

def fig_04_vsi_transition():
    fig, ax = create_figure()
    draw_title(ax, "VSI: CONTINUOUS SEVERITY SPECTRUM")
    
    gradient = np.linspace(0, 1, 512).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0.1, 0.9, 0.45, 0.55])
    
    labels = [(0.15, "Healthy", "VSI = 10"), (0.35, "Mild", "VSI = 30"), 
              (0.60, "Moderate\n(Interpolation Zone)", "VSI = 55"), (0.85, "Severe", "VSI = 85")]
    
    for x, top, bot in labels:
        ax.plot([x, x], [0.44, 0.56], color="#ffffff", lw=4)
        ax.text(x, 0.62, top, ha='center', va='bottom', color=TEXT_MAIN, fontsize=28, weight='bold')
        ax.text(x, 0.38, bot, ha='center', va='top', color=CYAN_GLOW, fontsize=24)
        
    ax.text(0.5, 0.15, "Unlike discrete CNN classes, Moderate emerges as an intermediate reasoning state.", 
            ha='center', va='center', color=TEXT_MUTED, fontsize=26, style='italic')

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_04_vsi_transition.png"), bbox_inches='tight')
    plt.close()

def fig_05_rule_activation():
    fig, ax = create_figure()
    draw_title(ax, "SYMBOLIC TRACEABILITY: EXPLAINABLE RULE ACTIVATION")
    
    table_data = [
        ["Rule ID", "Fuzzy Condition", "Firing Strength (w)", "Contribution %"],
        ["R1", "IF CNN Severe Medium THEN Moderate", "0.72", "48%"],
        ["R2", "IF Humidity Wet THEN ERI High", "0.61", "32%"],
        ["R3", "IF Mild High THEN Mild", "0.21", "20%"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='left', bbox=[0.1, 0.35, 0.8, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    
    for (i, j), cell in table.get_celld().items():
        cell.PAD = 0.08
        if i == 0:
            cell.set_facecolor("#0f766e")
            cell.set_text_props(weight='bold', color=TEXT_MAIN)
        else:
            cell.set_facecolor(BOX_BG)
            cell.set_text_props(color=TEXT_MAIN)
        cell.set_edgecolor("#334155")
        
    ax.text(0.5, 0.15, "Formula: Contribution % = (w * z) / Σ(w * z)", 
            ha='center', va='center', color=CYAN_GLOW, fontsize=28, weight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_05_rule_activation.png"), bbox_inches='tight')
    plt.close()

def fig_06_ood_uncertainty():
    fig, ax = create_figure()
    draw_title(ax, "UNCERTAINTY-AWARE REASONING & OOD DETECTION")
    
    draw_glass_box(ax, 0.25, 0.65, 0.35, 0.22, title="High Entropy Output", text="Mild = 0.34\nSevere = 0.33\nBlast = 0.33", border=WARNING_ORANGE, title_color=WARNING_ORANGE)
    draw_glow_arrow(ax, 0.25, 0.50, 0.25, 0.35)
    draw_glass_box(ax, 0.25, 0.20, 0.40, 0.22, title="HYBRID_WARNING", text="Smooths borderline decisions.\nEscalates context alert.", border=CYAN_GLOW)
    
    draw_glass_box(ax, 0.75, 0.65, 0.35, 0.22, title="Unrecognized Output", text="Max Confidence < 0.25", border=DANGER_RED, title_color=DANGER_RED)
    draw_glow_arrow(ax, 0.75, 0.50, 0.75, 0.35)
    draw_glass_box(ax, 0.75, 0.20, 0.40, 0.22, title="OOD REJECTION", text="Halt automated reasoning.\nRequire Expert Input.", border=DANGER_RED)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_06_ood_uncertainty.png"), bbox_inches='tight')
    plt.close()

def fig_07_contextual_alert():
    fig, ax = create_figure()
    draw_title(ax, "CONTEXTUAL ALERT ESCALATION")
    
    draw_glass_box(ax, 0.5, 0.75, 0.45, 0.12, title="Visual Severity Index (VSI)", text="MILD (Score: 30)", border="#3b82f6")
    
    draw_glow_arrow(ax, 0.5, 0.65, 0.25, 0.55)
    draw_glow_arrow(ax, 0.5, 0.65, 0.75, 0.55)
    
    draw_glass_box(ax, 0.25, 0.42, 0.4, 0.18, title="CASE A: Low ERI", text="Humidity: 50% | Temp: 22°C", border="#10b981", title_color="#10b981")
    draw_glow_arrow(ax, 0.25, 0.29, 0.25, 0.21)
    draw_glass_box(ax, 0.25, 0.10, 0.35, 0.14, title="ATTENTION", text="Regular Monitoring", border=WARNING_ORANGE, title_color=WARNING_ORANGE)
    
    draw_glass_box(ax, 0.75, 0.42, 0.4, 0.18, title="CASE B: High ERI", text="Humidity: 90% | Temp: 35°C", border=DANGER_RED, title_color=DANGER_RED)
    draw_glow_arrow(ax, 0.75, 0.29, 0.75, 0.21)
    draw_glass_box(ax, 0.75, 0.10, 0.35, 0.14, title="DANGER", text="Immediate Action Required", border=DANGER_RED, title_color=DANGER_RED)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_07_contextual_alert.png"), bbox_inches='tight')
    plt.close()

def fig_08_xai_report():
    fig, ax = create_figure()
    draw_title(ax, "NATURAL LANGUAGE EXPLAINABILITY (XAI)")
    
    report_text = """
    VSI: 58 (Moderate)      ERI: 85 (High)      Final Alert: DANGER
    -------------------------------------------------------------------------------------------------
    
    > ACTIVATED REASONING PATHWAYS:
    [ + ] IF CNN Severe is Medium THEN VSI is Moderate (Contributed 55%)
    [ + ] IF Humidity is Wet THEN ERI is High (Contributed 100%)
    [ + ] IF VSI is Moderate AND ERI is High THEN Alert is Danger
    
    > SYSTEM EXPLANATION:
    "Although the visual severity is only moderate (borderline transition), 
    the extremely high humidity (90%) significantly increases the risk of 
    rapid disease spread. Immediate action is recommended."
    """
    
    draw_glass_box(ax, 0.5, 0.45, 0.85, 0.65, text=report_text, border=CYAN_GLOW, fontsize=24)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_08_xai_report.png"), bbox_inches='tight')
    plt.close()

def fig_09_error_analysis():
    fig, ax = create_figure()
    draw_title(ax, "ERROR ANALYSIS: CONSERVATIVE DOWNGRADES")
    
    labels = ["Severe -> Mild\n(Downgrade)", "Mild -> Severe\n(Upgrade)", "Severe -> Healthy\n(Missed Danger)"]
    values = [2, 1, 0]
    colors = [WARNING_ORANGE, CYAN_GLOW, EMERALD]
    
    ax_bar = fig.add_axes([0.2, 0.35, 0.6, 0.45])
    ax_bar.set_facecolor(BG_COLOR)
    bars = ax_bar.bar(labels, values, color=colors, width=0.4)
    ax_bar.set_ylabel("Number of Occurrences", color=TEXT_MAIN, fontsize=22)
    ax_bar.tick_params(colors=TEXT_MAIN, labelsize=20)
    for spine in ax_bar.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(2)
        
    ax.text(0.5, 0.15, "Most errors are conservative downgrades in visually ambiguous cases.\nZero 'Severe -> Healthy' errors demonstrate safe decision-support behavior.", 
            ha='center', va='center', color=CYAN_GLOW, fontsize=26, style='italic', linespacing=1.6)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_09_error_analysis.png"), bbox_inches='tight')
    plt.close()

def fig_10_system_contributions():
    fig, ax = create_figure()
    draw_title(ax, "HYBRID SYSTEM COMPONENT CONTRIBUTIONS")
    
    table_data = [
        ["Component Layer", "Academic Contribution"],
        ["CNN (EfficientNet-B0)", "Visual disease perception & deep feature extraction"],
        ["Fuzzy Inference System", "Severity reasoning & continuous interpolation"],
        ["Environmental Risk Index", "Contextual alert escalation under real conditions"],
        ["OOD & Entropy Checks", "Uncertainty handling & anomaly rejection"],
        ["Explainability (XAI)", "Rule traceability & interpretable reasoning"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='left', bbox=[0.1, 0.25, 0.8, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    
    for (i, j), cell in table.get_celld().items():
        cell.PAD = 0.08
        if i == 0:
            cell.set_facecolor("#0284c7")
            cell.set_text_props(weight='bold', color=TEXT_MAIN)
        else:
            cell.set_facecolor(BOX_BG)
            cell.set_text_props(color=TEXT_MAIN)
        cell.set_edgecolor("#334155")

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_10_system_contributions.png"), bbox_inches='tight')
    plt.close()

def fig_11_hybrid_ai_architecture():
    fig, ax = create_figure()
    draw_title(ax, "HYBRID INTELLIGENT ARCHITECTURE")
    
    draw_glass_box(ax, 0.5, 0.46, 0.85, 0.72, border="#334155", bg=BG_COLOR)
    
    stack = [
        ("Explainable Reasoning (XAI Traceability)", EMERALD),
        ("Environmental Context (Temperature, Humidity)", WARNING_ORANGE),
        ("Fuzzy Logic (Uncertainty & Interpolation)", CYAN_GLOW),
        ("Deep Learning (EfficientNet-B0 Perception)", "#0284c7")
    ]
    
    y_start = 0.66
    y_gap = 0.13
    for i, (text, color) in enumerate(stack):
        draw_glass_box(ax, 0.5, y_start - i*y_gap, 0.75, 0.08, text=text, border=color, bg=BOX_BG, fontsize=26)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_11_hybrid_ai_architecture.png"), bbox_inches='tight')
    plt.close()

def fig_12_deployment_pipeline():
    fig, ax = create_figure()
    draw_title(ax, "END-TO-END DEPLOYMENT PIPELINE")
    
    boxes = [
        (0.125, 0.65, "User Upload", "#334155"),
        (0.375, 0.65, "FastAPI Backend", "#0284c7"),
        (0.625, 0.65, "CNN Model", CYAN_GLOW),
        (0.875, 0.65, "Fuzzy Engine", EMERALD),
        (0.875, 0.30, "XAI Logic", EMERALD),
        (0.500, 0.30, "JSON Response", "#334155"),
        (0.125, 0.30, "Streamlit UI", WARNING_ORANGE)
    ]
    
    for x, y, text, color in boxes:
        draw_glass_box(ax, x, y, 0.12, 0.12, text=text, border=color, fontsize=22, lw=2)
        
    draw_glow_arrow(ax, 0.22, 0.65, 0.28, 0.65, lw_core=2, lw_glow=6, mutation_scale=30)
    draw_glow_arrow(ax, 0.47, 0.65, 0.53, 0.65, lw_core=2, lw_glow=6, mutation_scale=30)
    draw_glow_arrow(ax, 0.72, 0.65, 0.78, 0.65, lw_core=2, lw_glow=6, mutation_scale=30)
    
    draw_glow_arrow(ax, 0.875, 0.55, 0.875, 0.40, lw_core=2, lw_glow=6, mutation_scale=30)
    draw_glow_arrow(ax, 0.77, 0.30, 0.61, 0.30, lw_core=2, lw_glow=6, mutation_scale=30)
    draw_glow_arrow(ax, 0.39, 0.30, 0.23, 0.30, lw_core=2, lw_glow=6, mutation_scale=30)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_12_deployment_pipeline.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Refined Premium Academic Figures...")
    fig_01_system_overview()
    fig_02_severity_pipeline()
    fig_03_membership_functions()
    fig_04_vsi_transition()
    fig_05_rule_activation()
    fig_06_ood_uncertainty()
    fig_07_contextual_alert()
    fig_08_xai_report()
    fig_09_error_analysis()
    fig_10_system_contributions()
    fig_11_hybrid_ai_architecture()
    fig_12_deployment_pipeline()
    print("Done!")
