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
# PREMIUM COLOR PALETTE (Nature/IEEE style + Dark Academic)
# ---------------------------------------------------------
BG_COLOR = "#0b1121"          # Deep navy blue/black
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
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Subtle grid background
    for i in np.linspace(0, 1, 20):
        ax.axhline(i, color="#ffffff", alpha=0.015, lw=1)
        ax.axvline(i, color="#ffffff", alpha=0.015, lw=1)
        
    return fig, ax

def draw_glass_box(ax, x, y, width, height, text="", title="", bg=BOX_BG, border=BOX_BORDER, alpha=0.8, fontsize=14, title_color=CYAN_GLOW):
    # Shadow
    shadow = patches.FancyBboxPatch((x - width/2 + 0.008, y - height/2 - 0.008), width, height, 
                                 boxstyle="round,pad=0.02,rounding_size=0.04", 
                                 ec="none", fc="#000000", alpha=0.4)
    ax.add_patch(shadow)
    
    # Main Box (Glassmorphism effect)
    box = patches.FancyBboxPatch((x - width/2, y - height/2), width, height, 
                                 boxstyle="round,pad=0.02,rounding_size=0.04", 
                                 ec=border, fc=bg, lw=1.5, alpha=alpha)
    ax.add_patch(box)
    
    if title:
        ax.text(x, y + height/2 - 0.02, title, ha='center', va='top', color=title_color, 
                fontsize=fontsize+2, weight='bold')
        if text:
            ax.text(x, y - 0.03, text, ha='center', va='center', color=TEXT_MAIN, 
                    fontsize=fontsize, linespacing=1.6)
    elif text:
        ax.text(x, y, text, ha='center', va='center', color=TEXT_MAIN, 
                fontsize=fontsize, linespacing=1.6)

def draw_glow_arrow(ax, x1, y1, x2, y2, color=CYAN_GLOW):
    # Glow
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=5, alpha=0.15, mutation_scale=20))
    # Core
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, mutation_scale=20))

def draw_title(ax, text):
    ax.text(0.5, 0.93, text, ha='center', va='top', fontsize=26, weight='bold', color=TEXT_MAIN, 
            bbox=dict(facecolor=BG_COLOR, edgecolor='none', pad=0, alpha=0.8))

def draw_bar(ax, x, y, width, height, value, color, label):
    # Background bar
    ax.add_patch(patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=0.01", fc="#1e293b", ec="none"))
    # Fill bar
    fill_w = width * value
    ax.add_patch(patches.FancyBboxPatch((x, y), fill_w, height, boxstyle="round,pad=0,rounding_size=0.01", fc=color, ec="none"))
    # Label
    ax.text(x, y + height + 0.01, label, color=TEXT_MUTED, fontsize=12)
    # Value
    ax.text(x + fill_w + 0.01, y + height/2, f"{value*100:.0f}%", color=color, va='center', weight='bold', fontsize=12)

# ---------------------------------------------------------
# FIGURES GENERATION
# ---------------------------------------------------------

def fig_01_system_overview():
    fig, ax = create_figure()
    draw_title(ax, "HYBRID INTELLIGENT DECISION SUPPORT PIPELINE")
    
    steps = [
        ("CNN Visual Perception\n(EfficientNet-B0)", 0.85, "#0284c7"),
        ("Uncertainty Handling\n(Entropy & OOD Detection)", 0.70, WARNING_ORANGE),
        ("Fuzzy Severity Reasoning\n(Sugeno Inference)", 0.55, CYAN_GLOW),
        ("Contextual Alert Escalation\n(Environmental Risk)", 0.40, EMERALD),
        ("Final Decision & XAI\n(Symbolic Traceability)", 0.25, SOFT_MINT)
    ]
    
    for i, (text, y, color) in enumerate(steps):
        draw_glass_box(ax, 0.5, y, 0.5, 0.08, text=text, border=color, title_color=color, fontsize=16)
        if i < len(steps) - 1:
            next_y = steps[i+1][1]
            draw_glow_arrow(ax, 0.5, y - 0.06, 0.5, next_y + 0.06, color=color)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_01_system_overview.png"))
    plt.close()

def fig_02_severity_pipeline():
    fig, ax = create_figure()
    draw_title(ax, "SEVERITY REASONING & CONTEXTUAL ESCALATION")
    
    # CNN Output
    draw_glass_box(ax, 0.5, 0.75, 0.5, 0.15, title="1. CNN Softmax Distribution", border="#0284c7")
    draw_bar(ax, 0.35, 0.74, 0.2, 0.02, 0.49, "#3b82f6", "Mild Confidence")
    draw_bar(ax, 0.35, 0.69, 0.2, 0.02, 0.51, WARNING_ORANGE, "Severe Confidence")
    
    draw_glow_arrow(ax, 0.5, 0.65, 0.5, 0.55)
    
    # Fuzzy Inference
    draw_glass_box(ax, 0.5, 0.45, 0.6, 0.15, title="2. Fuzzy Reasoning Zone", 
                   text="Transition region detected.\nModerate intermediate state activated via fuzzy interpolation.", border=CYAN_GLOW)
    
    draw_glow_arrow(ax, 0.5, 0.35, 0.5, 0.25)
    
    # Context & Alert
    draw_glass_box(ax, 0.3, 0.15, 0.3, 0.15, title="3. Environment", text="Humidity: 90%\nTemp: 35°C", border=EMERALD)
    draw_glow_arrow(ax, 0.48, 0.15, 0.62, 0.15, color=EMERALD)
    draw_glass_box(ax, 0.75, 0.15, 0.2, 0.15, title="Final Alert", text="DANGER", border=DANGER_RED, fontsize=22, title_color=DANGER_RED)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_02_severity_pipeline.png"))
    plt.close()

def fig_03_membership_functions():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
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
    
    ax.plot(x, y_mild, color="#3b82f6", lw=3, label="Mild")
    ax.fill_between(x, 0, y_mild, color="#3b82f6", alpha=0.15)
    
    ax.plot(x, y_mod, color=WARNING_ORANGE, lw=4, label="Moderate (Fuzzy Transition State)")
    ax.fill_between(x, 0, y_mod, color=WARNING_ORANGE, alpha=0.25)
    
    ax.plot(x, y_sev, color=DANGER_RED, lw=3, label="Severe")
    ax.fill_between(x, 0, y_sev, color=DANGER_RED, alpha=0.15)
    
    ax.set_title("CONTINUOUS SEVERITY INTERPOLATION (VSI)", color=TEXT_MAIN, fontsize=24, weight='bold', pad=20)
    ax.set_xlabel("Visual Severity Index (VSI)", color=TEXT_MUTED, fontsize=16)
    ax.set_ylabel("Membership Degree (μ)", color=TEXT_MUTED, fontsize=16)
    ax.tick_params(colors=TEXT_MUTED, labelsize=12)
    for spine in ax.spines.values():
        spine.set_color("#334155")
        
    ax.legend(loc='upper right', facecolor=BOX_BG, edgecolor=BOX_BORDER, labelcolor=TEXT_MAIN, fontsize=14)
    ax.grid(color="#ffffff", alpha=0.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_03_membership_functions.png"), facecolor=BG_COLOR)
    plt.close()

def fig_04_vsi_transition():
    fig, ax = create_figure()
    draw_title(ax, "VSI: CONTINUOUS SEVERITY SPECTRUM")
    
    # Draw gradient bar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0.1, 0.9, 0.45, 0.55])
    
    labels = [(0.15, "Healthy", "VSI = 10"), (0.35, "Mild", "VSI = 30"), 
              (0.60, "Moderate\n(Interpolation)", "VSI = 55"), (0.85, "Severe", "VSI = 85")]
    
    for x, top, bot in labels:
        ax.plot([x, x], [0.45, 0.55], color="#ffffff", lw=3)
        ax.text(x, 0.6, top, ha='center', va='bottom', color=TEXT_MAIN, fontsize=18, weight='bold')
        ax.text(x, 0.4, bot, ha='center', va='top', color=CYAN_GLOW, fontsize=16)
        
    ax.text(0.5, 0.2, "Unlike discrete CNN classes, Moderate emerges as an intermediate reasoning state.", 
            ha='center', va='center', color=TEXT_MUTED, fontsize=16, style='italic')

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_04_vsi_transition.png"))
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
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='left', bbox=[0.1, 0.35, 0.8, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    
    for (i, j), cell in table.get_celld().items():
        cell.PAD = 0.05
        if i == 0:
            cell.set_facecolor("#0f766e")
            cell.set_text_props(weight='bold', color=TEXT_MAIN)
        else:
            cell.set_facecolor(BOX_BG)
            cell.set_text_props(color=TEXT_MAIN)
        cell.set_edgecolor("#334155")
        
    ax.text(0.5, 0.2, "Formula: Contribution % = (w * z) / Σ(w * z)", 
            ha='center', va='center', color=CYAN_GLOW, fontsize=18, weight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_05_rule_activation.png"))
    plt.close()

def fig_06_ood_uncertainty():
    fig, ax = create_figure()
    draw_title(ax, "UNCERTAINTY-AWARE REASONING & OOD DETECTION")
    
    draw_glass_box(ax, 0.25, 0.7, 0.3, 0.15, title="High Entropy Output", text="Mild=0.34\nSevere=0.33\nBlast=0.33", border=WARNING_ORANGE, title_color=WARNING_ORANGE)
    draw_glow_arrow(ax, 0.25, 0.6, 0.25, 0.45)
    draw_glass_box(ax, 0.25, 0.35, 0.35, 0.15, title="HYBRID_WARNING", text="Smooths borderline decisions.\nEscalates context alert.", border=CYAN_GLOW)
    
    draw_glass_box(ax, 0.75, 0.7, 0.3, 0.15, title="Unrecognized Output", text="Max Confidence < 0.25", border=DANGER_RED, title_color=DANGER_RED)
    draw_glow_arrow(ax, 0.75, 0.6, 0.75, 0.45)
    draw_glass_box(ax, 0.75, 0.35, 0.35, 0.15, title="OOD REJECTION", text="Halt automated reasoning.\nRequire Expert Input.", border=DANGER_RED)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_06_ood_uncertainty.png"))
    plt.close()

def fig_07_contextual_alert():
    fig, ax = create_figure()
    draw_title(ax, "CONTEXTUAL ALERT ESCALATION")
    
    # Base VSI
    draw_glass_box(ax, 0.5, 0.8, 0.4, 0.1, title="Visual Severity Index (VSI)", text="MILD (Score: 30)", border="#3b82f6")
    draw_glow_arrow(ax, 0.5, 0.72, 0.25, 0.6)
    draw_glow_arrow(ax, 0.5, 0.72, 0.75, 0.6)
    
    # Case A
    draw_glass_box(ax, 0.25, 0.5, 0.35, 0.15, title="CASE A: Low ERI", text="Humidity: 50% | Temp: 22°C", border="#10b981", title_color="#10b981")
    draw_glow_arrow(ax, 0.25, 0.4, 0.25, 0.25)
    draw_glass_box(ax, 0.25, 0.15, 0.3, 0.12, title="ATTENTION", text="Regular Monitoring", border=WARNING_ORANGE, title_color=WARNING_ORANGE)
    
    # Case B
    draw_glass_box(ax, 0.75, 0.5, 0.35, 0.15, title="CASE B: High ERI", text="Humidity: 90% | Temp: 35°C", border=DANGER_RED, title_color=DANGER_RED)
    draw_glow_arrow(ax, 0.75, 0.4, 0.75, 0.25)
    draw_glass_box(ax, 0.75, 0.15, 0.3, 0.12, title="DANGER", text="Immediate Action Required", border=DANGER_RED, title_color=DANGER_RED)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_07_contextual_alert.png"))
    plt.close()

def fig_08_xai_report():
    fig, ax = create_figure()
    draw_title(ax, "NATURAL LANGUAGE EXPLAINABILITY (XAI)")
    
    report_text = """
    VSI: 58 (Moderate)  |  ERI: 85 (High)  |  Final Alert: DANGER
    ──────────────────────────────────────────────────────────────────
    
    > ACTIVATED REASONING PATHWAYS:
    [✓] IF CNN Severe is Medium THEN VSI is Moderate (Contributed 55%)
    [✓] IF Humidity is Wet THEN ERI is High (Contributed 100%)
    [✓] IF VSI is Moderate AND ERI is High THEN Alert is Danger
    
    > SYSTEM EXPLANATION:
    "Although the visual severity is only moderate (borderline transition), 
    the extremely high humidity (90%) significantly increases the risk of 
    rapid disease spread. Immediate action is recommended."
    """
    
    draw_glass_box(ax, 0.5, 0.45, 0.8, 0.6, text=report_text, border=CYAN_GLOW, fontsize=16)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_08_xai_report.png"))
    plt.close()

def fig_09_error_analysis():
    fig, ax = create_figure()
    draw_title(ax, "ERROR ANALYSIS: CONSERVATIVE DOWNGRADES")
    
    labels = ["Severe -> Mild\n(Downgrade)", "Mild -> Severe\n(Upgrade)", "Severe -> Healthy\n(Missed Danger)"]
    values = [2, 1, 0]
    colors = [WARNING_ORANGE, CYAN_GLOW, EMERALD]
    
    ax_bar = fig.add_axes([0.2, 0.4, 0.6, 0.4])
    ax_bar.set_facecolor(BG_COLOR)
    bars = ax_bar.bar(labels, values, color=colors, width=0.5)
    ax_bar.set_ylabel("Number of Occurrences", color=TEXT_MAIN, fontsize=14)
    ax_bar.tick_params(colors=TEXT_MAIN, labelsize=14)
    for spine in ax_bar.spines.values():
        spine.set_color("#334155")
        
    ax.text(0.5, 0.2, "Most errors are conservative downgrades in visually ambiguous cases.\nZero 'Severe -> Healthy' errors demonstrate safe decision-support behavior.", 
            ha='center', va='center', color=CYAN_GLOW, fontsize=18, style='italic')

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_09_error_analysis.png"))
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
    table.set_fontsize(16)
    
    for (i, j), cell in table.get_celld().items():
        cell.PAD = 0.05
        if i == 0:
            cell.set_facecolor("#0284c7")
            cell.set_text_props(weight='bold', color=TEXT_MAIN)
        else:
            cell.set_facecolor(BOX_BG)
            cell.set_text_props(color=TEXT_MAIN)
        cell.set_edgecolor("#334155")

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_10_system_contributions.png"))
    plt.close()

def fig_11_hybrid_ai_architecture():
    fig, ax = create_figure()
    draw_title(ax, "HYBRID INTELLIGENT ARCHITECTURE")
    
    draw_glass_box(ax, 0.5, 0.5, 0.8, 0.6, border="#334155", bg=BG_COLOR)
    
    stack = [
        ("Explainable Reasoning (XAI Traceability)", EMERALD),
        ("Environmental Context (Temperature, Humidity)", WARNING_ORANGE),
        ("Fuzzy Logic (Uncertainty & Interpolation)", CYAN_GLOW),
        ("Deep Learning (EfficientNet-B0 Perception)", "#0284c7")
    ]
    
    y_start = 0.65
    y_gap = 0.1
    for i, (text, color) in enumerate(stack):
        draw_glass_box(ax, 0.5, y_start - i*y_gap, 0.7, 0.08, text=text, border=color, bg=BOX_BG, fontsize=16)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_11_hybrid_ai_architecture.png"))
    plt.close()

def fig_12_deployment_pipeline():
    fig, ax = create_figure()
    draw_title(ax, "END-TO-END DEPLOYMENT PIPELINE")
    
    boxes = [
        (0.15, 0.6, "User Upload", "#334155"),
        (0.40, 0.6, "FastAPI Backend", "#0284c7"),
        (0.65, 0.6, "CNN Model", CYAN_GLOW),
        (0.85, 0.6, "Fuzzy Engine", EMERALD),
        (0.85, 0.3, "XAI Logic", EMERALD),
        (0.50, 0.3, "JSON Response", "#334155"),
        (0.15, 0.3, "Streamlit Dashboard", WARNING_ORANGE)
    ]
    
    for x, y, text, color in boxes:
        draw_glass_box(ax, x, y, 0.2, 0.12, text=text, border=color, fontsize=14)
        
    draw_glow_arrow(ax, 0.26, 0.6, 0.29, 0.6)
    draw_glow_arrow(ax, 0.51, 0.6, 0.54, 0.6)
    draw_glow_arrow(ax, 0.76, 0.6, 0.74, 0.6)  # CNN to Fuzzy
    
    draw_glow_arrow(ax, 0.85, 0.53, 0.85, 0.37)
    draw_glow_arrow(ax, 0.74, 0.3, 0.61, 0.3)
    draw_glow_arrow(ax, 0.39, 0.3, 0.26, 0.3)

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_12_deployment_pipeline.png"))
    plt.close()

if __name__ == "__main__":
    print("Generating Premium Academic Figures...")
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
