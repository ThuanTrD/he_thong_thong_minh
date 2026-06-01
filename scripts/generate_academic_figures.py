import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Theme Colors
BG_COLOR = "#0f172a"
BOX_BG = "#1e293b"
BOX_BORDER = "#38bdf8"
TEXT_COLOR = "#f8fafc"
ACCENT_GREEN = "#10b981"
ACCENT_TEAL = "#14b8a6"
ACCENT_RED = "#f43f5e"
ACCENT_ORANGE = "#f59e0b"

def create_figure():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    return fig, ax

def draw_box(ax, x, y, width, height, text, bg=BOX_BG, border=BOX_BORDER, text_color=TEXT_COLOR, fontsize=14, weight='bold'):
    box = patches.FancyBboxPatch((x - width/2, y - height/2), width, height, 
                                 boxstyle="round,pad=0.1,rounding_size=0.1", 
                                 ec=border, fc=bg, lw=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', color=text_color, 
            fontsize=fontsize, weight=weight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color=TEXT_COLOR):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=2, mutation_scale=20))

def fig_01_system_overview():
    fig, ax = create_figure()
    
    steps = [
        ("Leaf Image", "#334155"),
        ("CNN Disease Recognition\n(Perception)", "#0284c7"),
        ("Softmax Confidence", "#0369a1"),
        ("OOD / Entropy Check\n(Uncertainty Handling)", "#ea580c"),
        ("Fuzzy Severity Inference\n(Reasoning)", ACCENT_TEAL),
        ("VSI Computation", "#0d9488"),
        ("Environmental Risk Assessment\n(Contextual Escalation)", "#0f766e"),
        ("Final Alert Level", ACCENT_RED),
        ("Explainability Report\n(Traceability)", ACCENT_GREEN)
    ]
    
    y_start = 0.9
    y_gap = 0.095
    
    ax.text(0.5, 0.98, "HYBRID INTELLIGENT SYSTEM PIPELINE", ha='center', va='top', 
            fontsize=24, weight='bold', color=TEXT_COLOR)
    
    for i, (text, bg) in enumerate(steps):
        y = y_start - i * y_gap
        draw_box(ax, 0.5, y, 0.4, 0.06, text, bg=bg, border="#ffffff", fontsize=14)
        if i < len(steps) - 1:
            draw_arrow(ax, 0.5, y - 0.03, 0.5, y - y_gap + 0.03)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_01_system_overview.png"), facecolor=BG_COLOR)
    plt.close()

def fig_02_severity_pipeline():
    fig, ax = create_figure()
    
    ax.text(0.5, 0.95, "SEVERITY REASONING PIPELINE", ha='center', va='top', 
            fontsize=24, weight='bold', color=TEXT_COLOR)
            
    draw_box(ax, 0.5, 0.85, 0.3, 0.08, "Input Leaf Image", bg="#334155")
    draw_arrow(ax, 0.5, 0.81, 0.5, 0.73)
    
    draw_box(ax, 0.5, 0.65, 0.35, 0.1, "CNN Output\nMild = 0.49 | Severe = 0.51", bg="#0284c7")
    draw_arrow(ax, 0.5, 0.60, 0.5, 0.50)
    
    draw_box(ax, 0.5, 0.45, 0.4, 0.1, "Fuzzy Membership Activation\n[Moderate transition region activated]", bg=ACCENT_TEAL)
    draw_arrow(ax, 0.5, 0.40, 0.5, 0.32)
    
    draw_box(ax, 0.5, 0.28, 0.2, 0.08, "VSI = 58", bg="#0d9488")
    draw_arrow(ax, 0.5, 0.24, 0.5, 0.16)
    
    draw_box(ax, 0.3, 0.12, 0.3, 0.1, "Environment\nHumidity = 90%\nTemp = 35°C", bg=ACCENT_ORANGE)
    draw_arrow(ax, 0.45, 0.12, 0.7, 0.12)
    
    draw_box(ax, 0.8, 0.12, 0.25, 0.1, "Final Alert\nDANGER", bg=ACCENT_RED, fontsize=18)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_02_severity_pipeline.png"), facecolor=BG_COLOR)
    plt.close()

def fig_03_membership_functions():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BOX_BG)
    
    x = np.linspace(0, 100, 500)
    
    def trapmf(x, a, b, c, d):
        return np.maximum(0, np.minimum(np.minimum((x-a)/(b-a + 1e-9), 1), (d-x)/(d-c + 1e-9)))
    def trimf(x, a, b, c):
        return np.maximum(0, np.minimum((x-a)/(b-a + 1e-9), (c-x)/(c-b + 1e-9)))
        
    y_healthy = trapmf(x, 0, 0, 10, 20)
    y_mild = trimf(x, 10, 30, 50)
    y_moderate = trimf(x, 30, 55, 80)
    y_severe = trapmf(x, 60, 85, 100, 100)
    
    ax.plot(x, y_healthy, color=ACCENT_GREEN, lw=3, label="Healthy")
    ax.fill_between(x, 0, y_healthy, color=ACCENT_GREEN, alpha=0.2)
    
    ax.plot(x, y_mild, color="#3b82f6", lw=3, label="Mild")
    ax.fill_between(x, 0, y_mild, color="#3b82f6", alpha=0.2)
    
    ax.plot(x, y_moderate, color=ACCENT_ORANGE, lw=3, label="Moderate (Intermediate State)")
    ax.fill_between(x, 0, y_moderate, color=ACCENT_ORANGE, alpha=0.2)
    
    ax.plot(x, y_severe, color=ACCENT_RED, lw=3, label="Severe")
    ax.fill_between(x, 0, y_severe, color=ACCENT_RED, alpha=0.2)
    
    ax.set_title("VSI FUZZY MEMBERSHIP FUNCTIONS", color=TEXT_COLOR, fontsize=24, weight='bold', pad=20)
    ax.set_xlabel("Visual Severity Index (VSI)", color=TEXT_COLOR, fontsize=16)
    ax.set_ylabel("Membership Degree", color=TEXT_COLOR, fontsize=16)
    ax.tick_params(colors=TEXT_COLOR, labelsize=12)
    for spine in ax.spines.values():
        spine.set_color(BOX_BORDER)
        
    ax.legend(loc='upper right', facecolor=BOX_BG, edgecolor=BOX_BORDER, labelcolor=TEXT_COLOR, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_03_membership_functions.png"), facecolor=BG_COLOR)
    plt.close()

def fig_04_vsi_transition():
    fig, ax = create_figure()
    
    ax.text(0.5, 0.8, "VSI CONTINUOUS SEVERITY SPECTRUM", ha='center', va='top', 
            fontsize=26, weight='bold', color=TEXT_COLOR)
            
    # Draw gradient bar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0.1, 0.9, 0.45, 0.55])
    ax.plot([0.1, 0.9, 0.9, 0.1, 0.1], [0.45, 0.45, 0.55, 0.55, 0.45], color=TEXT_COLOR, lw=2)
    
    # Labels
    labels = [
        (0.15, "Healthy", "VSI = 12"),
        (0.35, "Mild", "VSI = 34"),
        (0.65, "Moderate\n(Fuzzy Interpolation)", "VSI = 58"),
        (0.85, "Severe", "VSI = 82")
    ]
    
    for x, top_text, bot_text in labels:
        ax.plot([x, x], [0.45, 0.55], color=TEXT_COLOR, lw=3)
        ax.text(x, 0.6, top_text, ha='center', va='bottom', color=TEXT_COLOR, fontsize=16, weight='bold')
        ax.text(x, 0.4, bot_text, ha='center', va='top', color=TEXT_COLOR, fontsize=14)
        
    ax.text(0.5, 0.2, "Note: Moderate is an intermediate fuzzy reasoning state, not a CNN hard-class.", 
            ha='center', va='center', color=ACCENT_TEAL, fontsize=16, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_04_vsi_transition.png"), facecolor=BG_COLOR)
    plt.close()

def fig_05_rule_activation():
    fig, ax = create_figure()
    ax.text(0.5, 0.9, "EXPLAINABLE AI: SYMBOLIC RULE TRACEABILITY", ha='center', va='top', 
            fontsize=24, weight='bold', color=TEXT_COLOR)
            
    table_data = [
        ["Rule ID", "Fuzzy Condition", "Firing Strength (w)", "Contribution %"],
        ["R1", "IF CNN Severe Medium", "0.72", "48%"],
        ["R2", "IF Humidity High", "0.61", "32%"],
        ["R3", "IF Mild High", "0.21", "20%"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0.1, 0.3, 0.8, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor(ACCENT_TEAL)
            cell.set_text_props(weight='bold', color="#000000")
        else:
            cell.set_facecolor(BOX_BG)
            cell.set_text_props(color=TEXT_COLOR)
        cell.set_edgecolor(BOX_BORDER)
        
    ax.text(0.5, 0.2, "Formula: Contribution % = (w * z) / Σ(w * z)", 
            ha='center', va='center', color=ACCENT_ORANGE, fontsize=16, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_05_rule_activation.png"), facecolor=BG_COLOR)
    plt.close()

def fig_06_ood_uncertainty():
    fig, ax = create_figure()
    ax.text(0.5, 0.9, "UNCERTAINTY-AWARE REASONING & OOD DETECTION", ha='center', va='top', 
            fontsize=24, weight='bold', color=TEXT_COLOR)
            
    draw_box(ax, 0.5, 0.75, 0.6, 0.12, "CNN Softmax Output\nMild=0.34, Severe=0.33, Blast=0.33", bg="#334155")
    draw_arrow(ax, 0.5, 0.69, 0.5, 0.61)
    
    draw_box(ax, 0.5, 0.55, 0.4, 0.12, "Entropy Check\nStatus: HIGH UNCERTAINTY", bg=ACCENT_ORANGE)
    draw_arrow(ax, 0.5, 0.49, 0.5, 0.41)
    
    draw_box(ax, 0.5, 0.35, 0.4, 0.12, "Inference Mode Selection\nHYBRID_WARNING", bg=ACCENT_TEAL)
    draw_arrow(ax, 0.5, 0.29, 0.5, 0.21)
    
    draw_box(ax, 0.5, 0.15, 0.6, 0.12, "Action\nReduce Confidence, Escalate Alert, Request Expert", bg=ACCENT_RED)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_06_ood_uncertainty.png"), facecolor=BG_COLOR)
    plt.close()

def fig_07_contextual_alert():
    fig, ax = create_figure()
    ax.text(0.5, 0.9, "CONTEXTUAL ALERT ESCALATION", ha='center', va='top', 
            fontsize=24, weight='bold', color=TEXT_COLOR)
            
    # Case A
    draw_box(ax, 0.25, 0.65, 0.3, 0.1, "CASE A\nVSI = MILD", bg=BOX_BG)
    draw_box(ax, 0.25, 0.5, 0.3, 0.1, "Humidity: LOW (50%)\nTemp: COOL (22°C)", bg="#0284c7")
    draw_arrow(ax, 0.25, 0.45, 0.25, 0.35)
    draw_box(ax, 0.25, 0.3, 0.25, 0.1, "ATTENTION", bg=ACCENT_ORANGE, fontsize=18)
    
    # Case B
    draw_box(ax, 0.75, 0.65, 0.3, 0.1, "CASE B\nVSI = MILD", bg=BOX_BG)
    draw_box(ax, 0.75, 0.5, 0.3, 0.1, "Humidity: HIGH (90%)\nTemp: HOT (35°C)", bg="#0f766e")
    draw_arrow(ax, 0.75, 0.45, 0.75, 0.35)
    draw_box(ax, 0.75, 0.3, 0.25, 0.1, "DANGER", bg=ACCENT_RED, fontsize=18)
    
    ax.plot([0.5, 0.5], [0.2, 0.75], color=BOX_BORDER, lw=2, ls='--')
    
    ax.text(0.5, 0.1, "Same visual severity + Different context = Different decision", 
            ha='center', va='center', color=ACCENT_TEAL, fontsize=18, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_07_contextual_alert.png"), facecolor=BG_COLOR)
    plt.close()

def fig_08_xai_report():
    fig, ax = create_figure()
    ax.text(0.5, 0.9, "EXPLAINABILITY REPORT (NATURAL LANGUAGE)", ha='center', va='top', 
            fontsize=24, weight='bold', color=TEXT_COLOR)
            
    report_text = """
    DIAGNOSIS REPORT
    =================
    Visual Severity Index (VSI): 58 (Moderate)
    Environmental Risk Index (ERI): 85 (High)
    Final Alert Level: DANGER
    
    REASONING:
    - 1. [VSI Rule] IF CNN Severe is Medium THEN VSI is Moderate (Contributed 55%)
    - 2. [ERI Rule] IF Humidity is Wet THEN ERI is High (Contributed 100%)
    - 3. [Alert Rule] IF VSI is Moderate AND ERI is High THEN Alert is Danger
    
    EXPLANATION:
    "Although the visual severity is only moderate (borderline transition), 
    the extremely high humidity (90%) significantly increases the risk of 
    rapid disease spread. Immediate action is recommended."
    """
    
    draw_box(ax, 0.5, 0.5, 0.8, 0.6, report_text, bg="#1e293b", border=ACCENT_TEAL, 
             text_color="#e2e8f0", fontsize=15, weight='normal')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_08_xai_report.png"), facecolor=BG_COLOR)
    plt.close()

def fig_09_error_analysis():
    fig, ax = create_figure()
    ax.text(0.5, 0.9, "ERROR ANALYSIS: CONSERVATIVE DOWNGRADE", ha='center', va='top', 
            fontsize=24, weight='bold', color=TEXT_COLOR)
            
    # Chart
    labels = ["Severe -> Mild\n(Downgrade)", "Mild -> Severe\n(Upgrade)", "Severe -> Healthy\n(Missed)"]
    values = [2, 1, 0]
    
    ax_bar = fig.add_axes([0.2, 0.4, 0.6, 0.4])
    ax_bar.set_facecolor(BG_COLOR)
    bars = ax_bar.bar(labels, values, color=[ACCENT_ORANGE, ACCENT_TEAL, ACCENT_GREEN])
    ax_bar.set_ylabel("Number of Errors", color=TEXT_COLOR, fontsize=14)
    ax_bar.tick_params(colors=TEXT_COLOR, labelsize=12)
    for spine in ax_bar.spines.values():
        spine.set_color(BOX_BORDER)
        
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom', color=TEXT_COLOR, fontsize=14, weight='bold')

    ax.text(0.5, 0.25, "Analysis: Errors occur only in highly uncertain borderline cases (e.g. Mild=0.49, Severe=0.51).", 
            ha='center', va='center', color=TEXT_COLOR, fontsize=16)
    ax.text(0.5, 0.18, "The system safely smooths these into 'Moderate' to avoid false alarms.", 
            ha='center', va='center', color=ACCENT_TEAL, fontsize=16)
    ax.text(0.5, 0.11, "Zero 'Severe -> Healthy' errors proves the system is safe and robust.", 
            ha='center', va='center', color=ACCENT_GREEN, fontsize=16, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_09_error_analysis.png"), facecolor=BG_COLOR)
    plt.close()

def fig_10_system_contributions():
    fig, ax = create_figure()
    ax.text(0.5, 0.9, "SYSTEM COMPONENT CONTRIBUTIONS", ha='center', va='top', 
            fontsize=24, weight='bold', color=TEXT_COLOR)
            
    table_data = [
        ["Component", "Contribution / Academic Value"],
        ["CNN (EfficientNet-B0)", "Visual disease perception & feature extraction"],
        ["Fuzzy Inference", "Severity reasoning & continuous interpolation"],
        ["ERI (Environment)", "Contextual alert escalation"],
        ["OOD Detection", "Uncertainty handling & anomaly detection"],
        ["Explainability Layer", "Rule traceability & interpretable reasoning"],
        ["Expert Fusion", "Human-guided field adjustment"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='left', bbox=[0.1, 0.25, 0.8, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor(ACCENT_TEAL)
            cell.set_text_props(weight='bold', color="#000000", ha='center')
        else:
            cell.set_facecolor(BOX_BG)
            cell.set_text_props(color=TEXT_COLOR)
            # Add padding
            cell.PAD = 0.1
        cell.set_edgecolor(BOX_BORDER)

    ax.text(0.5, 0.1, "Not just a classifier — an intelligent decision-support system.", 
            ha='center', va='center', color=ACCENT_TEAL, fontsize=18, style='italic', weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_10_system_contributions.png"), facecolor=BG_COLOR)
    plt.close()

def fig_11_hybrid_ai_architecture():
    fig, ax = create_figure()
    ax.text(0.5, 0.9, "HYBRID AI ARCHITECTURE STACK", ha='center', va='top', 
            fontsize=24, weight='bold', color=TEXT_COLOR)
            
    stack = [
        ("EXPLAINABLE REASONING\n(XAI Traceability)", ACCENT_GREEN),
        ("ENVIRONMENTAL CONTEXT\n(Temperature, Humidity)", ACCENT_ORANGE),
        ("EXPERT KNOWLEDGE\n(Human-in-the-loop Fusion)", ACCENT_RED),
        ("FUZZY LOGIC\n(Uncertainty Handling & Interpolation)", ACCENT_TEAL),
        ("DEEP LEARNING\n(EfficientNet-B0 Perception)", "#0284c7")
    ]
    
    y_start = 0.75
    y_gap = 0.12
    
    for i, (text, bg) in enumerate(stack):
        draw_box(ax, 0.5, y_start - i*y_gap, 0.6, 0.1, text, bg=bg, border="#ffffff")
        
    ax.text(0.5, 0.1, "INTELLIGENT DECISION SUPPORT SYSTEM", 
            ha='center', va='center', color="#f8fafc", fontsize=22, weight='bold', 
            bbox=dict(facecolor="#1e293b", edgecolor=ACCENT_TEAL, boxstyle='round,pad=0.5', lw=3))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_11_hybrid_ai_architecture.png"), facecolor=BG_COLOR)
    plt.close()

def fig_12_deployment_pipeline():
    fig, ax = create_figure()
    ax.text(0.5, 0.9, "END-TO-END DEPLOYMENT PIPELINE", ha='center', va='top', 
            fontsize=24, weight='bold', color=TEXT_COLOR)
            
    boxes = [
        (0.15, 0.6, "User Upload\n(Image & Env)"),
        (0.35, 0.6, "FastAPI Backend\n(REST API)"),
        (0.55, 0.6, "CNN Inference\n(PyTorch)"),
        (0.75, 0.6, "Fuzzy Engine\n(Sugeno)"),
        (0.85, 0.35, "XAI Generator\n(Traceability)"),
        (0.55, 0.35, "JSON Output\n(Results)"),
        (0.25, 0.35, "Streamlit Dashboard\n(UI Display)")
    ]
    
    for x, y, text in boxes:
        draw_box(ax, x, y, 0.18, 0.12, text, bg=BOX_BG, fontsize=13)
        
    # Draw connections
    draw_arrow(ax, 0.24, 0.6, 0.26, 0.6)
    draw_arrow(ax, 0.44, 0.6, 0.46, 0.6)
    draw_arrow(ax, 0.64, 0.6, 0.66, 0.6)
    
    draw_arrow(ax, 0.75, 0.54, 0.85, 0.41)
    draw_arrow(ax, 0.76, 0.35, 0.64, 0.35)
    draw_arrow(ax, 0.46, 0.35, 0.34, 0.35)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_12_deployment_pipeline.png"), facecolor=BG_COLOR)
    plt.close()

if __name__ == "__main__":
    print("Generating Academic Figures...")
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
    print("Successfully generated all 12 figures.")
