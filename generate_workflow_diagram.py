"""Generate high-precision publication-quality IEEE diagram for InterviewIQ Workflow."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

fig, ax = plt.subplots(figsize=(16, 8.2), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")

# Box styling helper
def draw_rounded_box(x, y, w, h, text="", title=None, bullets=None, bg="#FFFFFF", border="#000000", lw=1.4, ls="-", rx=1.0, fontsize=9.0, bold=False):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.25,rounding_size={rx}",
        facecolor=bg, edgecolor=border, linewidth=lw, linestyle=ls
    )
    ax.add_patch(box)
    
    if bullets:
        ax.text(x + w/2, y + h - 1.8, title or text, ha="center", va="top", fontsize=fontsize, fontweight="bold", color="#000000")
        start_y = y + h - 4.4
        for i, b in enumerate(bullets):
            ax.text(x + 1.2, start_y - (i * 2.0), f"•  {b}", ha="left", va="center", fontsize=fontsize - 0.8, color="#1A1A1A")
    elif title:
        ax.text(x + w/2, y + h/2 + 1.0, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", color="#000000")
        ax.text(x + w/2, y + h/2 - 1.2, text, ha="center", va="center", fontsize=fontsize - 1.0, color="#333333")
    elif text:
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color="#000000")

def draw_diamond(cx, cy, w, h, text, fontsize=8.0):
    pts = [[cx, cy + h/2], [cx + w/2, cy], [cx, cy - h/2], [cx - w/2, cy]]
    poly = patches.Polygon(pts, closed=True, facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.4)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, fontweight="bold")

def draw_cylinder(cx, cy, w, h, text, subtext=None):
    box = patches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.4
    )
    ax.add_patch(box)
    ic_x = cx - w/2 + 3.2
    ic_y = cy
    e1 = patches.Ellipse((ic_x, ic_y + 1.2), 3.0, 1.0, facecolor="#E2E8F0", edgecolor="#000000", lw=1.2)
    r1 = patches.Rectangle((ic_x - 1.5, ic_y - 1.4), 3.0, 2.6, facecolor="#FFFFFF", edgecolor="none")
    ax.add_patch(r1)
    ax.plot([ic_x - 1.5, ic_x - 1.5], [ic_y - 1.4, ic_y + 1.2], color="#000000", lw=1.2)
    ax.plot([ic_x + 1.5, ic_x + 1.5], [ic_y - 1.4, ic_y + 1.2], color="#000000", lw=1.2)
    e2 = patches.Arc((ic_x, ic_y - 1.4), 3.0, 1.0, theta1=180, theta2=360, edgecolor="#000000", lw=1.2)
    ax.add_patch(e1)
    ax.add_patch(e2)
    ax.text(cx + 1.5, cy + 0.9, text, ha="center", va="center", fontsize=9.2, fontweight="bold")
    if subtext:
        ax.text(cx + 1.5, cy - 1.1, subtext, ha="center", va="center", fontsize=7.8, color="#444444")

def draw_arrow(p1, p2, label=None, label_pos=(0, 0), rad=0.0):
    connectionstyle = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.3, shrinkA=2, shrinkB=2, connectionstyle=connectionstyle)
    )
    if label:
        lx = (p1[0] + p2[0]) / 2 + label_pos[0]
        ly = (p1[1] + p2[1]) / 2 + label_pos[1]
        ax.text(lx, ly, label, fontsize=8.2, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="#FFFFFF", edgecolor="none"))

# ─────────────────────────────────────────────────────────────────────────────
# 1. OUTER CONTAINERS
# ─────────────────────────────────────────────────────────────────────────────

# Left Container: User Interface (Frontend)
draw_rounded_box(2.5, 9.0, 18.0, 39.0, "", bg="#FFFFFF", border="#000000", lw=1.4, rx=0.8)
ax.text(11.5, 45.8, "User Interface\n(Frontend)", ha="center", va="center", fontsize=10.5, fontweight="bold")

# Middle Container: InterviewIQ – Backend / Core System (Dashed border)
draw_rounded_box(23.5, 3.5, 49.5, 54.0, "", bg="#FFFFFF", border="#000000", lw=1.4, ls="--", rx=1.0)
draw_rounded_box(32.5, 55.6, 31.5, 3.4, "InterviewIQ – Backend / Core System", bg="#FFFFFF", border="#000000", lw=1.4, rx=0.6, fontsize=10.2, bold=True)

# Right Container: Post-Processing & Feedback
draw_rounded_box(75.5, 5.0, 22.5, 49.0, "", bg="#FFFFFF", border="#000000", lw=1.4, rx=0.8)
draw_rounded_box(77.0, 51.5, 19.5, 3.0, "Post-Processing & Feedback", bg="#FFFFFF", border="#000000", lw=1.4, rx=0.5, fontsize=9.8, bold=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LEFT COLUMN: USER INTERFACE (FRONTEND)
# ─────────────────────────────────────────────────────────────────────────────
# Candidate (Student) Top pill
draw_rounded_box(4.5, 51.5, 14.0, 4.4, "Candidate\n(Student)", bg="#FFFFFF", border="#000000", lw=1.4, rx=1.0, fontsize=9.5, bold=True)
draw_arrow((11.5, 51.5), (11.5, 48.0))

# Step 1: Login / Open Interview Room
draw_rounded_box(3.8, 36.5, 15.4, 5.2, "Login / Open\nInterview Room", fontsize=8.8, bold=True)
draw_arrow((11.5, 36.5), (11.5, 33.0))

# Step 2: Select Interview Mode
draw_rounded_box(3.8, 24.5, 15.4, 7.5, "(HR / Behavioral /\nTechnical)", title="Select Interview\nMode", fontsize=8.8)
draw_arrow((11.5, 24.5), (11.5, 21.0))

# Step 3: Choose Question Type & Difficulty
draw_rounded_box(3.8, 12.5, 15.4, 7.5, "Choose Question\nType &\nDifficulty", fontsize=8.8, bold=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MIDDLE COLUMN: BACKEND EXECUTION FLOW
# ─────────────────────────────────────────────────────────────────────────────
# Question Engine
draw_rounded_box(26.0, 45.0, 18.5, 5.0, "(Question Generation)", title="Question Engine", fontsize=8.8)

# Connect UI Step 3 -> Question Engine
draw_arrow((19.2, 16.2), (24.5, 16.2))
ax.plot([24.5, 24.5], [16.2, 47.5], color="#000000", lw=1.3)
draw_arrow((24.5, 47.5), (26.0, 47.5))

# Load Questions from Store
draw_rounded_box(26.0, 35.5, 18.5, 5.8, "Load Questions from\nQuestion Engine &\nRubric Store", fontsize=8.3, bold=True)
draw_arrow((35.2, 45.0), (35.2, 41.3))

# Virtual Interviewer (Speech Synthesis)
draw_rounded_box(48.5, 45.0, 19.5, 5.0, "(Speech Synthesis)", title="Virtual Interviewer", fontsize=8.8)
draw_arrow((44.5, 47.5), (48.5, 47.5))

# Capture Student Response (MediaRecorder API)
draw_rounded_box(48.5, 36.2, 19.5, 5.2, "Capture Student Response\n(MediaRecorder API)", fontsize=8.4, bold=True)
draw_arrow((58.2, 45.0), (58.2, 41.4))

# Speech Recognition (Speech-to-Text)
draw_rounded_box(48.5, 27.5, 19.5, 5.0, "(Speech-to-Text)", title="Speech Recognition", fontsize=8.8)
draw_arrow((58.2, 36.2), (58.2, 32.5))

# Transcript of Response
draw_rounded_box(50.0, 20.0, 16.5, 4.0, "Transcript of Response", fontsize=8.6, bold=True)
draw_arrow((58.2, 27.5), (58.2, 24.0))

# Parallel Evaluation Container
content_bullets = ["Uses rubric", "Concept coverage", "Relevance", "STAR / behavioral"]
delivery_bullets = ["Pace", "Pauses", "Filler words", "Duration"]

p_box = patches.FancyBboxPatch(
    (25.5, 9.8), 45.5, 8.4,
    boxstyle="round,pad=0.2,rounding_size=0.6",
    facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.3
)
ax.add_patch(p_box)
ax.text(48.2, 17.0, "Parallel Evaluation", ha="center", va="center", fontsize=9.2, fontweight="bold")

# Inner box: Content Engine
draw_rounded_box(26.5, 10.3, 21.0, 5.8, "", title="Content Engine (Answer Quality)", bullets=content_bullets, fontsize=7.8)

# Inner box: Delivery Engine
draw_rounded_box(49.0, 10.3, 21.0, 5.8, "", title="Delivery Engine (Speaking Analysis)", bullets=delivery_bullets, fontsize=7.8)

# Connect Transcript -> Parallel Evaluation
draw_arrow((58.2, 20.0), (58.2, 18.2))

# Database (SQLite) inside Backend at bottom
draw_cylinder(48.2, 5.2, 26.0, 4.2, "Database (SQLite)", "Store questions, transcripts, scores and feedback")
draw_arrow((48.2, 9.8), (48.2, 7.3))


# ─────────────────────────────────────────────────────────────────────────────
# 4. RIGHT COLUMN: POST-PROCESSING & FEEDBACK
# ─────────────────────────────────────────────────────────────────────────────
# Connect Parallel Evaluation right side -> Decision Diamond
ax.plot([71.0, 74.0], [14.0, 14.0], color="#000000", lw=1.3)
ax.plot([74.0, 74.0], [14.0, 44.5], color="#000000", lw=1.3)
draw_arrow((74.0, 44.5), (78.0, 44.5))

# Decision Diamond: Need Follow-up?
draw_diamond(82.5, 44.5, 9.0, 5.6, "Need\nFollow-up?", fontsize=7.8)

# Decision: Yes -> Generate Follow-up Question
draw_rounded_box(90.0, 42.0, 7.2, 5.0, "Generate\nFollow-up\nQuestion", fontsize=7.2, bold=True)
draw_arrow((87.0, 44.5), (90.0, 44.5), label="Yes", label_pos=(0, 0.9))

# Decision: No -> Counterfactual Audit Checks
draw_arrow((82.5, 41.7), (82.5, 37.5), label="No", label_pos=(-1.1, 0))

# Loop from Generate Follow-up Question back to Question Engine
ax.plot([97.2, 98.4], [44.5, 44.5], color="#000000", lw=1.2)
ax.plot([98.4, 98.4], [44.5, 54.0], color="#000000", lw=1.2)
ax.plot([98.4, 35.2], [54.0, 54.0], color="#000000", lw=1.2)
draw_arrow((35.2, 54.0), (35.2, 50.0))

# Counterfactual Audit Checks Box
audit_bullets = [
    "Repeated-run consistency",
    "Paraphrase stability",
    "Delivery invariance",
    "Content sensitivity",
    "STT robustness"
]
draw_rounded_box(76.8, 22.5, 20.0, 14.2, "", title="Counterfactual Audit Checks\n(Evaluation Reliability)", bullets=audit_bullets, fontsize=8.0)

# Generate Explainable Feedback
draw_rounded_box(76.8, 14.8, 20.0, 5.0, "Generate Explainable\nFeedback", fontsize=8.5, bold=True)
draw_arrow((86.8, 22.5), (86.8, 19.8))

# Final Feedback Dashboard
draw_rounded_box(76.8, 7.0, 20.0, 5.2, "Final Feedback Dashboard\n(Scores, Analysis & Suggestions)", fontsize=8.0, bold=True)
draw_arrow((86.8, 14.8), (86.8, 12.2))

# Connect Final Feedback Dashboard back to Database (SQLite)
ax.plot([76.8, 73.5], [9.6, 9.6], color="#000000", lw=1.3)
ax.plot([73.5, 73.5], [5.2, 5.2], color="#000000", lw=1.3)
draw_arrow((73.5, 5.2), (61.2, 5.2))

# Save image
out_path = Path(r"C:\Users\Admin\Downloads\GEN\place\interviewiq\app\static\interviewiq_workflow.png")
artifact_path = Path(r"C:\Users\Admin\.gemini\antigravity\brain\e2c57e40-3913-45d5-ab03-95d86c5e38c5\interviewiq_workflow.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
plt.savefig(artifact_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
plt.close()

print(f"High-resolution diagram saved to:\n1. {out_path}\n2. {artifact_path}")
