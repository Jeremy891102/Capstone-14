"""Run the Q7 analysis for both models and stitch each figure into a 2x2 grid (rows = the two panels, left Qwen, right Argon, same y axes per row) into docs/cc4d/new_experiment/figs/.
python results/cc4d_k32_figs.py   (needs numpy, matplotlib)"""
import os, pathlib, shutil, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

H = pathlib.Path(__file__).parent
FIGS = H.parent / "docs/cc4d/new_experiment/figs"
MODELS = {"qwen": "Qwen3-VL-32B", "argon": "Gemini 4 Argon"}  # left, right
for m in MODELS:
    for script, out in (("cc4d_k32.py", f"cc4d_k32{'' if m == 'argon' else '_qwen'}_out.txt"), ("cc4d_k32_reg.py", f"cc4d_k32_reg{'' if m == 'argon' else '_qwen'}_out.txt")):
        with open(H / out, "w") as f:
            subprocess.run([sys.executable, str(H / script)], env={**os.environ, "K32_MODEL": m}, stdout=f, stderr=subprocess.STDOUT, check=True)
for name in ("q7_accuracy", "q7_abstention", "q7_position_error", "q7_separate"):
    fig, ax = plt.subplots(2, 2, figsize=(9, 7.4))
    for j, (m, label) in enumerate(MODELS.items()):
        for i, p in enumerate("ab"):
            ax[i, j].imshow(mpimg.imread(FIGS / "parts" / f"{m}_{name}_{p}.png")); ax[i, j].axis("off")
        ax[0, j].set_title(label, fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIGS / f"{name}.png", dpi=130); plt.close()
shutil.rmtree(FIGS / "parts")
