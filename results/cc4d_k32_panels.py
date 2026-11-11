"""save_panels(fig, ax, path_prefix): save each axes of a 1x2 figure as its own PNG (<prefix>_a.png, <prefix>_b.png)."""
import matplotlib.pyplot as plt


def save_panels(fig, axes, prefix, dpi=130):
    fig.tight_layout(); r = fig.canvas.get_renderer()
    for ax, s in zip(axes, "ab"):
        box = ax.get_tightbbox(r).transformed(fig.dpi_scale_trans.inverted()).padded(0.05)
        fig.savefig(f"{prefix}_{s}.png", dpi=dpi, bbox_inches=box)
    plt.close(fig)
