import numpy as np
import matplotlib.pyplot as plt
import logging


class QuantumPlotter:
    def __init__(self):
        self._setup_style()

    def _setup_style(self):
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.edgecolor": "0.3",
            "axes.linewidth": 1,
            "grid.alpha": 0.3,
            "figure.dpi": 100
        })

    def plot_eigenvalues_and_singular_values(self, results, save_path="eval.png", plot_title=None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if plot_title:
            fig.suptitle(plot_title, fontsize=16)

        self._plot_eigenvalues_complex_plane(axes[0], results)
        self._plot_eigenvalues_vs_singular_values(axes[1], results)

        plt.tight_layout(pad=2.5)
        plt.savefig(save_path)
        logging.info(f"Saved plot to {save_path}")
        return fig

    def _plot_eigenvalues_complex_plane(self, ax, results):
        scatter = ax.scatter(
            results['eigenvalues_real'], results['eigenvalues_imag'],
            c=results['eigenvalues_magnitude'], cmap="viridis",
            s=50, alpha=0.8, edgecolor="k", linewidth=0.2
        )
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        ax.set_title("Eigenvalues in the Complex Plane")
        ax.axhline(0, color="black", lw=0.8, alpha=0.4)
        ax.axvline(0, color="black", lw=0.8, alpha=0.4)

        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), "r--", lw=1.2, alpha=0.7, label="Unit Circle")
        ax.legend(frameon=True, loc="upper right")

        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Magnitude")

    def _plot_eigenvalues_vs_singular_values(self, ax, results):
        sorted_eigen_mag = np.sort(results['eigenvalues_magnitude'])[::-1]
        indices = np.arange(len(results['singular_values']))

        ax.plot(indices, sorted_eigen_mag, "-", color="tab:blue", lw=0.8, alpha=0.4)
        ax.scatter(indices, sorted_eigen_mag, color="tab:blue", s=10, alpha=0.8,
                   label=r"$|\lambda|$", zorder=3)
        ax.plot(indices, results['singular_values'], "-", color="tab:red", lw=0.8, alpha=0.4)
        ax.scatter(indices, results['singular_values'], color="tab:red", s=10, alpha=0.8,
                   label="Singular Values", zorder=3)

        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title("Eigenvalue Magnitudes vs. Singular Values")
        ax.legend(frameon=True, loc="upper right")

        ymax = max(sorted_eigen_mag.max(), results['singular_values'].max()) * 1.05
        ax.set_ylim(bottom=0, top=ymax)