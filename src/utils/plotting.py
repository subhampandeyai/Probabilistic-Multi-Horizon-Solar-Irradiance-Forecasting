"""
Shared matplotlib styling.

Centralises figure style and saving so no individual script sets its own
rcParams and figures stay visually consistent.

    from utils.plotting import setup_style, save_figure, COLORS
    setup_style()
    fig, ax = plt.subplots()
    save_figure(fig, "decomposition", stage="stage1")
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# -- Academic color palette ---
COLORS = {
    'primary':    '#1D3557',   # Dark navy
    'accent1':    '#E63946',   # Red
    'accent2':    '#2A9D8F',   # Teal
    'accent3':    '#457B9D',   # Steel blue
    'accent4':    '#F4A261',   # Orange
    'accent5':    '#6A4C93',   # Purple
    'accent6':    '#264653',   # Dark teal
    'neutral':    '#CBD5E0',   # Light gray
    'bg':         '#FAFBFC',   # Off-white background
    'grid':       '#E8ECF0',   # Grid lines
}

MODEL_COLORS = {
    'XGBoost':        '#E63946',
    'Random Forest':  '#457B9D',
    'Ridge':          '#2A9D8F',
    'LSTM':           '#264653',
    'GRU':            '#6A4C93',
    'MLP':            '#F4A261',
    'Transformer':    '#E9C46A',
    'SVR':            '#CBD5E0',
    'Persistence':    '#999999',
    'FAME':           '#1D3557',
}

COMPONENT_COLORS = {
    'cA':  '#E63946',   # Trend (approximation)
    'cD3': '#2A9D8F',   # Low frequency
    'cD2': '#F4A261',   # Mid frequency
    'cD1': '#457B9D',   # High frequency
}


def setup_style():
    """Apply consistent matplotlib style globally."""
    from utils.config import cfg

    plot_cfg = cfg.get("plotting", {})
    plt.rcParams.update({
        'font.family':       plot_cfg.get('font_family', 'serif'),
        'font.size':         plot_cfg.get('font_size', 10),
        'axes.titlesize':    12,
        'axes.labelsize':    11,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'legend.fontsize':   8.5,
        'figure.facecolor':  'white',
        'axes.facecolor':    COLORS['bg'],
        'axes.edgecolor':    COLORS['neutral'],
        'grid.color':        COLORS['grid'],
        'grid.linewidth':    0.5,
        'axes.grid':         True,
        'grid.alpha':        0.6,
        'savefig.dpi':       plot_cfg.get('dpi', 300),
        'savefig.bbox':      'tight',
        'savefig.facecolor': 'white',
        'figure.dpi':        150,
    })


def save_figure(fig, name: str, stage: str = None):
    """
    Save figure to outputs/plots/ with consistent naming.
    Figures are artifacts, saved to tracked location.
    """
    from utils.config import get_path
    plot_dir = get_path("outputs_plots")

    if stage:
        plot_dir = plot_dir / stage
        plot_dir.mkdir(parents=True, exist_ok=True)

    from utils.config import cfg
    fmt = cfg.get("plotting", {}).get("format", "png")
    filepath = plot_dir / f"{name}.{fmt}"

    fig.savefig(filepath, facecolor='white', edgecolor='none', pad_inches=0.15)
    plt.close(fig)
    print(f"    OK Plot: {filepath.relative_to(get_path('outputs_plots').parent.parent)}")
    return filepath
