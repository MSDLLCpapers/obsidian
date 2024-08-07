import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import Figure
from obsidian.campaign import Campaign
import numpy as np


def visualize_inputs(campaign: Campaign) -> Figure:
    """
    Visualizes the input variables of a campaign.

    Args:
        campaign (Campaign): The campaign object containing the input data.

    Returns:
        Figure: The matplotlib Figure object containing the visualization.
    """
    n_dim = campaign.X_space.n_dim

    fig = plt.figure(constrained_layout=True, figsize=(2*(n_dim+1), 2.5))
    gs = gridspec.GridSpec(ncols=n_dim+1, nrows=1, figure=fig)

    X0 = campaign.data[list(campaign.X_space.X_names)]
    X_u = campaign.X_space.unit_map(X0)
    m_exp = X0.shape[0]

    for i, param in enumerate(X0.columns):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(X0[param], 'o')
        ax.set_xlabel('Experiment')
        ax.set_title(param)
        ax.set_xticks(np.around(np.linspace(0, m_exp, 5)))

    ax = fig.add_subplot(gs[0, -1])
    plt.imshow(abs(X_u.corr()), cmap='viridis')
    ax.set_xticks(range(n_dim))
    ax.set_yticks(range(n_dim))
    ax.set_xticklabels(X0.columns, rotation=90, ha='right')
    ax.set_yticklabels(X0.columns)
    plt.colorbar()
    plt.title('Correlation Plot')

    return fig
