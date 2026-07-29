"""Worker script for subprocess-based RNG reproducibility test"""

import sys
import pandas as pd
from obsidian.campaign import Campaign
from obsidian.parameters import Target
from obsidian.experiment import Simulator
from obsidian.experiment.benchmark import shifted_parab
from obsidian.tests.param_configs import X_sp_default


def run_campaign_cycles(seed, n_cycles):
    """Run a complete campaign with multiple optimization cycles"""
    X_space = X_sp_default
    target = Target(name="Response", f_transform="Standard", aim="max")
    sim = Simulator(X_space, shifted_parab, eps=0.05, rng=seed)
    campaign = Campaign(X_space, target, seed=seed)

    # Initialize
    X0 = campaign.initialize(m_initial=5, method="LHS")
    y0 = sim.simulate(X0)
    campaign.add_data(pd.concat([X0, y0], axis=1))
    campaign.fit()

    # Run cycles
    for _ in range(n_cycles):
        X_suggest, _ = campaign.suggest(m_batch=2, acquisition=["EI"])
        y_suggest = sim.simulate(X_suggest)
        campaign.add_data(pd.concat([X_suggest, y_suggest], axis=1))
        campaign.fit()

    return campaign.data


if __name__ == "__main__":
    seed = int(sys.argv[1])
    n_cycles = int(sys.argv[2])
    output_file = sys.argv[3]

    data = run_campaign_cycles(seed, n_cycles)
    data.to_csv(output_file, index=False)
