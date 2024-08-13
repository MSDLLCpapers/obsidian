"Analysis utility functions for examining metrics over the context of an optimization campaign"

from obsidian.parameters import Param_Continuous
from obsidian.optimizer import Optimizer
import numpy as np
import pandas as pd


def calc_ofat_ranges(optimizer: Optimizer,
                     threshold: float,
                     X_ref: pd.DataFrame | pd.Series | None = None,
                     PI_range: float = 0.95,
                     steps: int = 100,
                     response_id: int = 0,
                     calc_interacts: bool = True):
    """
    Calculates an OFAT design space using confidence bounds around the optimizer prediction. Also
    includes a matrix of interaction scores.

    Args:
        optimizer (Optimizer): The optimizer object which contains a surrogate that has been fit to data
            and can be used to make predictions.
        X_ref (pd.DataFrame): The reference data point from which the OFAT variations are calculated.
        threshold (float): The response value threshold (minimum value) which would be considered passing for OFAT variations.
        PI_range (float, optional): The prediction interval coverage (fraction of density)
        steps (int, optional): The number of steps to use in the search for the OFAT boundaries.
            The default value is ``100``.
        response_id (int, optional): The index of the relevant response within the fitted optimizer object.
            The default value is ``0``.
        calc_interacts (bool, optional): Whether or not to return the interaction matrix; default is ``True``.

    Returns:
        ofat_ranges (pd.DataFrame): A dataframe describing the min/max OFAT values using each LB, UB, and average prediction.
            Values are scaled in the (0,1) space based on optimizer.X_space.
        cor (np.array): A matrix of interaction values between every combination of two parameters.
            Each value is the fractional reduction in size for the acceptable range envelope created by a 2-factor variation,
            in comparison to the corresponding two independent 1-factor variations. As such, diagonal elements are 0.
    """

    ofat_ranges = []
    response_name = optimizer.target[response_id].name

    if X_ref is None:
        X_ref = optimizer.X_space.mean()
    if isinstance(X_ref, pd.Series):
        X_ref = X_ref.to_frame().T

    # Calculate 1D OFAT ranges
    for p in optimizer.X_space:
        if isinstance(p, Param_Continuous):
            
            X_span = np.linspace(0, 1, steps)
            X_sim = pd.DataFrame(np.repeat(X_ref.values, repeats=steps, axis=0), columns=X_ref.columns)
            X_sim[p.name] = p.unit_demap(X_span)
            
            df_pred = optimizer.predict(X_sim, PI_range=PI_range)
            lb = df_pred[response_name + ' lb']
            ub = df_pred[response_name + ' ub']
            pred_mu = df_pred[response_name + ' (pred)']
        
            row = {'Name': p.name, 'PI Range': PI_range, 'Threshold': threshold, 'Response': response_name}
            labels = ['Mu', 'LB', 'UB']

            for label, y in zip(labels, [pred_mu, lb, ub]):
                pass_ids = np.where(y > threshold)
                pass_vals = X_sim[p.name].iloc[pass_ids]

                row['Min_'+label] = p.encode(pass_vals.min())
                row['Max_'+label] = p.encode(pass_vals.max())
            ofat_ranges.append(row)

    ofat_ranges = pd.DataFrame(ofat_ranges).set_index('Name')

    # Calculate the correlation matrix as 2-FI range / diagional of 1-FI box
    if calc_interacts:
        cor = []

        # Calculate with a nested loop of parameters
        for pi in optimizer.X_space:
            cor_j = []

            if np.isnan(ofat_ranges['Min_Mu'][pi.name]):
                cor.append([np.nan]*len(optimizer.X_space))
                continue
                
            # Enumerate a grid over the passing range at the MEAN
            Xi_pass_span = pi.unit_demap(np.linspace(ofat_ranges['Min_Mu'][pi.name],
                                                     ofat_ranges['Max_Mu'][pi.name], steps))

            for pj in optimizer.X_space:
                
                if np.isnan(ofat_ranges['Min_Mu'][pj.name]):
                    cor_j.append([np.nan]*len(optimizer.X_space))
                    continue
            
                Xj_pass_span = pi.unit_demap(np.linspace(ofat_ranges['Min_Mu'][pj.name],
                                                         ofat_ranges['Max_Mu'][pj.name], steps))

                # Set up a simulation dataframe where these parameters will co-vary
                X_sim_cor = pd.DataFrame(np.repeat(X_ref.values, repeats=steps, axis=0), columns=X_ref.columns)
                X_sim_cor[pj.name] = Xj_pass_span
                X_sim_cor[pi.name] = Xi_pass_span

                # Predict the responses, and extract the target one
                pred_mu_cor_all = optimizer.predict(X_sim_cor)
                pred_mu_cor = pred_mu_cor_all.iloc[:, response_id]
                cor_passing = np.where(pred_mu_cor > threshold)[0]

                # Want to calculate the number of steps along the diagonal which pass
                # A value of 0 for cor_j means that the two parameters are independent
                if len(cor_passing) > 0:
                    start = cor_passing[0]
                    stop = cor_passing[-1]
                    pass_ij = (stop-start)/(steps-1)
                else:
                    pass_ij = 0
                cor_j.append(1 - pass_ij)

            cor.append(cor_j)
        cor = np.array(cor)
    else:
        cor = None

    return ofat_ranges, cor
