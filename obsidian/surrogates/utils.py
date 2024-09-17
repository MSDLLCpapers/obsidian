"""Utility functions for surrogate model handling"""

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module, Parameter
import torch.optim as optim


def check_parameter_change(params: list[Parameter],
                           prev_params: list[Parameter]) -> float:
    """Check the average change in model parameters."""
    
    changes = []
    for param, prev_param in zip(params, prev_params):
        changes.append(torch.norm(param.data - prev_param.data))
    
    total_change = sum(changes)
    avg_change = total_change / len(list(params))

    return avg_change


def fit_pytorch(model: Module,
                X: Tensor,
                y: Tensor,
                loss_fcn: Module | None = None,
                verbose: bool = False,
                max_iter: int = 5000) -> None:
    """
    Fits a PyTorch model to the given input data.
    Args:
        model (Module): The PyTorch model to be trained.
        X (Tensor): The input data.
        y (Tensor): The target data.
        loss_fcn (Module, optional): The loss function to be used.
            If not provided, the Mean Squared Error (MSE) loss function will
            be used. Defaults to ``None``.
        verbose (bool, optional): Whether to print the loss at each epoch.
            Defaults to ``False``.
    """
            
    if loss_fcn is None:
        loss_fcn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(),
                           lr=1e-2,
                           weight_decay=1e-2)

    # Set up early stoppping
    avg_change = 1e10
    param_tol = 1e-2
    converge_buffer = 5
    converge_count = 0
    
    model.train()
    for epoch in range(max_iter):
  
        prev_params = [param.clone().detach() for param in model.parameters()]
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fcn(output, y)
        loss.backward()
        optimizer.step()

        # Check change
        current_params = [param.clone().detach() for param in model.parameters()]
        avg_change = check_parameter_change(current_params, prev_params)

        if (epoch % 50 == 0 and verbose):
            print(f'Epoch {epoch}: Loss {loss.item()}')
        
        # If the parameter change is under tolerance, increment the counter
        if avg_change < param_tol:
            converge_count += 1
        else:
            converge_count = 0

        # Auto-converge when the count reaches the desired length
        if converge_count == converge_buffer:
            if verbose:
                print(f'Converged at epoch {epoch}')
            break

    model.eval()
    return
