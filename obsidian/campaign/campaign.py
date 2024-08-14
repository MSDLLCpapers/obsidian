"""Campaign class definition"""

from obsidian.parameters import ParamSpace, Target
from obsidian.optimizer import Optimizer, BayesianOptimizer
from obsidian.experiment import ExpDesigner
from obsidian.objectives import Objective, Objective_Sequence, obj_class_dict
from obsidian.exceptions import IncompatibleObjectiveError
import obsidian

import pandas as pd
import torch
import warnings


class Campaign():
    """
    Base class for tracking optimization progress and other metrics
    over multiple iterations.

    Attributes:
        X_space (ParamSpace): The parameter space for the campaign.
        data (pd.DataFrame): The data collected during the campaign.
        optimizer (Optimizer): The optimizer used for optimization.
        designer (ExpDesigner): The experimental designer used for experiment design.
        iter (int): The current iteration number.
        seed (int): The seed for random number generation.
    
    Properties:
        m_exp (int): The number of observations in campaign.data
        y (pd.Series): The response data in campaign.data
        y_names (list): The names of the response data columns
        f (pd.Series): The transformed response data
        o (pd.Series): The objective function evaluated on f
        o_names (list): The names of the objective function columns
        X (pd.DataFrame): The input features of campaign.data
        response_max (float | pd.Series): The maximum for each response
        target (Target | list[Target]): The target(s) for optimization.
        objective (Objective, optional): The objective of the optimization campaign
    
    """
    
    def __init__(self,
                 X_space: ParamSpace,
                 target: Target | list[Target],
                 optimizer: Optimizer | None = None,
                 designer: ExpDesigner | None = None,
                 objective: Objective | None = None,
                 seed: int | None = None):
        
        self.X_space = X_space
        self.data = pd.DataFrame()
        
        optimizer = BayesianOptimizer(X_space, seed=seed) if optimizer is None else optimizer
        self.set_optimizer(optimizer)

        designer = ExpDesigner(X_space, seed=seed) if designer is None else designer
        self.set_designer(designer)
        
        self.set_target(target)
        self.set_objective(objective)
        
        # Non-object attributes
        self.iter = 0
        self.seed = seed
        self.version = obsidian.__version__

    def add_data(self, df: pd.DataFrame):
        """
        Adds data to the campaign.

        Args:
            Z_i (pd.DataFrame): The data to be added to the campaign.
        
        Raises:
            KeyError: If all X_names are not in the dataset
            KeyError: If all y_names are not in the dataset
        """
        
        if not all(name in df.columns for name in self.X_space.X_names):
            raise KeyError('Input dataset does not contain all of the required parameter names')
        if not all(name in df.columns for name in self.y_names):
            raise KeyError('Input dataset does not contain all of the required response target names')
        
        new_data = df

        if 'Iteration' not in new_data.columns:
            new_data['Iteration'] = self.iter
        else:
            self.iter = int(new_data['Iteration'].max())

        self.iter += 1
        self.data = pd.concat([self.data, new_data], axis=0, ignore_index=True)
        self.data.index.name = 'Observation ID'
        self.data.index = self.data.index.astype('int')
        
        if self.optimizer.is_fit:
            self._analyze()

    def clear_data(self):
        """Clears campaign data"""
        self.data = pd.DataFrame()

    @property
    def optimizer(self) -> Optimizer:
        """Campaign Optimizer"""
        return self._optimizer
    
    def set_optimizer(self, optimizer: Optimizer):
        """Sets the campaign optimizer"""
        self._optimizer = optimizer
        
    @property
    def designer(self) -> ExpDesigner:
        """Campaign Experimental Designer"""
        return self._designer
    
    def set_designer(self, designer: ExpDesigner):
        """Sets the campaign experiment designer"""
        self._designer = designer

    @property
    def objective(self) -> Objective | None:
        """Campaign Objective function"""
        return self._objective
    
    def _eval_objective(self):
        """Evaluates objective and appends it to campaign data"""
        df_o = self.o
        for col in df_o.columns:
            self.data[col] = df_o[col].values
            self.o_names = [col for col in self.data.columns if 'Objective' in col]
    
    def set_objective(self, objective: Objective | None):
        """(Re)sets the campaign objective function"""
        self._objective = objective
        if not self.data.empty:
            # Remove previous objective evaluations
            self.data = self.data.drop(
                columns=[col for col in self.data.columns if 'Objective' in col]
                )
            if self.optimizer.is_fit:
                self._analyze()
                
    def clear_objective(self):
        """Clears the campaign objective function"""
        self._objective = None
                
    @property
    def target(self):
        """Campaign experimental target(s)"""
        return self._target

    def set_target(self,
                   target: Target | list[Target]):
        """
        Sets the experimental target context for the campaign.

        Args:
            target (Target | list[Target] | None): The target or list of targets to set.
            
        """
        if isinstance(target, Target):
            self._target = [target]
        else:
            self._target = target

        self.y_names = [t.name for t in self._target]
        self.n_response = len(self.y_names)

    @property
    def _is_mo(self) -> bool:
        """
        Boolean flag for multi-output
        """
        if self.objective:
            return self.objective._is_mo
        else:
            return self.n_response > 1
    
    @property
    def m_exp(self) -> int:
        """
        Number of observations in training data
        """
        return self.data.shape[0]

    @property
    def y(self) -> pd.Series | pd.DataFrame:
        """
        Experimental response data

        """
        if not self.data.empty:
            return self.data[self.y_names]
        else:
            return None

    @property
    def response_max(self) -> float | pd.Series:
        """
        Maximum response data in training set
        """
        return self.y.max()

    @property
    def f(self) -> pd.Series | pd.DataFrame:
        """
        Experimental response data, in transformed space
        """
        f = pd.concat([t.transform_f(self.y[t.name]) for t in self.target], axis=1)
        return f

    @property
    def o(self) -> pd.Series | pd.DataFrame:
        """
        Objective function evaluated on f
        """
        if self.objective:
            try:
                x = self.X_space.encode(self.X).values
                o = self.objective(torch.tensor(self.f.values).unsqueeze(0),
                                   X=torch.tensor(x)).squeeze(0)
                if o.ndim < 2:
                    o = o.unsqueeze(1)  # Rearrange into m x o
                return pd.DataFrame(o.detach().cpu().numpy(),
                                    columns=[f'Objective {o_i+1}' for o_i in range(o.shape[1])])
            except Exception:
                raise IncompatibleObjectiveError('Objective(s) did not successfully execute on sample')
        else:
            return None
    
    @property
    def out(self) -> pd.Series | pd.DataFrame:
        """
        Returns the objective function as appropriate, else the response data
        """
        if self.objective and self.optimizer.is_fit:
            return self.o
        else:
            return self.y
    
    @property
    def X(self) -> pd.DataFrame:
        """
        Feature columns of the training data
        """
        return self.data[list(self.X_space.X_names)]
            
    def save_state(self) -> dict:
        """
        Saves the state of the Campaign object as a dictionary.

        Returns:
            dict: A dictionary containing the saved state of the Campaign object.
        """
        obj_dict = {}
        obj_dict['X_space'] = self.X_space.save_state()
        obj_dict['optimizer'] = self.optimizer.save_state()
        obj_dict['data'] = self.data.to_dict()
        obj_dict['target'] = [t.save_state() for t in self.target]
        if self.objective:
            obj_dict['objective'] = self.objective.save_state()
        obj_dict['seed'] = self.seed

        return obj_dict
    
    @classmethod
    def load_state(cls,
                   obj_dict: dict):
        """
        Loads the state of the campaign from a dictionary.

        Args:
            cls (Campaign): The class object.
            obj_dict (dict): A dictionary containing the campaign state.

        Returns:
            Campaign: A new campaign object with the loaded state.
        """
        
        if 'objective' in obj_dict:
            if obj_dict['objective']['name'] == 'Objective_Sequence':
                new_objective = Objective_Sequence.load_state(obj_dict['objective'])
            else:
                obj_class = obj_class_dict[obj_dict['objective']['name']]
                new_objective = obj_class.load_state(obj_dict['objective'])
        else:
            new_objective = None
        
        new_campaign = cls(X_space=ParamSpace.load_state(obj_dict['X_space']),
                           target=[Target.load_state(t_dict) for t_dict in obj_dict['target']],
                           optimizer=BayesianOptimizer.load_state(obj_dict['optimizer']),
                           objective=new_objective,
                           seed=obj_dict['seed'])
        new_campaign.data = pd.DataFrame(obj_dict['data'])
        new_campaign.data.index = new_campaign.data.index.astype('int')
        
        new_campaign.iter = new_campaign.data['Iteration'].astype('int').max()

        return new_campaign

    def __repr__(self):
        """String representation of object"""
        return f"obsidian Campaign for {getattr(self,'y_names', None)}; {getattr(self,'m_exp', 0)} observations"

    def initialize(self, **design_kwargs):
        """
        Maps ExpDesigner.initialize method
        """
        return self.designer.initialize(**design_kwargs)
    
    def fit(self):
        """
        Maps Optimizer.fit method

        Raises:
            ValueError: If no data has been registered to the campaign
        """

        if self.m_exp <= 0:
            raise ValueError('Must register data before fitting')

        self.optimizer.fit(self.data, target=self.target)

    def suggest(self, **optim_kwargs):
        """
        Maps Optimizer.suggest method
        """
        if self.optimizer.is_fit:
            try:
                X, eval = self.optimizer.suggest(objective=self.objective, **optim_kwargs)
                return (X, eval)
            except Exception:
                warnings.warn('Optimization failed')
                return None
        else:
            warnings.warn('Optimizer is not fit to data. Suggesting initial experiments.', UserWarning)
            X0 = self.initialize()
            return X0

    def evaluate(self, X_suggest: pd.DataFrame):
        """
        Maps Optimizer.evaluate method
        """
        return self.optimizer.evaluate(X_suggest, objective=self.objective)

    def _profile_hv(self):
        """
        Calculate and assign the hypervolume values to each iteration in the data.

        Returns:
            None
        """
        iters = self.data['Iteration'].unique()
        hv = {}
        
        for i in iters:
            iter_index = self.data.query(f'Iteration <= {i}').index
            out_iter = self.out.loc[iter_index, :]
            out_iter = torch.tensor(out_iter.values).to(self.optimizer.device)
            hv[i] = self.optimizer.hypervolume(out_iter)
        
        self.data['Hypervolume (iter)'] = self.data.apply(lambda x: hv[x['Iteration']], axis=1)
        self.data['Pareto Front'] = self.optimizer.pareto(torch.tensor(self.out.values).to(self.optimizer.device))
        
        return

    def _profile_max(self):
        """
        Calculate the maximum values achieved for targets at each iteration

        Returns:
            None
        """
        
        # Remove previous max-profiling
        self.data = self.data.drop(
            columns=[col for col in self.data.columns if '(max) (iter)' in col]
        )
        
        for out in self.out.columns:
            self.data[out+' (max) (iter)'] = self.data.apply(
                lambda x: self.data.query(f'Iteration<={x["Iteration"]}')[out].max(), axis=1
            )

        return
    
    def _analyze(self):
        """
        Analyzes the campaign data for practical optimization performance metrics

        Returns:
            None
        """
        if self.objective:
            self._eval_objective()
        self._profile_max()
        if self._is_mo:
            self._profile_hv()
        else:
            # Remove previous HV-profiling
            self.data = self.data.drop(
                columns=[col for col in self.data.columns
                         if 'Hypervolume' in col or 'Pareto' in col]
            )
            
        return
