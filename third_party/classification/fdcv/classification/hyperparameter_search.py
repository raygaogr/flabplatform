"""
Copyright (C) 2025 dsl.
"""
from typing import Dict, List, Union, Optional, Any, Callable, Tuple
import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import logging
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from .image_classifier import Classifier
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective
import warnings
import torch
from .classifier_data import ClassificationData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ['HyperparameterSearch']

class HyperparameterSearch:
    """Class for performing hyperparameter search for classification models."""

    def __init__(self, data_module: ClassificationData, metric: str = 'val_loss', max_epochs: int = 10, patience: int = 3, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize hyperparameter search.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader (optional)
            metric: Metric to optimize ('val_acc', 'val_loss')
            max_epochs: Maximum epochs for each trial
            patience: Early stopping patience
            output_dir: Directory to save results
        """
        self.data_module = data_module
        self.metric = metric
        self.max_epochs = max_epochs
        self.patience = patience

        if output_dir is None:
            output_dir = Path('hparam_search_results')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.best_params = None
        self.best_score = float('-inf') if 'acc' in metric else float('inf')

        # Add attributes for bayesian optimization
        self.space = None
        self.dimensions = None
        self.bo_results = None

    def grid_search(self, param_grid: Dict[str, List[Any]], n_trials: Optional[int] = None, **model_kwargs: Any) -> Dict[str, Any]:
        """
        Perform grid search over hyperparameter space.

        Args:
            param_grid: Dictionary of parameters and their possible values
            n_trials: Maximum number of trials (optional)
            **model_kwargs: Additional arguments passed to Classifier
            
        Returns:
            Dict containing best parameters and search results
        """
        # Generate all possible combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))

        if n_trials is not None:
            combinations = random.sample(combinations, min(n_trials, len(combinations)))

        logger.info(f"Starting grid search with {len(combinations)} combinations")

        for i, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo)) 
            logger.info(f"\nTrial {i}/{len(combinations)}")
            logger.info(f"Parameters: {params}")

            score = self._evaluate_params(params, model_kwargs)
            self.results.append({**params, 'score': score})

            # Update best parameters
            is_better = score > self.best_score if 'acc' in self.metric else score < self.best_score
            if is_better:
                self.best_score = score
                self.best_params = params

        self._save_results(method='grid_search')
        self._plot_results(method='grid_search')
        return {'best_params': self.best_params, 'results': self.results}

    def random_search(self, param_distributions: Dict[str, Union[List[Any], Callable]], n_trials: int, **model_kwargs: Any) -> Dict[str, Any]:
        """
        Perform random search over hyperparameter space.
        
        Args:
            param_distributions: Dictionary of parameters and their distributions
            n_trials: Number of random trials
            **model_kwargs: Additional arguments passed to Classifier
            
        Returns:
            Dict containing best parameters and search results
        """
        logger.info(f"Starting random search with {n_trials} trials")

        for i in range(n_trials):
            # Sample parameters
            params = {}
            for name, dist in param_distributions.items():
                if callable(dist):
                    params[name] = dist()
                else:
                    params[name] = random.choice(dist)
                    
            logger.info(f"\nTrial {i+1}/{n_trials}")
            logger.info(f"Parameters: {params}")

            score = self._evaluate_params(params, model_kwargs)
            self.results.append({**params, 'score': score})

            # Update best parameters
            is_better = score > self.best_score if 'acc' in self.metric else score < self.best_score
            if is_better:
                self.best_score = score
                self.best_params = params
                
        self._save_results(method='random_search')
        self._plot_results(method='random_search')
        return {'best_params': self.best_params, 'results': self.results}

    def bayesian_optimization(
        self,
        param_space: Dict[str, Union[List[Any], Tuple[float, float, str]]],
        n_trials: int = 50,
        n_initial_points: int = 10,
        acq_func: str = 'gp_hedge',
        **model_kwargs: Any
    ) -> Dict[str, Any]:
        """
        Perform Bayesian optimization over hyperparameter space.
        
        Args:
            param_space: Dictionary of parameters and their spaces. Format:
                - For categorical: list of options
                - For continuous: (low, high, 'uniform' or 'log-uniform')
                - For integer: (low, high, 'uniform')
            n_trials: Number of optimization iterations
            n_initial_points: Number of initial random points
            acq_func: Acquisition function ('gp_hedge', 'EI', 'PI', 'UCB')
                'gp_hedge': 一种集成策略，会尝试多种不同的采集函数。
                'EI': 期望提升（Expected Improvement），优先挖掘那些预期能带来最大性能提升的区域。
                'PI': 提升概率（Probability of Improvement），优先挖掘那些有概率超过当前最好结果的区域。
                'UCB': 上置信界限（Upper Confidence Bound），平衡探索和利用，会考虑均值较高和不确定性较高的区域。
            **model_kwargs: Additional arguments passed to Classifier
            
        Returns:
            Dict containing best parameters and optimization results
        """
        # Convert parameter space to skopt dimensions
        dimensions = []
        dim_names = []
        
        for name, space in param_space.items():
            if isinstance(space, (list, tuple)):
                if isinstance(space[0], (int, float)) and len(space) == 3:
                    # Continuous or integer space
                    low, high, prior = space
                    if isinstance(low, int) and isinstance(high, int):
                        dim = Integer(low, high, prior, name=name)
                    else:
                        dim = Real(low, high, prior, name=name)
                else:
                    # Categorical space
                    dim = Categorical(space, name=name)
            else:
                raise ValueError(f"Invalid space specification for parameter {name}")
                
            dimensions.append(dim)
           # self.dimensions.append((name, dim))
            dim_names.append(name)

        self.space = dimensions
        
        # Define objective function
        @use_named_args(dimensions=dimensions)
        def objective(**params):
            try:
                score = self._evaluate_params(params, model_kwargs)
                # Convert to minimization problem
                return -score if 'acc' in self.metric else score
            except Exception as e:
                logger.error(f"Error during evaluation: {str(e)}")
                # Return worst possible score
                return float('inf')

        # Run Bayesian optimization
        logger.info(f"Starting Bayesian optimization with {n_trials} trials")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.bo_results = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=n_trials,
                n_initial_points=n_initial_points,
                acq_func=acq_func,
                noise=0.1,
                random_state=42
            )

        # Process results
        all_params = []
        for i, x in enumerate(self.bo_results.x_iters):
            params = dict(zip(dim_names, x))
            score = -self.bo_results.func_vals[i] if 'acc' in self.metric else self.bo_results.func_vals[i]
            all_params.append({**params, 'score': score})
            
        self.results = all_params
        best_idx = self.bo_results.func_vals.argmin()
        self.best_params = dict(zip(dim_names, self.bo_results.x_iters[best_idx]))
        self.best_score = -self.bo_results.fun if 'acc' in self.metric else self.bo_results.fun
        
        # Save and plot results
        self._save_results(method='bo_search')
        self._plot_results(method='bo_search')
        self._plot_optimization_results()
        
        return {
            'best_params': self.best_params,
            'results': self.results,
            'optimization_results': self.bo_results
        }


    def _evaluate_params(self, params: Dict[str, Any], model_kwargs: Dict[str, Any]) -> float:
        """Evaluate a set of hyperparameters."""
        try:
            # 更新数据模块的参数
            if 'batch_size' in params:
                self.data_module.batch_size = params['batch_size']
            if 'mixup_alpha' in params:
                self.data_module.mixup_alpha = params['mixup_alpha']
                self.data_module.mixup = True
            if 'label_smoothing' in params:
                self.data_module.label_smoothing = params['label_smoothing']
            
            # 重新设置数据模块以应用新参数
            self.data_module.setup()
            
            # 创建模型
            model = Classifier(
                **model_kwargs,
                lr=params.get('lr', 0.001),
                dropout=params.get('dropout', 0.5)
            )
            
            # 设置训练器
            trainer = Trainer(
                max_epochs=self.max_epochs,
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        patience=self.patience,
                        mode='min'
                    )
                ],
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                enable_progress_bar=True
            )
            
            # 训练模型
            trainer.fit(
                model, 
                train_dataloaders=self.data_module.train_dataloader(),
                val_dataloaders=self.data_module.val_dataloader()
            )
            
            # 获取最佳指标值
            # Get best score
            score = trainer.callback_metrics[self.metric].item()
            logger.info(f"Score ({self.metric}): {score:.4f}")
            
            return score
            
        except Exception as e:
            logger.error(f"Error during parameter evaluation: {str(e)}")
            return float('-inf') if 'acc' in self.metric else float('inf')

    def _save_results(self, method: str) -> None:
        """Save search results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results as CSV
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / f'{method}_results_{timestamp}.csv', index=False)
        
        # Save best parameters
        with open(self.output_dir / f'{method}_best_params_{timestamp}.json', 'w') as f:
            json.dump(self.best_params, f, indent=4)
            
        # Save optimization results if available
        if self.bo_results is not None:
            optimization_info = {
                'n_iterations': len(self.bo_results.x_iters),
                'best_score': float(self.best_score),
                'func_vals': self.bo_results.func_vals.tolist(),
                'space': str(self.bo_results.space),
                'x_iters': [x.tolist() if isinstance(x, np.ndarray) else x 
                           for x in self.bo_results.x_iters]
            }
            # Try to add n_calls if available
            try:
                optimization_info['n_calls'] = self.bo_results.n_calls
            except AttributeError:
                optimization_info['n_calls'] = len(self.bo_results.x_iters)
            
            with open(self.output_dir / f'{method}_optimization_results_{timestamp}.json', 'w') as f:
                json.dump(optimization_info, f, indent=4)
            
        logger.info(f"\nResults saved to {self.output_dir}")

    def _plot_results(self, method: str) -> None:
        """Plot search results."""
        df = pd.DataFrame(self.results)
        
        # Plot parameter distributions
        param_cols = [col for col in df.columns if col != 'score']
        n_params = len(param_cols)
        
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 4*n_params))
        if n_params == 1:
            axes = [axes]
            
        for ax, param in zip(axes, param_cols):
            if df[param].dtype in [np.float64, np.int64]:
                ax.scatter(df[param], df['score'])
                ax.set_xlabel(param)
                ax.set_ylabel(self.metric)
            else:
                scores = df.groupby(param)['score'].agg(['mean', 'std']).reset_index()
                ax.bar(range(len(scores)), scores['mean'], yerr=scores['std'])
                ax.set_xticks(range(len(scores)))
                ax.set_xticklabels(scores[param], rotation=45)
                ax.set_ylabel(self.metric)
                
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.output_dir / f'{method}_results_{timestamp}.png')
        plt.close()
    
    def _plot_optimization_results(self) -> None:
        """Plot Bayesian optimization specific visualizations."""
        if self.bo_results is None:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plot_convergence(self.bo_results)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'convergence_{timestamp}.png')
        plt.close()
        
        # Plot objective
        if len(self.space) <= 2:  # Only plot for 1-2 dimensions
            plt.figure(figsize=(12, 8))
            plot_objective(self.bo_results)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'objective_{timestamp}.png')
            plt.close() 
