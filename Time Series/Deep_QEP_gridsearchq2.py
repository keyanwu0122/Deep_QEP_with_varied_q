
"""Deep Q-Exponential Process Model with q Grid Search as Outer Loop and Seeds as Inner Loop - 3 Layers
   Modified to use the same power value across all layers"""

import os
import random
import numpy as np
import timeit
from matplotlib import pyplot as plt
import itertools
import copy

import torch
import tqdm

# gpytorch imports
import sys
sys.path.insert(0,'../gpytorch')
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, LMCVariationalStrategy
from gpytorch.distributions import MultivariateQExponential
from gpytorch.models.deep_qeps import DeepQEPLayer, DeepQEP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskQExponentialLikelihood

# Setting manual seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)

# generate data
train_x = torch.linspace(0, 1, 100)

f = {'step': lambda ts: torch.tensor([1*(t>=0 and t<=1) + 0.5*(t>1 and t<=1.5) + 2*(t>1.5 and t<=2) for t in ts]),
     'turning': lambda ts: torch.tensor([1.5*t*(t>=0 and t<=1) + (3.5-2*t)*(t>1 and t<=1.5) + (3*t-4)*(t>1.5 and t<=2) for t in ts])}

train_y = torch.stack([
    f['step'](train_x * 2) + torch.randn(train_x.size()) * 0.1,
    f['turning'](train_x * 2) + torch.randn(train_x.size()) * 0.1,
], -1)

train_x = train_x.unsqueeze(-1)

# Generate validation data for MAE evaluation
val_x = torch.linspace(0, 1, 50).unsqueeze(-1)
val_y = torch.stack([
    f['step'](val_x * 2),
    f['turning'](val_x * 2),
], -1)


# Here's a simple standard layer
class DQEPLayer(DeepQEPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=64, mean_type='constant', power=1.0):
        self.power = torch.tensor(power)
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape,
            power=self.power
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = {'constant': ConstantMean(), 'linear': LinearMean(input_dims)}[mean_type]
        self.covar_module = ScaleKernel(
            MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, power=self.power)

    def update_power(self, new_power):
        """Update the power parameter of the layer"""
        self.power = torch.tensor(new_power)
        # Also update the power in variational distribution
        self.variational_strategy.variational_distribution.power = self.power

# define the main model - Modified to have exactly 3 layers
class MultitaskDeepQEP(DeepQEP):
    def __init__(self, in_features, out_features, hidden_features=[5, 4], power=1.0):
        super().__init__()
        # Fixed 3 layers configuration: input -> hidden1 -> hidden2 -> output
        layer_config = [in_features] + hidden_features + [out_features]

        # Ensure we have exactly 3 layers (4 dimensions including input and output)
        assert len(layer_config) == 4, "Model must have exactly 3 layers (2 hidden layers + output layer)"

        # Using the same power for all layers
        self.current_power = power

        layers = []
        for i in range(len(layer_config)-1):
            layers.append(DQEPLayer(
                input_dims=layer_config[i],
                output_dims=layer_config[i+1],
                mean_type='linear' if i < len(layer_config)-2 else 'constant',
                power=power
            ))
        self.num_layers = len(layers)
        self.layers = torch.nn.ModuleList(layers)

        # We're going to use a multitask likelihood with the same power
        self.likelihood = MultitaskQExponentialLikelihood(num_tasks=out_features, power=torch.tensor(power))

    def forward(self, inputs):
        output = self.layers[0](inputs)
        for i in range(1, len(self.layers)):
            output = self.layers[i](output)
        return output

    def update_power(self, new_power):
        """Update the power parameter of all layers to the same value"""
        for layer in self.layers:
            layer.update_power(new_power)

        # Also update the likelihood power
        self.likelihood.power = torch.tensor(new_power)
        self.current_power = new_power

    def predict(self, test_x):
        with torch.no_grad():
            # The output of the model is a multitask QEP
            output = self(test_x)
            # To compute the marginal predictive NLL of each data point
            preds = self.likelihood(output).to_data_independent_dist()

        return preds.mean.mean(0), preds.variance.mean(0), output


def train_model(power, seed, num_epochs=1000):
    """Train a model with fixed power value for a specific seed"""

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    num_tasks = train_y.size(-1)
    # Fixed hidden features for 3 layers model
    hidden_features = [5, 4]  # 2 hidden layers

    # Initialize model with provided power (same for all layers)
    model = MultitaskDeepQEP(
        in_features=train_x.shape[-1],
        out_features=num_tasks,
        hidden_features=hidden_features,
        power=power
    )

    # Set up optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))

    # For tracking training progress
    loss_history = []
    mae_history = []
    beginning = timeit.default_timer()

    # Training loop
    epochs_iter = tqdm.tqdm(range(num_epochs), desc=f"Training seed {seed} with power {power}")

    for epoch in epochs_iter:
        # Forward pass for training
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        # Calculate MAE on validation set
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean, _, _ = model.predict(val_x)
            mae = torch.mean(torch.abs(mean - val_y)).item()
        model.train()

        # Track progress
        loss_history.append(loss.item())
        mae_history.append(mae)

        epochs_iter.set_postfix(loss=loss.item(), mae=mae)

    training_time = timeit.default_timer() - beginning

    # Evaluation
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51).unsqueeze(-1)
        mean, var, mdl_output = model.predict(test_x)
        lower = mean - 2 * var.sqrt()
        upper = mean + 2 * var.sqrt()

    # Truth
    test_y = torch.stack([
        f['step'](test_x * 2),
        f['turning'](test_x * 2),
    ], -1)

    # Compute metrics
    MAE = torch.mean(torch.abs(mean-test_y)).item()
    RMSE = torch.mean(torch.pow(mean-test_y, 2)).sqrt().item()
    PSD = torch.mean(var.sqrt()).item()
    NLL = -model.likelihood.log_marginal(test_y, mdl_output).mean(0).mean().item()

    from sklearn.metrics import r2_score
    R2 = r2_score(test_y.numpy(), mean.numpy())

    print(f'Seed {seed}, Power {power}:')
    print(f'  Test MAE: {MAE:.4f}')
    print(f'  Test RMSE: {RMSE:.4f}')
    print(f'  Test R2: {R2:.4f}')
    print(f'  Training time: {training_time:.2f}s')

    return {
        'model': model,
        'loss_history': loss_history,
        'mae_history': mae_history,
        'metrics': {
            'MAE': MAE,
            'RMSE': RMSE,
            'PSD': PSD,
            'R2': R2,
            'NLL': NLL,
            'training_time': training_time
        },
        'predictions': {
            'mean': mean,
            'lower': lower,
            'upper': upper,
            'test_x': test_x
        }
    }


def plot_combined_results(all_loss_histories, all_mae_histories, all_predictions, power,
                          save_dir='./results/power_plots'):
    """Create plots for all seeds with a given power value"""
    os.makedirs(save_dir, exist_ok=True)

    # We'll use the first seed's predictions for visualization
    first_seed_pred = all_predictions[0]

    # Create figure with four subplots in one row
    num_tasks = train_y.size(-1)
    fig, axs = plt.subplots(1, num_tasks+2, figsize=(4 * (num_tasks+2), 4))

    test_x = torch.linspace(0, 1, 51).unsqueeze(-1)
    test_y = torch.stack([
        f['step'](test_x * 2),
        f['turning'](test_x * 2),
    ], -1)

    for task, ax in enumerate(axs[:num_tasks]):
        ax.plot(test_x.squeeze(-1).numpy(), list(f.values())[task](test_x*2).numpy(), 'r--')
        ax.plot(train_x.squeeze(-1).detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
        ax.plot(test_x.squeeze(-1).numpy(), first_seed_pred['mean'][:, task].numpy(), 'b')
        ax.fill_between(
            test_x.squeeze(-1).numpy(),
            first_seed_pred['lower'][:, task].numpy(),
            first_seed_pred['upper'][:, task].numpy(),
            alpha=0.5
        )
        ax.set_ylim([-.5, 3])
        ax.legend(['Truth','Observed Data', 'Mean', 'Confidence'], fontsize=12)
        ax.set_title(f'Task {task + 1}: '+list(f.keys())[task]+' function', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Plot average loss history across all seeds
    avg_loss = np.mean(all_loss_histories, axis=0)
    axs[num_tasks].plot(avg_loss)
    axs[num_tasks].set_title('Average Loss History', fontsize=20)
    axs[num_tasks].set_xlabel('Epoch', fontsize=14)
    axs[num_tasks].set_ylabel('Negative ELBO', fontsize=14)
    axs[num_tasks].tick_params(axis='both', which='major', labelsize=14)

    # Plot average MAE history across all seeds
    avg_mae = np.mean(all_mae_histories, axis=0)
    axs[num_tasks+1].plot(avg_mae, 'g-')
    axs[num_tasks+1].set_title('Average MAE History', fontsize=20)
    axs[num_tasks+1].set_xlabel('Epoch', fontsize=14)
    axs[num_tasks+1].set_ylabel('MAE', fontsize=14)
    axs[num_tasks+1].tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    power_str = str(power).replace('.', 'p')
    plt.savefig(f'{save_dir}/combined_plots_power_{power_str}.png', bbox_inches='tight')
    plt.close(fig)


def main_grid_search(q_values=[0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0],
                    seeds=[1, 11, 21, 31, 41],
                    num_epochs=1000,
                    plot_dir='./results/power_plots'):
    """
    Main function that tests each q value across multiple seeds and plots results.
    Modified to use the same power for all layers.
    """
    # Create directories to store results and plots
    os.makedirs('./results', exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Store results for each q value
    q_results = {}

    # Create a summary table for metrics
    summary_table = []
    header = ['Power', 'Avg MAE', 'Std MAE', 'Avg RMSE', 'Std RMSE', 'Avg R2', 'Std R2', 'Avg NLL', 'Std NLL']

    # Outer loop - test different q values
    for power in q_values:
        print(f"\nTesting power value: {power}")

        # Store metrics for each seed with this power
        seed_results = []
        all_loss_histories = []
        all_mae_histories = []
        all_predictions = []

        # Inner loop - test with different seeds
        for seed in seeds:
            print(f"Training with seed {seed}...")
            results = train_model(power, seed, num_epochs)
            seed_results.append(results['metrics'])
            all_loss_histories.append(results['loss_history'])
            all_mae_histories.append(results['mae_history'])
            all_predictions.append(results['predictions'])

        # Calculate average metrics across seeds
        avg_metrics = {
            'MAE': np.mean([r['MAE'] for r in seed_results]),
            'RMSE': np.mean([r['RMSE'] for r in seed_results]),
            'PSD': np.mean([r['PSD'] for r in seed_results]),
            'R2': np.mean([r['R2'] for r in seed_results]),
            'NLL': np.mean([r['NLL'] for r in seed_results]),
            'training_time': np.mean([r['training_time'] for r in seed_results])
        }

        # Calculate standard deviation for each metric
        std_metrics = {
            'MAE_std': np.std([r['MAE'] for r in seed_results], ddof=1),
            'RMSE_std': np.std([r['RMSE'] for r in seed_results], ddof=1),
            'PSD_std': np.std([r['PSD'] for r in seed_results], ddof=1),
            'R2_std': np.std([r['R2'] for r in seed_results], ddof=1),
            'NLL_std': np.std([r['NLL'] for r in seed_results], ddof=1),
            'training_time_std': np.std([r['training_time'] for r in seed_results], ddof=1)
        }

        # Store results for this q value
        q_results[power] = {
            'seed_results': seed_results,
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'all_loss_histories': all_loss_histories,
            'all_mae_histories': all_mae_histories,
            'all_predictions': all_predictions
        }

        # Add to summary table
        summary_table.append([
            str(power),
            f"{avg_metrics['MAE']:.4f}",
            f"{std_metrics['MAE_std']:.4f}",
            f"{avg_metrics['RMSE']:.4f}",
            f"{std_metrics['RMSE_std']:.4f}",
            f"{avg_metrics['R2']:.4f}",
            f"{std_metrics['R2_std']:.4f}",
            f"{avg_metrics['NLL']:.4f}",
            f"{std_metrics['NLL_std']:.4f}"
        ])

        print(f"Power {power}:")
        print(f"  Avg ± Std MAE: {avg_metrics['MAE']:.4f} ± {std_metrics['MAE_std']:.4f}")
        print(f"  Avg ± Std RMSE: {avg_metrics['RMSE']:.4f} ± {std_metrics['RMSE_std']:.4f}")
        print(f"  Avg ± Std PSD: {avg_metrics['PSD']:.4f} ± {std_metrics['PSD_std']:.4f}")
        print(f"  Avg ± Std R2: {avg_metrics['R2']:.4f} ± {std_metrics['R2_std']:.4f}")
        print(f"  Avg ± Std NLL: {avg_metrics['NLL']:.4f} ± {std_metrics['NLL_std']:.4f}")

        # Plot combined results for this power value
        plot_combined_results(all_loss_histories, all_mae_histories, all_predictions, power, plot_dir)

    # Find the best q value based on average MAE
    best_power = min(q_results.keys(), key=lambda p: q_results[p]['avg_metrics']['MAE'])
    best_avg_metrics = q_results[best_power]['avg_metrics']
    best_std_metrics = q_results[best_power]['std_metrics']

    print("\n===== RESULTS =====")
    print(f"Best power value: {best_power}")
    print(f"Average MAE: {best_avg_metrics['MAE']:.4f} ± {best_std_metrics['MAE_std']:.4f}")
    print(f"Average RMSE: {best_avg_metrics['RMSE']:.4f} ± {best_std_metrics['RMSE_std']:.4f}")
    print(f"Average PSD: {best_avg_metrics['PSD']:.4f} ± {best_std_metrics['PSD_std']:.4f}")
    print(f"Average R2: {best_avg_metrics['R2']:.4f} ± {best_std_metrics['R2_std']:.4f}")
    print(f"Average NLL: {best_avg_metrics['NLL']:.4f} ± {best_std_metrics['NLL_std']:.4f}")

    # Create a table plot for the complete results
    fig, ax = plt.subplots(figsize=(14, len(q_values) * 0.5 + 2))
    ax.axis('off')
    table = ax.table(cellText=summary_table,
                     colLabels=header,
                     loc='center',
                     cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Summary of Metrics by Power Value', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/metrics_summary_table.png', bbox_inches='tight')
    plt.close()

    # Save summary results to CSV
    np.savetxt(
        os.path.join('./results', 'power_metrics_summary.csv'),
        summary_table,
        fmt='%s',
        delimiter=',',
        header=','.join(header)
    )

    return best_power, best_avg_metrics, best_std_metrics, q_results


if __name__ == '__main__':
    # Define the q values to search over
    q_values = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]

    # Use 7 seeds as specified in the code
    seeds = [1, 11, 21, 31, 41]

    # Run experiment with plots for all q values
    best_power, best_avg_metrics, best_std_metrics, all_results = main_grid_search(
        q_values=q_values,
        seeds=seeds,
        num_epochs=1000,
        plot_dir='./results/power_plots'
    )

    print(f"\nExperiment Complete!")
    print(f"Best power value: {best_power}")
    print(f"Average metrics with standard deviation:")
    print(f"MAE: {best_avg_metrics['MAE']:.4f} ± {best_std_metrics['MAE_std']:.4f}")
    print(f"RMSE: {best_avg_metrics['RMSE']:.4f} ± {best_std_metrics['RMSE_std']:.4f}")
    print(f"PSD: {best_avg_metrics['PSD']:.4f} ± {best_std_metrics['PSD_std']:.4f}")
    print(f"R2: {best_avg_metrics['R2']:.4f} ± {best_std_metrics['R2_std']:.4f}")
    print(f"NLL: {best_avg_metrics['NLL']:.4f} ± {best_std_metrics['NLL_std']:.4f}")

    print(f"\nAll plots have been saved to the ./results/power_plots directory")