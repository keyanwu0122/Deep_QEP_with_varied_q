
"""Deep Q-Exponential Process Model with q Grid Search as Outer Loop and Seeds as Inner Loop -
heterogeneous search over 2 layers"""

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

# define the main model
class MultitaskDeepQEP(DeepQEP):
    def __init__(self, in_features, out_features, hidden_features=2, powers=None):
        super().__init__()
        if isinstance(hidden_features, int):
            layer_config = torch.cat([torch.arange(in_features, out_features, step=(out_features-in_features)/max(1,hidden_features)).type(torch.int), torch.tensor([out_features])])
        elif isinstance(hidden_features, list):
            layer_config = [in_features]+hidden_features+[out_features]

        # Set default powers if not provided
        if powers is None:
            powers = [1.0] * (len(layer_config) - 1)
        # Make sure we have enough powers for all layers
        if len(powers) < len(layer_config) - 1:
            powers = powers + [powers[-1]] * (len(layer_config) - 1 - len(powers))
        self.current_powers = powers

        layers = []
        for i in range(len(layer_config)-1):
            layers.append(DQEPLayer(
                input_dims=layer_config[i],
                output_dims=layer_config[i+1],
                mean_type='linear' if i < len(layer_config)-2 else 'constant',
                power=powers[i]
            ))
        self.num_layers = len(layers)
        self.layers = torch.nn.ModuleList(layers)

        # We're going to use a multitask likelihood
        self.likelihood = MultitaskQExponentialLikelihood(num_tasks=out_features, power=torch.tensor(powers[-1]))

    def forward(self, inputs):
        output = self.layers[0](inputs)
        for i in range(1, len(self.layers)):
            output = self.layers[i](output)
        return output

    def update_powers(self, new_powers):
        """Update the power parameters of all layers"""
        if len(new_powers) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} power values, got {len(new_powers)}")

        for i, layer in enumerate(self.layers):
            layer.update_power(new_powers[i])

        # Also update the likelihood power
        self.likelihood.power = torch.tensor(new_powers[-1])
        self.current_powers = new_powers

    def predict(self, test_x):
        with torch.no_grad():
            # The output of the model is a multitask QEP
            output = self(test_x)
            # To compute the marginal predictive NLL of each data point
            preds = self.likelihood(output).to_data_independent_dist()

        return preds.mean.mean(0), preds.variance.mean(0), output


def train_model(powers, seed, num_epochs=1000):
    """Train a model with fixed power values for a specific seed"""

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    num_tasks = train_y.size(-1)
    hidden_features = [3]

    # Calculate number of layers for power parameters
    if isinstance(hidden_features, int):
        num_layers = 1 + 1  # Input to hidden, hidden to output
    else:
        num_layers = len(hidden_features) + 1

    # Initialize model with provided powers
    model = MultitaskDeepQEP(
        in_features=train_x.shape[-1],
        out_features=num_tasks,
        hidden_features=hidden_features,
        powers=powers
    )

    # Set up optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))

    # For tracking training progress
    loss_history = []
    mae_history = []
    beginning = timeit.default_timer()

    # Training loop
    epochs_iter = tqdm.tqdm(range(num_epochs), desc=f"Training seed {seed} with powers {powers}")

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

    print(f'Seed {seed}, Powers {powers}:')
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
            'upper': upper
        }
    }


def generate_average_plots(q_values=[0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0],
                           seeds=[1, 1001, 1011, 1021, 1031, 1041, 1051, 1061, 1071, 1081],
                           num_epochs=1000,
                           plots_dir='./results/avg_power_plots'):
    """
    Generate plots for each power combination showing the average performance across all seeds.
    """
    # Create directories to store results and plots
    os.makedirs('./results', exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Calculate number of layers for power parameters
    hidden_features = [3]
    if isinstance(hidden_features, int):
        num_layers = 1 + 1  # Input to hidden, hidden to output
    else:
        num_layers = len(hidden_features) + 1

    # Generate all possible combinations of q values for each layer
    power_combinations = [tuple([q] * num_layers) for q in q_values]
    power_results = {}

    # Iterate through each power combination
    for power_combo in power_combinations:
        print(f"\nProcessing power combination: {power_combo}")

        # Data structures to accumulate results across seeds
        all_means = []
        all_lowers = []
        all_uppers = []
        all_loss_histories = []
        all_mae_histories = []
        all_metrics = {
            'MAE': [],
            'RMSE': [],
            'R2': [],
            'NLL': []
        }

        # Generate test x points for plotting
        test_x = torch.linspace(0, 1, 51).unsqueeze(-1)

        # Process each seed
        for seed in seeds:
            print(f"Training with seed {seed}...")
            # Train model and get results
            results = train_model(power_combo, seed, num_epochs)

            # Collect predictions
            all_means.append(results['predictions']['mean'].numpy())
            all_lowers.append(results['predictions']['lower'].numpy())
            all_uppers.append(results['predictions']['upper'].numpy())

            # Collect histories
            all_loss_histories.append(results['loss_history'])
            all_mae_histories.append(results['mae_history'])

            # Collect metrics
            for metric in all_metrics.keys():
                all_metrics[metric].append(results['metrics'][metric])

        # Calculate averages across seeds
        avg_mean = np.mean(all_means, axis=0)
        avg_lower = np.mean(all_lowers, axis=0)
        avg_upper = np.mean(all_uppers, axis=0)

        # Calculate average loss and MAE histories
        # First, find the minimum length (some runs might have early stopping)
        min_loss_len = min(len(hist) for hist in all_loss_histories)
        min_mae_len = min(len(hist) for hist in all_mae_histories)

        # Truncate histories to minimum length and calculate averages
        truncated_loss_histories = [hist[:min_loss_len] for hist in all_loss_histories]
        truncated_mae_histories = [hist[:min_mae_len] for hist in all_mae_histories]

        avg_loss_history = np.mean(truncated_loss_histories, axis=0)
        avg_mae_history = np.mean(truncated_mae_histories, axis=0)

        # Calculate average metrics
        avg_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}
        std_metrics = {metric: np.std(values, ddof=1) for metric, values in all_metrics.items()}

        # Store average results
        power_results[power_combo] = {
            'avg_mean': avg_mean,
            'avg_lower': avg_lower,
            'avg_upper': avg_upper,
            'avg_loss_history': avg_loss_history,
            'avg_mae_history': avg_mae_history,
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics
        }

        # Generate and save plot for this power combination
        plot_average_result(power_combo, power_results[power_combo], test_x, plots_dir)

        print(f"Power {power_combo}:")
        print(f"  Avg ± Std MAE: {avg_metrics['MAE']:.4f} ± {std_metrics['MAE']:.4f}")
        print(f"  Avg ± Std RMSE: {avg_metrics['RMSE']:.4f} ± {std_metrics['RMSE']:.4f}")
        print(f"  Avg ± Std R2: {avg_metrics['R2']:.4f} ± {std_metrics['R2']:.4f}")
        print(f"  Avg ± Std NLL: {avg_metrics['NLL']:.4f} ± {std_metrics['NLL']:.4f}")

    # Find the best power combination based on average MAE
    best_power_combo = min(power_results.keys(), key=lambda p: power_results[p]['avg_metrics']['MAE'])
    best_avg_metrics = power_results[best_power_combo]['avg_metrics']
    best_std_metrics = power_results[best_power_combo]['std_metrics']

    print("\n===== RESULTS =====")
    print(f"Best power combination: {best_power_combo}")
    print(f"Average MAE: {best_avg_metrics['MAE']:.4f} ± {best_std_metrics['MAE']:.4f}")
    print(f"Average RMSE: {best_avg_metrics['RMSE']:.4f} ± {best_std_metrics['RMSE']:.4f}")
    print(f"Average R2: {best_avg_metrics['R2']:.4f} ± {best_std_metrics['R2']:.4f}")
    print(f"Average NLL: {best_avg_metrics['NLL']:.4f} ± {best_std_metrics['NLL']:.4f}")

    return power_results, best_power_combo


def plot_average_result(power_combo, results, test_x, plots_dir):
    """Generate and save plot for average performance across seeds for a specific power combination"""
    num_tasks = 2  # Step and turning functions
    fig, axs = plt.subplots(1, num_tasks + 2, figsize=(4 * (num_tasks + 2), 4))

    # Prepare ground truth data
    test_x_np = test_x.squeeze(-1).numpy()

    # Get ground truth functions
    f = {'step': lambda ts: torch.tensor(
        [1 * (t >= 0 and t <= 1) + 0.5 * (t > 1 and t <= 1.5) + 2 * (t > 1.5 and t <= 2) for t in ts]),
         'turning': lambda ts: torch.tensor(
             [1.5 * t * (t >= 0 and t <= 1) + (3.5 - 2 * t) * (t > 1 and t <= 1.5) + (3 * t - 4) * (t > 1.5 and t <= 2)
              for t in ts])}

    # Plot truth, mean and confidence intervals for each task
    task_names = ['step', 'turning']
    for task, ax in enumerate(axs[:num_tasks]):
        # Plot ground truth
        ax.plot(test_x_np, f[task_names[task]](test_x * 2).numpy(), 'r--')

        # Plot model prediction (mean)
        ax.plot(test_x_np, results['avg_mean'][:, task], 'b')

        # Plot confidence interval
        ax.fill_between(
            test_x_np,
            results['avg_lower'][:, task],
            results['avg_upper'][:, task],
            alpha=0.5
        )

        # Add simulated observations (assuming train_x is defined globally)
        # This would need to be adapted if train_x is not accessible
        train_x = torch.linspace(0, 1, 100).unsqueeze(-1)
        train_y = torch.stack([
            f['step'](train_x * 2),
            f['turning'](train_x * 2),
        ], -1)
        noise = torch.randn(train_x.size()) * 0.1
        ax.plot(train_x.squeeze(-1).detach().numpy(),
                (train_y[:, task] + noise).detach().numpy(), 'k*')

        ax.set_ylim([-.5, 3])
        ax.legend(['Truth', 'Observed Data', 'Mean', 'Confidence'], fontsize=12)
        ax.set_title(f'Task {task + 1}: {task_names[task]} function\nPower: {power_combo}', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Plot average loss history
    axs[num_tasks].plot(results['avg_loss_history'])
    axs[num_tasks].set_title(f'Avg Loss History\nPower: {power_combo}', fontsize=16)
    axs[num_tasks].set_xlabel('Epoch', fontsize=14)
    axs[num_tasks].set_ylabel('Negative ELBO', fontsize=14)
    axs[num_tasks].tick_params(axis='both', which='major', labelsize=14)

    # Plot average MAE history
    axs[num_tasks + 1].plot(results['avg_mae_history'], 'g-')
    axs[num_tasks + 1].set_title(f'Avg MAE History\nPower: {power_combo}', fontsize=16)
    axs[num_tasks + 1].set_xlabel('Epoch', fontsize=14)
    axs[num_tasks + 1].set_ylabel('MAE', fontsize=14)
    axs[num_tasks + 1].tick_params(axis='both', which='major', labelsize=14)

    # Add metrics as text in the plot
    metrics_text = (
        f"MAE: {results['avg_metrics']['MAE']:.4f} ± {results['std_metrics']['MAE']:.4f}\n"
        f"RMSE: {results['avg_metrics']['RMSE']:.4f} ± {results['std_metrics']['RMSE']:.4f}\n"
        f"R2: {results['avg_metrics']['R2']:.4f} ± {results['std_metrics']['R2']:.4f}\n"
        f"NLL: {results['avg_metrics']['NLL']:.4f} ± {results['std_metrics']['NLL']:.4f}"
    )
    fig.text(0.5, 0.01, metrics_text, ha='center', va='bottom', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    # Save the figure
    fig.suptitle(f'Average Performance for Power = {power_combo} (Across {10} Seeds)', fontsize=20)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for the text
    plt.savefig(f'{plots_dir}/avg_power_{power_combo}.png', bbox_inches='tight')
    plt.close(fig)


def train_model(powers, seed, num_epochs=1000):
    """Train a model with fixed power values for a specific seed"""

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Generate data
    train_x = torch.linspace(0, 1, 100)

    f = {'step': lambda ts: torch.tensor(
        [1 * (t >= 0 and t <= 1) + 0.5 * (t > 1 and t <= 1.5) + 2 * (t > 1.5 and t <= 2) for t in ts]),
         'turning': lambda ts: torch.tensor(
             [1.5 * t * (t >= 0 and t <= 1) + (3.5 - 2 * t) * (t > 1 and t <= 1.5) + (3 * t - 4) * (t > 1.5 and t <= 2)
              for t in ts])}

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

    num_tasks = train_y.size(-1)
    hidden_features = [3]

    # Calculate number of layers for power parameters
    if isinstance(hidden_features, int):
        num_layers = 1 + 1  # Input to hidden, hidden to output
    else:
        num_layers = len(hidden_features) + 1

    # Initialize model with provided powers
    model = MultitaskDeepQEP(
        in_features=train_x.shape[-1],
        out_features=num_tasks,
        hidden_features=hidden_features,
        powers=powers
    )

    # Set up optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))

    # For tracking training progress
    loss_history = []
    mae_history = []
    beginning = timeit.default_timer()

    # Training loop
    epochs_iter = tqdm.tqdm(range(num_epochs), desc=f"Training seed {seed} with powers {powers}")

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
    MAE = torch.mean(torch.abs(mean - test_y)).item()
    RMSE = torch.mean(torch.pow(mean - test_y, 2)).sqrt().item()
    PSD = torch.mean(var.sqrt()).item()
    NLL = -model.likelihood.log_marginal(test_y, mdl_output).mean(0).mean().item()

    from sklearn.metrics import r2_score
    R2 = r2_score(test_y.numpy(), mean.numpy())

    print(f'Seed {seed}, Powers {powers}:')
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
            'upper': upper
        }
    }

if __name__ == '__main__':
    print("This script should be imported and used in a context where the model classes are defined.")
    print("Run the following in a context with full model definitions:")
    print("results, best_power = generate_average_plots()")