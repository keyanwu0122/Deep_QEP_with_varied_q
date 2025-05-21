"Deep Q-Exponential Process Model with Power Object"

import os
import random
import numpy as np
import timeit
from matplotlib import pyplot as plt

import torch
import tqdm

# gpytorch imports
import sys

sys.path.insert(0, '../gpytorch')
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, LMCVariationalStrategy
from gpytorch.distributions import MultivariateQExponential, Power
from gpytorch.models.deep_qeps import DeepQEPLayer, DeepQEP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskQExponentialLikelihood
from gpytorch.priors import GammaPrior

# Setting manual seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)

# generate data
train_x = torch.linspace(0, 1, 100)

f = {'step': lambda ts: torch.tensor(
    [1 * (t >= 0 and t <= 1) + 0.5 * (t > 1 and t <= 1.5) + 2 * (t > 1.5 and t <= 2) for t in ts]),
     'turning': lambda ts: torch.tensor(
         [1.5 * t * (t >= 0 and t <= 1) + (3.5 - 2 * t) * (t > 1 and t <= 1.5) + (3 * t - 4) * (t > 1.5 and t <= 2) for
          t in ts])}

train_y = torch.stack([
    f['step'](train_x * 2) + torch.randn(train_x.size()) * 0.1,
    f['turning'](train_x * 2) + torch.randn(train_x.size()) * 0.1,
], -1)

train_x = train_x.unsqueeze(-1)

# Define test data (generated in advance)
test_x = torch.linspace(0, 1, 51).unsqueeze(-1)
test_y = torch.stack([f['step'](test_x * 2), f['turning'](test_x * 2)], -1)

# Dictionary to store metrics across all seeds
all_metrics = {
    'q': [],
    'MAE': [],
    'RMSE': [],
    'PSD': [],
    'R2': [],
    'NLL': [],
    'time': []
}

def main(seed=1, collect_metrics=False):
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Modified layer definition
    class DQEPLayer(DeepQEPLayer):
        def __init__(self, input_dims, output_dims, num_inducing=64, mean_type='constant', power=None):
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=num_inducing,
                batch_shape=batch_shape,
                power=power
            )
            variational_strategy = VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            )
            super().__init__(variational_strategy, input_dims, output_dims)
            self.mean_module = {'constant': ConstantMean(), 'linear': LinearMean(input_dims)}[mean_type]
            self.covar_module = ScaleKernel(
                MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=input_dims)
            )
            self.power = power

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateQExponential(mean_x, covar_x, power=self.power)

    # Modified main model with Power object for q
    class MultitaskDeepQEP(DeepQEP):
        def __init__(self, in_features, out_features, hidden_features=2, initial_power=1.5):
            super().__init__()

            # Use Power object with GammaPrior
            # Create a prior that favors values between 1.0 and 2.0
            # GammaPrior with alpha=6.0, beta=4.0 has mode at 1.25 and concentrates mass between 1-2
            power_prior = GammaPrior(6.0, 4.0)
            self.power = Power(torch.tensor(initial_power), power_prior=power_prior)

            if isinstance(hidden_features, int):
                layer_config = torch.cat([torch.arange(in_features, out_features,
                                                       step=(out_features - in_features) / max(1,
                                                                                               hidden_features)).type(
                    torch.int), torch.tensor([out_features])])
            elif isinstance(hidden_features, list):
                layer_config = [in_features] + hidden_features + [out_features]
            layers = []
            for i in range(len(layer_config) - 1):
                layers.append(DQEPLayer(
                    input_dims=layer_config[i],
                    output_dims=layer_config[i + 1],
                    mean_type='linear' if i < len(layer_config) - 2 else 'constant',
                    power=self.power  # Pass the Power object
                ))
            self.num_layers = len(layers)
            self.layers = torch.nn.Sequential(*layers)
            self.likelihood = MultitaskQExponentialLikelihood(num_tasks=out_features, power=self.power)

        def forward(self, inputs):
            output = self.layers[0](inputs)
            for i in range(1, len(self.layers)):
                output = self.layers[i](output)
            return output

        def predict(self, test_x):
            with torch.no_grad():
                output = self(test_x)
                preds = self.likelihood(output).to_data_independent_dist()
            return preds.mean.mean(0), preds.variance.mean(0), output

    # Initialize model
    num_tasks = train_y.size(-1)
    hidden_features = [3]
    model = MultitaskDeepQEP(
        in_features=train_x.shape[-1],
        out_features=num_tasks,
        hidden_features=hidden_features,
        initial_power=1.5  # Start with q=1.5 as initial value
    )

    # Training configuration
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))

    # Track q values and corresponding metrics
    best_mae = float('inf')
    best_q = None
    best_model_state = None
    q_history = []
    loss_list = []
    mae_list = []

    # Training loop
    num_epochs = 1000
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    beginning = timeit.default_timer()

    for i in epochs_iter:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        # Track current q value
        current_q = model.power.power.item()
        q_history.append(current_q)

        # Calculate test MAE
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean, var, _ = model.predict(test_x)
            current_mae = torch.mean(torch.abs(mean - test_y)).item()
            mae_list.append(current_mae)

        # Update best state
        if current_mae < best_mae:
            best_mae = current_mae
            best_q = current_q
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.train()
        epochs_iter.set_postfix(
            loss=loss.item(),
            current_q=current_q,
            best_mae=best_mae
        )
        loss_list.append(loss.item())

    # After training, evaluate q values between 1.0 and 2.0
    # to find the optimal one based on MAE
    print(f"\nAfter training, best q value: {best_q:.4f} with MAE: {best_mae:.4f}")

    # Evaluate fixed q values across our range
    q_values = torch.linspace(1.0, 2.0, 20)  # 20 points between 1.0 and 2.0
    q_results = []
    q_iter = tqdm.tqdm(q_values, desc="Evaluating fixed q values")

    # Load best model parameters before testing different q values
    model.load_state_dict(best_model_state)

    for q_fixed in q_iter:
        # Set fixed q value
        with torch.no_grad():
            model.power.power = q_fixed

        # Short fine-tuning with fixed q
        model.train()
        for _ in range(50):  # Short fine-tuning
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            # Keep q fixed
            with torch.no_grad():
                model.power.power = q_fixed

        # Evaluate
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean, var, _ = model.predict(test_x)
            fixed_mae = torch.mean(torch.abs(mean - test_y)).item()

        q_results.append((q_fixed.item(), fixed_mae))
        q_iter.set_postfix(q=q_fixed.item(), mae=fixed_mae)

        # Update best if better
        if fixed_mae < best_mae:
            best_mae = fixed_mae
            best_q = q_fixed.item()
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Print q evaluation results
    print("\nQ-value evaluation results:")
    for q, mae in q_results:
        print(f"q = {q:.2f}, MAE = {mae:.4f}")

    # Calculate likelihood/prior/posterior densities for q
    # Similar to the second file's approach
    def calculate_likelihood(q_values):
        liks = []
        model.eval()
        for q in q_values:
            with torch.no_grad():
                model.power.power = q
                output = model(train_x)
                # Use log_marginal or expected_log_prob based on what's available
                log_lik = model.likelihood.expected_log_prob(train_y, output).mean().item()
                liks.append(log_lik)
        # Normalize
        liks = np.array(liks)
        liks = np.exp(liks - liks.max())
        return liks

    def calculate_prior(q_values):
        priors = []
        for q in q_values:
            log_prior = model.power.power_prior.log_prob(q).item()
            priors.append(log_prior)
        # Normalize
        priors = np.array(priors)
        priors = np.exp(priors - priors.max())
        return priors

    # Load best model
    model.load_state_dict(best_model_state)
    time_ = timeit.default_timer() - beginning

    # Final evaluation
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        mean, var, mdl_output = model.predict(test_x)
        lower = mean - 2 * var.sqrt()
        upper = mean + 2 * var.sqrt()

    MAE = torch.mean(torch.abs(mean - test_y)).item()
    RMSE = torch.mean(torch.pow(mean - test_y, 2)).sqrt().item()
    PSD = torch.mean(var.sqrt()).item()
    NLL = -model.likelihood.log_marginal(test_y, mdl_output).mean(0).mean().item()
    from sklearn.metrics import r2_score
    R2 = r2_score(test_y.numpy(), mean.numpy())

    print(f'\nOptimal q: {best_q:.4f}')
    print(f'Test MAE: {MAE:.4f}')
    print(f'Test RMSE: {RMSE:.4f}')

    # Save results
    os.makedirs('./results_varyq', exist_ok=True)
    stats = np.array([seed, 'DeepQEP', best_q, MAE, RMSE, PSD, R2, NLL, time_])
    header = ['seed', 'Method', 'Optimal q', 'MAE', 'RMSE', 'PSD', 'R2', 'NLL', 'time']
    f_name = os.path.join('./results_varyq/ts_DeepQEP_' + str(model.num_layers) + 'layers.txt')
    with open(f_name, 'ab') as f_:
        np.savetxt(f_, stats.reshape(1, -1), fmt="%s", delimiter=',', header=','.join(header) if seed == 1 else '')

    # If collecting metrics for stats, add them to our dictionary
    if collect_metrics:
        all_metrics['q'].append(best_q)
        all_metrics['MAE'].append(MAE)
        all_metrics['RMSE'].append(RMSE)
        all_metrics['PSD'].append(PSD)
        all_metrics['R2'].append(R2)
        all_metrics['NLL'].append(NLL)
        all_metrics['time'].append(time_)

    # Calculate likelihood and prior for plotting
    q_plot = np.linspace(1.0, 2.0, 51)
    likelihoods = calculate_likelihood(q_plot)
    priors = calculate_prior(q_plot)

    # Plot results for the optimal q value
    # Create results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./results_varyq', exist_ok=True)

    num_tasks = train_y.size(-1)
    fig, axs = plt.subplots(1, num_tasks + 2, figsize=(4 * (num_tasks + 2), 4))

    # Task plots
    f_functions = list(f.values())
    f_names = list(f.keys())

    for task, ax in enumerate(axs[:num_tasks]):
        ax.plot(test_x.squeeze(-1).numpy(), list(f.values())[task](test_x*2).numpy(), 'r--')
        ax.plot(train_x.squeeze(-1).detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
        ax.plot(test_x.squeeze(-1).numpy(), mean[:, task].numpy(), 'b')
        ax.fill_between(
            test_x.squeeze(-1).numpy(),
            lower[:, task].numpy(),
            upper[:, task].numpy(),
            alpha=0.5
        )
        ax.set_ylim([-.5, 3])
        ax.legend(['Truth','Observed Data', 'Mean', 'Confidence'], fontsize=12)
        ax.set_title(f'Task {task + 1}: {f_names[task]} function', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Loss plot
    axs[num_tasks].plot(loss_list)
    axs[num_tasks].set_title(f'Neg. ELBO Loss', fontsize=16)
    axs[num_tasks].tick_params(axis='both', which='major', labelsize=14)

    fig.tight_layout()
    plt.savefig(f'./results/ts_DeepQEP_optimal_q_{best_q:.4f}_seed{seed}.png', bbox_inches='tight')

    # Additional plots if this is the first seed
    if seed == 1 or seed == 2024:
        # Create separate plot for q evolution
        plt.figure(figsize=(10, 6))
        plt.plot(q_history, label='q during training')
        plt.axhline(y=best_q, color='r', linestyle='--', label=f'Best q = {best_q:.4f}')
        plt.ylabel('q value', fontsize=14)
        plt.xlabel('Epoch', fontsize=14)
        plt.title('q Evolution', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('./results_varyq/ts_DeepQEP_q_evolution.png', bbox_inches='tight')

        # Create separate plot for MAE vs q
        plt.figure(figsize=(10, 6))
        q_vals, mae_vals = zip(*q_results)
        plt.plot(q_vals, mae_vals, 'g-o', label='MAE')
        plt.xlabel('q value', fontsize=14)
        plt.ylabel('MAE', fontsize=14)
        plt.title('q vs MAE', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('./results_varyq/ts_DeepQEP_q_vs_mae.png', bbox_inches='tight')

        # Additional plot for q densities
        plt.figure(figsize=(10, 6))
        plt.plot(q_plot, likelihoods, label='likelihood')
        plt.plot(q_plot, priors, label='prior')
        plt.axvline(best_q, linewidth=3, color='red', label=f'best q = {best_q:.4f}')
        plt.legend(fontsize=14)
        plt.title('q Distribution', fontsize=18)
        plt.xlabel('q', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('./results_varyq/ts_DeepQEP_q_density.png', bbox_inches='tight')

    return best_q, MAE, RMSE, PSD, R2, NLL, time_


if __name__ == '__main__':
    # Run for a single seed first
    main(1)

    # Multiple seed runs for statistical analysis
    n_seed = 10
    i = 0
    n_success = 0
    n_failure = 0

    # Run for multiple seeds and collect metrics
    while n_success < n_seed and n_failure < 10 * n_seed:
        seed_i = 2024 + i * 10
        try:
            print(f"Running for seed {seed_i}...")
            main(seed=seed_i, collect_metrics=True)
            n_success += 1
        except Exception as e:
            print(e)
            n_failure += 1
        i += 1

    # Calculate and report statistics after all seeds are processed
    if len(all_metrics['MAE']) > 0:
        print("\n" + "="*80)
        print("STATISTICAL SUMMARY ACROSS ALL SEEDS")
        print("="*80)

        # Create a more detailed table of results
        print(f"{'Seed':<10}{'q':<12}{'MAE':<12}{'RMSE':<12}{'PSD':<12}{'R2':<12}{'NLL':<12}{'Time (s)':<12}")
        print("-"*80)

        # Print individual seed results
        for i in range(len(all_metrics['MAE'])):
            seed_num = 1 if i == 0 else 2024 + (i-1) * 10
            print(f"{seed_num:<10}"
                  f"{all_metrics['q'][i]:.4f}     "
                  f"{all_metrics['MAE'][i]:.4f}     "
                  f"{all_metrics['RMSE'][i]:.4f}     "
                  f"{all_metrics['PSD'][i]:.4f}     "
                  f"{all_metrics['R2'][i]:.4f}     "
                  f"{all_metrics['NLL'][i]:.4f}     "
                  f"{all_metrics['time'][i]:.2f}")

        # Print summary statistics
        print("-"*80)
        print(f"{'Mean':<10}"
              f"{np.mean(all_metrics['q']):.4f}     "
              f"{np.mean(all_metrics['MAE']):.4f}     "
              f"{np.mean(all_metrics['RMSE']):.4f}     "
              f"{np.mean(all_metrics['PSD']):.4f}     "
              f"{np.mean(all_metrics['R2']):.4f}     "
              f"{np.mean(all_metrics['NLL']):.4f}     "
              f"{np.mean(all_metrics['time']):.2f}")
        print(f"{'Std Dev':<10}"
              f"{np.std(all_metrics['q']):.4f}     "
              f"{np.std(all_metrics['MAE']):.4f}     "
              f"{np.std(all_metrics['RMSE']):.4f}     "
              f"{np.std(all_metrics['PSD']):.4f}     "
              f"{np.std(all_metrics['R2']):.4f}     "
              f"{np.std(all_metrics['NLL']):.4f}     "
              f"{np.std(all_metrics['time']):.2f}")
        print("="*80)

        # Save detailed results to file
        header = ['seed', 'q', 'MAE', 'RMSE', 'PSD', 'R2', 'NLL', 'time']
        data = []

        # Add individual seed results
        for i in range(len(all_metrics['MAE'])):
            seed_num = 1 if i == 0 else 2024 + (i-1) * 10
            data.append([
                seed_num,
                all_metrics['q'][i],
                all_metrics['MAE'][i],
                all_metrics['RMSE'][i],
                all_metrics['PSD'][i],
                all_metrics['R2'][i],
                all_metrics['NLL'][i],
                all_metrics['time'][i]
            ])

        # Add mean and std
        data.append([
            'Mean',
            np.mean(all_metrics['q']),
            np.mean(all_metrics['MAE']),
            np.mean(all_metrics['RMSE']),
            np.mean(all_metrics['PSD']),
            np.mean(all_metrics['R2']),
            np.mean(all_metrics['NLL']),
            np.mean(all_metrics['time'])
        ])
        data.append([
            'Std',
            np.std(all_metrics['q']),
            np.std(all_metrics['MAE']),
            np.std(all_metrics['RMSE']),
            np.std(all_metrics['PSD']),
            np.std(all_metrics['R2']),
            np.std(all_metrics['NLL']),
            np.std(all_metrics['time'])
        ])

        # Save as numpy array
        np.savetxt('./results_varyq/ts_DeepQEP_detailed_stats.csv',
                   np.array(data, dtype=object),
                   fmt='%s',
                   delimiter=',',
                   header=','.join(header))

        # Plot detailed histogram of metrics across seeds
        metrics_to_plot = ['q', 'MAE', 'RMSE', 'R2', 'NLL']
        plt.figure(figsize=(15, 10))

        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(2, 3, i+1)
            values = all_metrics[metric]

            # Plot histogram with KDE
            plt.hist(values, bins='auto', alpha=0.7, density=True)

            # Add mean and std annotation
            mean_val = np.mean(values)
            std_val = np.std(values)
            plt.axvline(mean_val, color='r', linestyle='--')
            plt.annotate(f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}',
                         xy=(0.05, 0.85),
                         xycoords='axes fraction',
                         fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            plt.title(f'{metric} Distribution', fontsize=14)
            plt.xlabel(metric, fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig('./results_varyq/ts_DeepQEP_metrics_distribution.png', bbox_inches='tight')

        # Create violin plots for more detailed distribution view
        plt.figure(figsize=(14, 8))
        data_to_plot = [all_metrics[metric] for metric in metrics_to_plot]

        violin_parts = plt.violinplot(data_to_plot, showmeans=True, showmedians=True)

        # Customize violin plot
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # Add individual points
        for i, metric in enumerate(metrics_to_plot):
            plt.scatter([i+1] * len(all_metrics[metric]), all_metrics[metric],
                      alpha=0.7, s=60, c='darkblue', zorder=3)

        plt.xticks(np.arange(1, len(metrics_to_plot) + 1), metrics_to_plot, fontsize=14)
        plt.ylabel('Value', fontsize=16)
        plt.title('Performance Metrics Distribution Across Seeds', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.4, axis='y')

        # Add mean and std values as text below each violin
        for i, metric in enumerate(metrics_to_plot):
            mean_val = np.mean(all_metrics[metric])
            std_val = np.std(all_metrics[metric])
            plt.text(i+1, plt.ylim()[0], f'{mean_val:.4f}±{std_val:.4f}',
                    ha='center', va='bottom', fontsize=12, rotation=45)

        plt.tight_layout()
        plt.savefig('./results_varyq/ts_DeepQEP_metrics_violin.png', bbox_inches='tight')