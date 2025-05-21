"""Deep Q-Exponential Process Model with q Grid Search - same q across all layer"""

import os
import random
import numpy as np
import timeit
from matplotlib import pyplot as plt

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


def main(seed=1, power=1.0):
    """Main function to train and evaluate the model with a specific power value and seed"""

    # Setting manual seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Here's a simple standard layer
    class DQEPLayer(DeepQEPLayer):
        def __init__(self, input_dims, output_dims, num_inducing=64, mean_type='constant'):
            self.power = torch.tensor(power)  # Use the power parameter from function argument
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

    # define the main model
    class MultitaskDeepQEP(DeepQEP):
        def __init__(self, in_features, out_features, hidden_features=2):
            super().__init__()
            if isinstance(hidden_features, int):
                layer_config = torch.cat([torch.arange(in_features, out_features, step=(out_features-in_features)/max(1,hidden_features)).type(torch.int), torch.tensor([out_features])])
            elif isinstance(hidden_features, list):
                layer_config = [in_features]+hidden_features+[out_features]
            layers = []
            for i in range(len(layer_config)-1):
                layers.append(DQEPLayer(
                    input_dims=layer_config[i],
                    output_dims=layer_config[i+1],
                    mean_type='linear' if i < len(layer_config)-2 else 'constant'
                ))
            self.num_layers = len(layers)
            self.layers = torch.nn.Sequential(*layers)
            self.likelihood = MultitaskQExponentialLikelihood(num_tasks=out_features, power=torch.tensor(power))

        def forward(self, inputs):
            output = self.layers[0](inputs)
            for i in range(1,len(self.layers)):
                output = self.layers[i](output)
            return output

        def predict(self, test_x):
            with torch.no_grad():
                # The output of the model is a multitask QEP
                output = self(test_x)
                # To compute the marginal predictive NLL of each data point
                preds = self.likelihood(output).to_data_independent_dist()
            return preds.mean.mean(0), preds.variance.mean(0), output

    num_tasks = train_y.size(-1)
    hidden_features = [3]
    model = MultitaskDeepQEP(in_features=train_x.shape[-1], out_features=num_tasks, hidden_features=hidden_features)

    # training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))

    loss_list = []
    num_epochs = 1000
    epochs_iter = tqdm.tqdm(range(num_epochs), desc=f"Epoch (power={power})")
    beginning = timeit.default_timer()

    for i in epochs_iter:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        epochs_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if i == 0:
            min_loss = loss_list[-1]
            optim_model = model.state_dict()
        else:
            if loss_list[-1] < min_loss:
                min_loss = loss_list[-1]
                optim_model = model.state_dict()

    time_ = timeit.default_timer() - beginning
    print(f'Training uses: {time_} seconds.')

    # load the best model
    model.load_state_dict(optim_model)

    # Make predictions
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51).unsqueeze(-1)
        mean, var, mdl_output = model.predict(test_x)
        lower = mean - 2 * var.sqrt()
        upper = mean + 2 * var.sqrt()

    # truth
    test_y = torch.stack([
        f['step'](test_x * 2),
        f['turning'](test_x * 2),
    ], -1)

    # Calculate metrics
    MAE = torch.mean(torch.abs(mean-test_y)).item()
    RMSE = torch.mean(torch.pow(mean-test_y, 2)).sqrt().item()
    PSD = torch.mean(var.sqrt()).item()
    NLL = -model.likelihood.log_marginal(test_y, mdl_output).mean(0).mean().item()
    from sklearn.metrics import r2_score
    R2 = r2_score(test_y, mean)

    print(f'Power {power}, Seed {seed}:')
    print(f'Test MAE: {MAE}')
    print(f'Test RMSE: {RMSE}')
    print(f'Test PSD: {PSD}')
    print(f'Test R2: {R2}')
    print(f'Test NLL: {NLL}')

    # save to file
    os.makedirs('./results', exist_ok=True)
    stats = np.array([MAE, RMSE, PSD, R2, NLL, time_])
    stats = np.array([seed, f'DeepQEP_q{power}']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['seed', 'Method', 'MAE', 'RMSE', 'PSD', 'R2', 'NLL', 'time']
    f_name = os.path.join(f'./results/ts_DeepQEP_q{power}_{model.num_layers}layers.txt')
    with open(f_name,'ab') as f_:
        np.savetxt(f_, stats, fmt="%s", delimiter=',', header=','.join(header) if seed==1 else '')

    # Plot results
    fig, axs = plt.subplots(1, num_tasks+1, figsize=(4 * (num_tasks+1), 4))
    for task, ax in enumerate(axs):
        if task < num_tasks:
            ax.plot(test_x.squeeze(-1).numpy(), list(f.values())[task](test_x*2).numpy(), 'r--')
            ax.plot(train_x.squeeze(-1).detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
            ax.plot(test_x.squeeze(-1).numpy(), mean[:, task].numpy(), 'b')
            ax.fill_between(test_x.squeeze(-1).numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
            ax.set_ylim([-.5, 3])
            ax.legend(['Truth','Observed Data', 'Mean', 'Confidence'], fontsize=12)
            ax.set_title(f'Task {task + 1}: '+list(f.keys())[task]+' function', fontsize=20)
        else:
            ax.plot(loss_list)
            ax.set_title('Neg. ELBO Loss', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
    fig.tight_layout()

    os.makedirs('./results/power_plots', exist_ok=True)
    plt.savefig(f'./results/power_plots/ts_DeepQEP_q{power}_seed{seed}.png', bbox_inches='tight')
    plt.close(fig)

    return {
        'seed': seed,
        'power': power,
        'metrics': {
            'MAE': MAE,
            'RMSE': RMSE,
            'PSD': PSD,
            'R2': R2,
            'NLL': NLL,
            'time': time_
        }
    }


def grid_search_powers(powers=[0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]):
    """Run grid search over different power values"""
    os.makedirs('./results/grid_search', exist_ok=True)

    all_results = []
    power_summaries = []

    # For each power value, run multiple seeds
    for power in powers:
        print(f"\n=== Running trials for power={power} ===")
        power_results = []

        # Run with initial seed and then additional seeds
        try:
            first_result = main(seed=1, power=power)
            power_results.append(first_result)
            all_results.append(first_result)
        except Exception as e:
            print(f"Seed 1 failed with error: {e}")

        # Run additional seeds like in original code
        n_seed = 10
        i = 0
        n_success = 1  #
        n_failure = 0

        while n_success < n_seed and n_failure < 10 * n_seed:
            seed_i = 2024 + i * 10
            try:
                print(f"Running for seed {seed_i} with power={power}...")
                result = main(seed=seed_i, power=power)
                power_results.append(result)
                all_results.append(result)
                n_success += 1
            except Exception as e:
                print(f"Seed {seed_i} failed with error: {e}")
                n_failure += 1
            i += 1

        # Calculate summary statistics for this power value
        if power_results:
            mae_values = [r['metrics']['MAE'] for r in power_results]
            rmse_values = [r['metrics']['RMSE'] for r in power_results]
            psd_values = [r['metrics']['PSD'] for r in power_results]
            r2_values = [r['metrics']['R2'] for r in power_results]
            nll_values = [r['metrics']['NLL'] for r in power_results]

            power_summary = {
                'power': power,
                'mae_mean': np.mean(mae_values),
                'mae_std': np.std(mae_values, ddof=1),
                'rmse_mean': np.mean(rmse_values),
                'rmse_std': np.std(rmse_values, ddof=1),
                'psd_mean': np.mean(psd_values),
                'psd_std': np.std(psd_values, ddof=1),
                'r2_mean': np.mean(r2_values),
                'r2_std': np.std(r2_values, ddof=1),
                'nll_mean': np.mean(nll_values),
                'nll_std': np.std(nll_values, ddof=1),
                'n_runs': len(power_results)
            }

            power_summaries.append(power_summary)

            print(f"\nSummary for power={power} ({power_summary['n_runs']} successful runs):")
            print(f"MAE: {power_summary['mae_mean']:.4f} ± {power_summary['mae_std']:.4f}")
            print(f"RMSE: {power_summary['rmse_mean']:.4f} ± {power_summary['rmse_std']:.4f}")
            print(f"PSD: {power_summary['psd_mean']:.4f} ± {power_summary['psd_std']:.4f}")
            print(f"R2: {power_summary['r2_mean']:.4f} ± {power_summary['r2_std']:.4f}")
            print(f"NLL: {power_summary['nll_mean']:.4f} ± {power_summary['nll_std']:.4f}")

    # Find best power value based on mean MAE
    best_power_summary = min(power_summaries, key=lambda x: x['mae_mean'])

    print("\n=== GRID SEARCH RESULTS ===")
    print(f"Best power value: {best_power_summary['power']}")
    print(f"MAE: {best_power_summary['mae_mean']:.4f} ± {best_power_summary['mae_std']:.4f}")
    print(f"RMSE: {best_power_summary['rmse_mean']:.4f} ± {best_power_summary['rmse_std']:.4f}")
    print(f"PSD: {best_power_summary['psd_mean']:.4f} ± {best_power_summary['psd_std']:.4f}")
    print(f"R2: {best_power_summary['r2_mean']:.4f} ± {best_power_summary['r2_std']:.4f}")
    print(f"NLL: {best_power_summary['nll_mean']:.4f} ± {best_power_summary['nll_std']:.4f}")

    # Create plot comparing different power values
    plt.figure(figsize=(10, 6))
    powers = [ps['power'] for ps in power_summaries]
    mae_means = [ps['mae_mean'] for ps in power_summaries]
    mae_stds = [ps['mae_std'] for ps in power_summaries]

    plt.errorbar(powers, mae_means, yerr=mae_stds, fmt='o-', capsize=5, linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Power value (q)', fontsize=14)
    plt.ylabel('Average MAE', fontsize=14)
    plt.title('MAE vs Power Value', fontsize=16)

    min_idx = np.argmin(mae_means)
    plt.annotate(f'Min MAE: {mae_means[min_idx]:.4f} at q={powers[min_idx]}',
                 xy=(powers[min_idx], mae_means[min_idx]),
                 xytext=(powers[min_idx], mae_means[min_idx] + 0.03),
                 fontsize=12,
                 ha='center',
                 arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    plt.savefig('./results/grid_search/mae_vs_power_summary.png')
    plt.close()

    # Create summary table
    summary_data = []
    header = ['Power', 'MAE', 'MAE_std', 'RMSE', 'RMSE_std', 'PSD', 'PSD_std', 'R2', 'R2_std', 'NLL', 'NLL_std', 'N_runs']

    for ps in power_summaries:
        row = [
            ps['power'],
            f"{ps['mae_mean']:.4f}",
            f"{ps['mae_std']:.4f}",
            f"{ps['rmse_mean']:.4f}",
            f"{ps['rmse_std']:.4f}",
            f"{ps['psd_mean']:.4f}",
            f"{ps['psd_std']:.4f}",
            f"{ps['r2_mean']:.4f}",
            f"{ps['r2_std']:.4f}",
            f"{ps['nll_mean']:.4f}",
            f"{ps['nll_std']:.4f}",
            ps['n_runs']
        ]
        summary_data.append(row)

    # Save summary to file
    summary_array = np.array(summary_data)
    np.savetxt('./results/grid_search/power_summary.csv', summary_array, fmt='%s', delimiter=',', header=','.join(header))

    return best_power_summary, power_summaries


if __name__ == '__main__':
    # Define power values to search over
    power_values = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]

    # Run grid search
    best_power, all_summaries = grid_search_powers(powers=power_values)

    print("\nGrid search complete!")
    print(f"Best power value: {best_power['power']}")