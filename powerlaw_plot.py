import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import scipy.stats as stats
import pandas as pd
# set font to helvetica
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['helvetica']
plt.rcParams.update({'font.size': 12})

colors = ["#D55E00", "#FF9123", "#61C2FF", "#07AB92", "#014C4C", "#FF7DBE"]
main_color = '#DDDDDD'
second_color = colors[5]

cm = 1 / 2.54  # centimeters in inches



def generate_power_law_cdf(exponent, size=100000, xmin=1):
    U = np.random.uniform(size=size)
    return xmin * (1 - U) ** (-1 / (exponent - 1))
    
def generate_power_law_discrete(exponent, size=10000, xmin=1,xmax=1000):
    omega = np.linspace(xmin, xmax, num=xmax)
    weights = omega ** (-exponent)
    data = np.random.choice(omega, size=size, p=weights/weights.sum())
    return data
    
   
def generate_power_law_data(exponent, size=100000, xmin=1):
    """
    Generate data from a power-law distribution using the scipy powerlaw package.
    """
    # Use powerlaw to generate synthetic data
    # define powerlaw distribution
    theoretical_distribution = powerlaw.Power_Law(
        xmin=xmin, parameters=[exponent])  # define the power-law distribution
    # take 1000 random variates
    # np.random.seed(0)
    simulated_data = theoretical_distribution.generate_random(size)
    return simulated_data

def estimate_exponent_mle(data, xmin):
    """
    Estimate the exponent using the maximum likelihood estimation method (MLCSN).
    """
    fit = powerlaw.Fit(data,discrete=True)
    print(fit.alpha, fit.xmin)
    return fit.alpha

def estimate_exponent_ls(data, xmin):
    """
    Estimate the exponent using the Least Squares method.
    """
    counts, bin_edges = np.histogram(data, bins=np.logspace(np.log10(data.min()), np.log10(data.max()), num=10), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    log_counts = np.log(counts[counts > 0])
    log_bin_centers = np.log(bin_centers[:len(log_counts)])
    
    #use scipy optimize to fit a line to the log-log plot
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_bin_centers, log_counts)
    
    return -slope

def estimate_ml_star_estimator(data, xmin=1, xmax=None, tol=1e-5, max_iter=1000):
    """
    Estimate the power-law exponent using the ML* method for discrete data.

    Parameters:
    - data: array-like, discrete data following a power-law distribution
    - xmin: float, the minimum value to be considered in the data
    - xmax: float, the maximum value to be considered in the data (optional)
    - tol: float, tolerance level for convergence
    - max_iter: int, maximum number of iterations for the estimator

    Returns:
    - lambda_star: float, estimated power-law exponent
    """
    # Filter data based on the given xmin and xmax
    if xmax is None:
        xmax = max(data)
    data = data[(data >= xmin) & (data <= xmax)]
    # Get unique values and their frequencies
    unique, counts = np.unique(data, return_counts=True)
    frequencies = counts / np.sum(counts)
    # Initialize lambda
    lambda_star = 2.0  # Initial guess for the exponent
    delta_lambda = 1.0  # Initial step size for iteration
    for _ in range(max_iter):
        # Calculate left-hand and right-hand sides of the equation
        lhs = np.sum(frequencies * np.log(unique))  # Left-hand side of the equation
        rhs = (np.sum(unique ** -lambda_star)) ** -1 * np.sum((unique ** -lambda_star) * np.log(unique))  # Right-hand side
        # Update lambda_star based on the difference between lhs and rhs
        new_lambda_star = lambda_star - (lhs - rhs)
        # Check convergence
        if abs(new_lambda_star - lambda_star) < tol:
            break
        lambda_star = new_lambda_star
    return lambda_star
# Define main colors
colors = {
    'ls_main': "#61C2FF",  # Main color for LS estimator
    'ml_main': "#FF7DBE",  # Main color for MLCSN estimator
    'ml_star_main': "#07AB92",  # Main color for ML* estimator
    'true_line': "#000000"  # Color for the true exponent line
}

def estimate_power_law(n_data_points=100000, xmin=10, n_trials=40, exponents=np.linspace(0.1, 4, 200)):
    """
    Perform estimation for LS, MLCSN, and ML* estimators on power-law data.

    Parameters:
    - n_data_points: Number of data points to generate for each exponent
    - xmin: Minimum value for fitting
    - n_trials: Number of trials for averaging estimations
    - exponents: List or array of exponents to test

    Returns:
    - dict with true exponents, estimates, and percentiles for LS, MLCSN, and ML*
    """
    ls_estimates = []
    ml_estimates = []
    ml_star_estimates = []
    ls_percentiles = []
    ml_percentiles = []
    ml_star_percentiles = []
    
    for exponent in exponents:
        ls_estimates_tmp = np.zeros(n_trials)
        ml_estimates_tmp = np.zeros(n_trials)
        ml_star_estimates_tmp = np.zeros(n_trials)
        
        for i in range(n_trials):
            data = generate_power_law_discrete(exponent, size=n_data_points, xmin=xmin, xmax=1000)
            
            # LS estimator
            ls_exponent = estimate_exponent_ls(data, xmin)
            ls_estimates_tmp[i] = ls_exponent
            
            # MLCSN estimator
            ml_exponent = estimate_exponent_mle(data, xmin)
            ml_estimates_tmp[i] = ml_exponent
            
            # ML* estimator
            ml_star = estimate_ml_star_estimator(data, xmin)
            ml_star_estimates_tmp[i] = ml_star
        
        # Save mean estimates
        ls_estimates.append(np.mean(ls_estimates_tmp))
        ml_estimates.append(np.mean(ml_estimates_tmp))
        ml_star_estimates.append(np.mean(ml_star_estimates_tmp))
        
        # Save 5th and 95th percentiles
        ls_percentiles.append(np.percentile(ls_estimates_tmp, [10, 90]))
        ml_percentiles.append(np.percentile(ml_estimates_tmp, [10, 90]))
        ml_star_percentiles.append(np.percentile(ml_star_estimates_tmp, [10, 90]))

    # Convert percentiles to numpy arrays
    ls_percentiles = np.array(ls_percentiles)
    ml_percentiles = np.array(ml_percentiles)
    ml_star_percentiles = np.array(ml_star_percentiles)
    
    return {
        'true_exponent': exponents,
        'ls_estimate': ls_estimates,
        'ls_percentiles': ls_percentiles,
        'ml_estimate': ml_estimates,
        'ml_percentiles': ml_percentiles,
        'ml_star_estimate': ml_star_estimates,
        'ml_star_percentiles': ml_star_percentiles
    }


def save_estimates_to_csv(estimates, filename='power_law_estimates.csv'):
    """
    Save the estimates and percentiles to a CSV file.

    Parameters:
    - estimates: Dictionary containing the estimates and percentiles
    - filename: Filename for saving the CSV
    """
    data_dict = {
        'true_exponent': estimates['true_exponent'],
        'ls_estimate': estimates['ls_estimate'],
        'ls_5th_percentile': estimates['ls_percentiles'][:, 0],
        'ls_95th_percentile': estimates['ls_percentiles'][:, 1],
        'ml_estimate': estimates['ml_estimate'],
        'ml_5th_percentile': estimates['ml_percentiles'][:, 0],
        'ml_95th_percentile': estimates['ml_percentiles'][:, 1],
        'ml_star_estimate': estimates['ml_star_estimate'],
        'ml_star_5th_percentile': estimates['ml_star_percentiles'][:, 0],
        'ml_star_95th_percentile': estimates['ml_star_percentiles'][:, 1]
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(filename, index=False)
    print(f"Estimates saved to '{filename}'")


def plot_estimates(estimates, save_figure=True, filename='figures/estimators_power_law.png'):
    """
    Plot the estimated exponents and their 5th and 95th percentiles.

    Parameters:
    - estimates: Dictionary containing the estimates and percentiles
    - save_figure: Whether to save the figure
    - filename: Filename for saving the figure
    """
    exponents = estimates['true_exponent']
    ls_estimates = estimates['ls_estimate']
    ml_estimates = estimates['ml_estimate']
    ml_star_estimates = estimates['ml_star_estimate']
    ls_5th_percentile = estimates['ls_5th_percentile']
    ls_95th_percentile = estimates['ls_95th_percentile']
    ml_5th_percentile = estimates['ml_5th_percentile']
    ml_95th_percentile = estimates['ml_95th_percentile']
    ml_star_5th_percentile = estimates['ml_star_5th_percentile']
    ml_star_95th_percentile = estimates['ml_star_95th_percentile']
    # Plot results
    fig, ax = plt.subplots(figsize=(15 * 0.3937, 9 * 0.3937))  # Convert cm to inches for figure size
    
    # Plot the true exponents
    ax.plot(exponents, exponents, color=colors['true_line'], label="true exponent", linestyle='-')
    
    # Plot LS estimator with percentiles
    ax.fill_between(exponents, ls_5th_percentile, ls_95th_percentile, color=colors['ls_main'], alpha=0.2)
    ax.plot(exponents, ls_estimates, color=colors['ls_main'], label="ls")
    
    # Plot MLCSN estimator with percentiles
    ax.fill_between(exponents, ml_5th_percentile, ml_95th_percentile, color=colors['ml_main'], alpha=0.2)
    ax.plot(exponents, ml_estimates, color=colors['ml_main'], label="ml")
    
    # Plot ML* estimator with percentiles
    ax.fill_between(exponents, ml_star_5th_percentile, ml_star_95th_percentile, color=colors['ml_star_main'], alpha=0.3)
    ax.plot(exponents, ml_star_estimates, color=colors['ml_star_main'], label="ml bounded")

    # Labels and legend
    ax.set_xlabel("true exponent")
    ax.set_ylabel("estimated exponent")
    #no grid for legend, but white box around it to make it stand, square border, line width 1
    legend = ax.legend(frameon=False, loc='upper right', facecolor='white', framealpha=1, edgecolor='black', fontsize=8.5)
    legend.get_frame().set_linewidth(0.5)
    # Inset: zoom in on the range 2-3
    ax_inset = fig.add_axes([0.39, 0.62, 0.25, 0.25])  # Position and size of the inset #0.658 #0.38
    ax_inset.set_xlim([2.5, 2.7])
    ax_inset.set_ylim([2.5, 2.7])
    #reduce ticklabels font size
    ax_inset.tick_params(axis='both', which='major', labelsize=8)
    ax_inset.tick_params(axis='both', which='minor', labelsize=8)
    
    
    # Plot the zoomed-in section in the inset
    ax_inset.plot(exponents, exponents, color=colors['true_line'])
    ax_inset.plot(exponents, ls_estimates, color=colors['ls_main'])
    ax_inset.plot(exponents, ml_estimates, color=colors['ml_main'])
    ax_inset.plot(exponents, ml_star_estimates, color=colors['ml_star_main'])
    #add grid for insert
    #ax_inset.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4,colors='white')
    #remove top and right spines
    ax_inset.spines['top'].set_visible(False)
    ax_inset.spines['right'].set_visible(False)
    #alpha the inset
    ax_inset.patch.set_alpha(0.8)
    ax_inset.axvspan(2.5, 2.7, color='gray', alpha=0.2, lw=0)
    
    ax.set_xlim([exponents.values[0], exponents.values[-1]])
    ax.set_ylim([exponents.values[0], 5.8])
    
    #only integer ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    
    #add color on ax for regime of exponents, 2-3
    ax.axvspan(2, 3, color='gray', alpha=0.2, lw=0)
    #line at 1
    ax.axvline(x=1, color='gray', linestyle='--', lw=1)
    #equal aspect ratio
    #ax.set_aspect('equal')
    #remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Save the figure
    if save_figure:
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Figure saved as '{filename}'")

    #plt.show()


def power_law_distribution(x, alpha, xmin):
    """
    Compute the power-law distribution.
    """
    return (alpha - 1) * (xmin ** (alpha - 1)) * (x ** (-alpha))

#function to plot a power law distribution and fit
def plot_fit(data):
    """
    Plot the data and the fitted power-law distribution.
    """
    # Fit the data
    fit = powerlaw.Fit(data, discrete=True)
    bins = np.logspace(np.log10(data.min()), np.log10(data.max()), num=50)
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig,ax = plt.subplots(figsize=(18.6 *cm, 10*cm))  # Figure size in cm, converted to inches
    plt.plot(bin_centers, counts, '.', color=main_color, label="data")
    plt.plot(bin_centers, power_law_distribution(bin_centers, fit.alpha, fit.xmin), color=second_color, label="power-law fit")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.title(f'fit {fit.alpha:.2f}')
    plt.tight_layout()
    plt.savefig('figures/power_law_fit.png')
    
# data = generate_power_law_data(2.5, size=int(2e4), xmin=100)
data = generate_power_law_discrete(4, size=int(2e4), xmin=1,xmax=10000)
plot_fit(data)

#load the estimate fro msave_estimates_to_csv
def load_estimates():
    df = pd.read_csv('power_law_estimates.csv')
    return df
    
    
if __name__ == "__main__":
    # estimates = estimate_power_law()
    # save_estimates_to_csv(estimates)
    estimates = load_estimates()
    plot_estimates(estimates)