"""
Windowing and clustering on shear-wave splitting measurements, 
following the exact methodology of Teanby et al., 2004.
Many comments adapted or quoted from Teanby et al., 2004. to explain the methodology.
By: Michael Hemmett, assisted by Claude Sonnet 4.5
"""

# Import relevant modules
import numpy as np
import splitting_functions as sf
#from splitting_functions import perform_splitting_analysis_baillard


# First, we set up the windows for clustering. 
def setup_windows(T_beg_1, T_end_0, dT_beg, dT_end, N_beg, N_end):
    """
    Set up the time windows for clustering shear-wave splitting measurements.

    Parameters:
    T_beg_1 (float): Start time of the first window.
    T_end_0 (float): End time of the last window.
    dT_beg (float): Duration of the beginning windows.
    dT_end (float): Duration of the ending windows.
    N_beg (int): Number of beginning windows.
    N_end (int): Number of ending windows.

    Returns:
    list: List of tuples representing the window number, start and end times of each window.
    """
    windows = []
    
    # Create beginning windows
    for i in range(N_beg):
        T_beg = T_beg_1 - (i - 1) * dT_beg
    
        # Create ending windows
        for j in range(N_end):
            T_end = T_end_0 + (j - 1) * dT_end

            window_id = (j - 1) * N_beg + i
            windows.append((window_id, T_beg, T_end))
            

    return windows

# Once windows are defined, the splitting analysis is performed for each window
# We will call the relevant functions from splitting_functions.py to do this.

def split_all_windows(event_data, 
                      windows,
                      N_beg=5, N_end=5,
                      min_lag=0, max_lag=60,
                      Nlags=60, Nangles=90,
                      flag_adapt_window=True,
                      flag_adapt_maxlag=True,
                      min_thres=0.5,
                      min_numbers=2,
                      plot_results=False,
                      output_dir=None):
    """
    Perform shear-wave splitting analysis for all defined windows, using the Baillard method in splitting_functions.py.
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms
    windows : list of tuples
        List of windows as returned by setup_windows()
    min_lag : int
        Minimum delay time in samples
    max_lag : int
        Maximum delay time in samples
    Nlags : int
        Number of lag values to test
    Nangles : int
        Number of angle values to test
    flag_adapt_window : bool
        Whether to adapt window based on dominant period
    flag_adapt_maxlag : bool
        Whether to adapt max_lag based on dominant period
    min_thres : float
        Threshold for minimum selection (0-1)
    min_numbers : int
        Maximum number of minima to keep
    plot_results : bool
        Whether to generate diagnostic plots (default: False)
    output_dir : str or None
        Directory to save plots (if None and plot_results=True, displays interactively)

    Returns:
    --------
    results : list of tuples
        List of results for each window, each as (window_id, result_dict)
    """

    # Define empty list to hold results
    results = []
    # Iterate over all windows and perform splitting analysis
    for window in windows:
        # Define start and end times for current window
        s_window = [window[1], window[2]]

        # Perform Baillard-like splitting analysis for current window
        single_window_result = sf.perform_splitting_analysis_baillard(event_data,
                                    s_window,
                                    min_lag=min_lag,
                                    max_lag=max_lag,
                                    Nlags=Nlags,
                                    Nangles=Nangles,
                                    flag_adapt_window=flag_adapt_window,
                                    flag_adapt_maxlag=flag_adapt_maxlag,
                                    min_thres=min_thres,
                                    min_numbers=min_numbers,
                                    plot_results=plot_results,
                                    output_dir=output_dir)

        # Add window ID and splitting results to results list
        results.append((window[0], single_window_result))

        # Output progress
        print()
        print(f"Completed splitting analysis for window {window[0]+1}: {s_window[0]:.2f} to {s_window[1]:.2f} s")
        print(f"Window {window[0]+1} / {N_beg * N_end} complete.")
        print()

    return results

# Now that we have the results for all windows, we can cluster them.
# Teanby uses bottom-up hierarchical clustering
# We start with each measurement as its own cluster, then merge the closest clusters iteratively.
# After each merge, we recalculate the cluster centers until there is only one cluster left, comprising the whole dataset.
# For each number of clusters M = 1... N, we calculate the number of data points N_j in each cluster C_j
# And the positions of the cluster centers (dt_j, phi_j) given by the mean positiion of points with the cluster.

# First, define helper functions for the clustering algorithm
# Position of the cluster centers for each cluster 'j' in (dt, phi) space:

def calc_dt_j(dts, N_j):
    """
    Calculate the delay time center of a cluster.
    Parameters:
    -----------
    dts: A list of delay times in the cluster
    N_j: Number of data points in the cluster

    Returns:
    --------
    dt_j: The delay time center of the cluster
    """
    dt_j = np.sum(dts) / N_j
    return dt_j

def calc_phi_j(phis, N_j):
    """
    Calculate the fast axis orientation center of a cluster.
    Parameters:
    -----------
    phis: A list of fast axis orientations in the cluster
    N_j: Number of data points in the cluster

    Returns:
    --------
    phi_j: The fast axis orientation center of the cluster
    """

    phi_j = np.sum(phis) / N_j
    return phi_j

def overall_dt_center(all_dts, N):
    """
    Calculate the overall delay time center of all data points.
    Parameters:
    -----------
    all_dts: A list of all delay times
    N: Total number of data points

    Returns:
    --------
    overall_dt: The overall delay time center
    """
    overall_dt = np.sum(all_dts) / N
    return overall_dt

def overall_phi_center(all_phis, N):
    """
    Calculate the overall fast axis orientation center of all data points.
    Parameters:
    -----------
    all_phis: A list of all fast axis orientations
    N: Total number of data points

    Returns:
    --------
    overall_phi: The overall fast axis orientation center
    """
    overall_phi = np.sum(all_phis) / N
    return overall_phi

# The final clustering requires we define the number of clusters 'M' we want to end up with.
# We wish to do this in an unsupervised way, and following Teanby we will use the methods of 
# Calinski-Harabasz and Duda-Hart to determine the optimal number of clusters.
# Clustering is stopped wehn these criteria pass specific thresholds

# First, we define covariance parameters 

def within_cluster_covariance(clusters, cluster_centers):
    """
    Calculate the within-cluster covariance for a set of clusters.
    Parameters:
    -----------
    clusters: A list of clusters, each containing data points
    cluster_centers: A list of cluster centers corresponding to each cluster

    The output will be a matrix W as a double sum of squared distances between each point and its cluster center.
    # We will sum over the number of clusters 'M' and the number of points in each cluster 'N_j'.
    Position 1,1 corresponds to delay time covariance, position 2,2 to fast axis orientation covariance.
    Position 1,2 and 2,1 are cross-terms.

    Returns:
    --------
    W: The within-cluster covariance
    """
    W = np.matrix([[0., 0.], [0., 0.]], dtype=float)
    M = len(clusters)
    for j in range(M):
        cluster = clusters[j]
        center = cluster_centers[j]
        for point in cluster:
            dt_diff = point[0] - center[0]
            phi_diff = point[1] - center[1]
            W[0, 0] += (dt_diff ** 2)
            W[1, 1] += (phi_diff ** 2)
            W[0, 1] += (dt_diff * phi_diff)
            W[1, 0] += (dt_diff * phi_diff)

    return W

def between_cluster_variance(cluster_centers, overall_center):
    """
    Calculate the between-cluster variance for a set of cluster centers.
    Parameters:
    -----------
    cluster_centers: A list of cluster centers
    overall_center: The overall center of all data points

    The output will be a matrix B as a double sum of squared distances between each cluster center and the overall center.
    # We will sum over the number of clusters 'M'.
    Position 1,1 corresponds to delay time variance, position 2,2 to fast axis orientation variance.
    Position 1,2 and 2,1 are cross-terms.

    Returns:
    --------
    B: The between-cluster variance
    """
    B = np.matrix([[0, 0], [0, 0]], dtype=float)
    M = len(cluster_centers)
    for center in cluster_centers:
        dt_diff = center[0] - overall_center[0]
        phi_diff = center[1] - overall_center[1]
        B[0, 0] += (dt_diff ** 2)
        B[1, 1] += (phi_diff ** 2)
        B[0, 1] += (dt_diff * phi_diff)
        B[1, 0] += (dt_diff * phi_diff)

    return B

# We will now define our unsupervised splitting stopping criteria functions

def calinski_harabasz_criteria(B, W, N, M):
    """
    Calculate the Calinski-Harabasz criterion for clustering.
    Parameters:
    -----------
    B: Between-cluster variance matrix
    W: Within-cluster covariance matrix
    N: Total number of data points
    M: Number of clusters

    Returns:
    --------
    c_M: The Calinski-Harabasz criterion value
    """
    trace_B = np.trace(B)
    trace_W = np.trace(W)

    if trace_W == 0:
        trace_W = 1  # Prevent division by zero
    c_M = ((N - M) * trace_B) / ((M - 1) * trace_W)

    return c_M

# Duda-Hart criterion is based on the ratio of within-cluster variances when two clusters are combined into one cluster
# First we will define helper functions to calculate the Duda-Hart criterion

def two_cluster_variance(clusters, cluster_centers):
    """
    Calculate the variance for two clusters combined into one.
    Parameters:
    -----------
    clusters: A list of two clusters, each containing data points
    cluster_centers: A list of two cluster centers corresponding to each cluster

    Returns:
    --------
    sigma_squared: The variance of the combined clusters
    """
    
    # We will sum over both clusters and then sum over all N_j points in each cluster
    # We will calcualte the squared distance between each point in the cluster and the cluster center, then sum results for both clusters
    sigma_squared = 0
    for j in range(2):
        cluster = clusters[j]
        center = cluster_centers[j]
        N_j = len(cluster)
        for point in cluster:
            dt_diff = point[0] - center[0]
            phi_diff = point[1] - center[1]
            sigma_squared += ((dt_diff ** 2) + (phi_diff ** 2)) / N_j

    return sigma_squared

def one_cluster_variance(clusters, overall_center):
    """
    Calculate the variance for one cluster formed by combining two clusters.
    Parameters:
    -----------
    clusters: A list of two clusters, each containing data points
    overall_center: The overall center of all data points

    Returns:
    --------
    sigma_squared: The variance of the combined cluster
    """

    # We will sum over both clusters and then sum over all N_j points in each cluster
    # We will calcualte the squared distance between each point in the cluster and the overall center, then sum results for both clusters
    sigma_squared = 0
    for cluster in clusters:
        N_j = len(cluster)
        for point in cluster:
            dt_diff = point[0] - overall_center[0]
            phi_diff = point[1] - overall_center[1]
            sigma_squared += ((dt_diff ** 2) + (phi_diff ** 2)) / N_j

    return sigma_squared

# The null hypothesis is that the two clusters should be combined into one cluster
# Normally distributed wihtin-cluster distances are assumed, and the null hypothesis
# is rejected when:

def single_duda_hart_criteria(sigma_two, sigma_one, N_j, c_critical=3.20):
    """
    Calculate the Duda-Hart criterion for clustering for each pair of clusters.
    Parameters:
    -----------
    sigma_two: Variance of two clusters combined into one
    sigma_one: Variance of one cluster formed by combining two clusters
    N_j: Total number of data points in the two clusters
    c_critical: Critical value for Duda-Hart criterion (default: 3.20, from Milligan and Cooper, 1985)

    Returns:
    --------
    pass_dh: Boolean indicating whether the Duda-Hart criterion is passed
    If true, the null hypothesis is rejected and the two clusters should remain separate.
    """
    # Def number of parameters 'p' as 2 (dt and phi)
    p = 2

    ratio_dh = (1 - (sigma_two / sigma_one) - (2 / (np.pi * p)))
    N_j_term = ((N_j * p) / (2 * (1 - (8 / ((np.pi ** 2) * p)))))

    dh_value = ratio_dh * (np.sqrt(N_j_term))

    if dh_value > c_critical:
        pass_dh = False
    else:
        pass_dh = True

    return pass_dh

# We will consider the hierarchy of clusters from M = 1... N and 
# halt the subdivision of clusters when dh = True. 

def duda_hart_criteria(clusters, cluster_centers, overall_center):
    """
    Apply the Duda-Hart criterion to a set of clusters.
    Iteratively merge cluster pairs that pass the criterion until no more merges are possible.
    
    Parameters:
    -----------
    clusters: A list of clusters, each containing data points
    cluster_centers: A list of cluster centers corresponding to each cluster
    overall_center: The overall center of all data points

    Returns:
    --------
    M_dh: The final number of clusters after applying Duda-Hart criterion
    """
    # Keep merging until no pair passes the Duda-Hart criterion
    while len(clusters) > 1:
        merge_found = False
        
        # Check all pairs of clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_pair = [clusters[i], clusters[j]]
                center_pair = [cluster_centers[i], cluster_centers[j]]
                N_j = len(clusters[i]) + len(clusters[j])

                sigma_two = two_cluster_variance(cluster_pair, center_pair)
                sigma_one = one_cluster_variance(cluster_pair, overall_center)

                pass_dh = single_duda_hart_criteria(sigma_two, sigma_one, N_j)

                if pass_dh:
                    # Merge these two clusters
                    merged_cluster = clusters[i] + clusters[j]
                    merged_center = calculate_center_from_cluster(merged_cluster)
                    
                    # Update clusters and centers
                    new_clusters = []
                    new_centers = []
                    for idx in range(len(clusters)):
                        if idx != i and idx != j:
                            new_clusters.append(clusters[idx])
                            new_centers.append(cluster_centers[idx])
                    new_clusters.append(merged_cluster)
                    new_centers.append(merged_center)
                    
                    clusters = new_clusters
                    cluster_centers = new_centers
                    merge_found = True
                    break  # Restart checking from the beginning with new cluster configuration
            
            if merge_found:
                break
        
        # If no merge was found, stop iterating
        if not merge_found:
            break

    M_dh = len(clusters)
    return M_dh

# We will define the optimum number of clusters 'M' as
# the maximum value of M predicted by both criteria, whichever is greater.

# Define function for setting maximum value of M based on both criteria

def optimum_num_clusters(clusters, cluster_centers, overall_center, N):
    """
    Determine the optimum number of clusters based on Calinski-Harabasz and Duda-Hart criteria.
    Parameters:
    -----------
    clusters: A list of clusters, each containing data points
    cluster_centers: A list of cluster centers corresponding to each cluster
    overall_center: The overall center of all data points
    N: Total number of data points

    Returns:
    --------
    M_opt: The optimum number of clusters
    """
    print()
    print("Calculating optimum number of clusters...")

    M = len(clusters)
    print(f"Total number of clusters: M = {M}")

    # Calculate within-cluster covariance and between-cluster variance
    W = within_cluster_covariance(clusters, cluster_centers)
    print(f"Within-cluster covariance W:\n{W}")

    B = between_cluster_variance(cluster_centers, overall_center)
    print(f"Between-cluster variance B:\n{B}")

    # Calculate Calinski-Harabasz criterion
    c_M = calinski_harabasz_criteria(B, W, N, M)
    print(f"Calinski-Harabasz criterion: {c_M}")

    # Calculate Duda-Hart criterion
    M_dh = duda_hart_criteria(clusters, cluster_centers, overall_center)
    print(f"Duda-Hart criterion: {M_dh}")

    # Determine optimum number of clusters
    M_opt = max(c_M, M_dh)

    if M_opt < 1:
        M_opt = 1

    print(f"Optimum number of clusters determined: M_opt = {M_opt}")
    print()

    return M_opt

# Once the clsuter centers and optimum number of clusters are determined,
# we must seelct the best cluster adn the best measurement from within this cluster.
# Criteria for the best cluster are based on teh number of points and the variance within the cluster.
# All clusters with less than N_c_min data points are considered spurious and rejected.
# If this leaves no clusters, then there is no stable solution for this event.
# N_c_min is chosen such that it corresponds to approximately a cycle's worth of points
# The within-cluster variance sigma_c_j^2 and mean data variance, sigma_d_j^2 of the remaining clusters are then calculated.
# We will define the within-clsuter variance and mean data variance functions below.

def within_cluster_variance_single_cluster(cluster, cluster_center):
    """
    Calculate the within-cluster variance for a single cluster.
    Parameters:
    -----------
    cluster: A list of data points in the cluster
    cluster_center: The center of the cluster

    Returns:
    --------
    sigma_c_j_squared: The within-cluster variance
    """
    sigma_c_j_squared = 0
    N_j = len(cluster)
    for point in cluster:
        dt_diff = point[0] - cluster_center[0]
        phi_diff = point[1] - cluster_center[1]
        sigma_c_j_squared += (dt_diff ** 2) + (phi_diff ** 2)

    sigma_c_j_squared /= N_j

    return sigma_c_j_squared

# Before calculating mean data variance, define variance helper function that calculates variance between data and center for a single parameter in a single cluster

def calc_variance(cluster, cluster_center):
    """
    Calculate the variance for single parameter in a single cluster.
    Parameters:
    -----------
    cluster: A list of data points in the cluster
    cluster_center: The center of the cluster

    Returns:
    --------
    variance: The variance for the parameter
    """
    sigma_squared = 0

    # Add check for single data point clusters
    if type(cluster) == np.float64:
        cluster = [cluster]

    N_j = len(cluster)
    cluster_center = cluster_center
    for point in cluster:
        diff = point - cluster_center
        sigma_squared += (diff ** 2)

    variance = sigma_squared / N_j

    return variance


# This is related to the harmonic mean, which reduces the effects of outliers
def mean_data_variance(cluster):
    """
    Calculate the mean data variance for a single cluster.
    We find the inverse of the sum over all N_j of (1 / sigma_dt_i_j^2) 
    + the inverse of the sum over all N_j of (1 / sigma_phi_i_j^2)

    Params:
    -----------
    cluster: A list of data points in the cluster

    Returns:
    --------
    sigma_d_j_squared: The mean data variance
    """
    N_j = len(cluster)
    sum_inv_dt_variance = 0
    sum_inv_phi_variance = 0

    for point in cluster:
        dt_variance = calc_variance(point[0], cluster[0][0])
        phi_variance = calc_variance(point[1], cluster[0][1])
        sum_inv_dt_variance += 1 / dt_variance
        sum_inv_phi_variance += 1 / phi_variance

    sigma_d_j_squared = (1 / sum_inv_dt_variance) + (1 / sum_inv_phi_variance)

    return sigma_d_j_squared

# Now define the overall variance for the cluster as the max of within-cluster variance and mean data variance
def overall_cluster_variance(cluster, cluster_center):
    """
    Calculate the overall variance for a single cluster.
    This is defined as the maximum of the within-cluster variance and mean data variance.

    Params:
    -----------
    cluster: A list of data points in the cluster
    cluster_center: The center of the cluster

    Returns:
    --------
    sigma_j_squared: The overall variance for the cluster
    """
    sigma_c_j_squared = within_cluster_variance_single_cluster(cluster, cluster_center)
    sigma_d_j_squared = mean_data_variance(cluster)

    sigma_j_squared = max(sigma_c_j_squared, sigma_d_j_squared)

    return sigma_j_squared

# Add a helper function to calculate cluster center or overall center from a single cluster:
def calculate_center_from_cluster(cluster):
    """
    Calculate the center (dt, phi) from a single cluster.
    Parameters:
    -----------
    cluster: A list of data points in the cluster

    Returns:
    --------
    center: A tuple representing the center (dt, phi) of the cluster
    """
    N_j = len(cluster)
    dt_j = calc_dt_j([point[0] for point in cluster], N_j)
    phi_j = calc_phi_j([point[1] for point in cluster], N_j)

    center = (dt_j, phi_j)
    return center


# We will plot all measurements in (dt, phi) space, as well as show the best cluster and best measurement.
def plot_measurements(measurements, best_cluster, best_measurement):
    """
    Plot all clusters, highlighting the best cluster and best measurement.
    Parameters:
    -----------
    measurements: A list of tuples representing shear-wave splitting measurements as (delay_time, fast_axis_orientation)
    best_cluster: The best cluster selected based on minimum overall variance
    best_measurement: The best measurement from the best cluster
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))


    # Plot all measurements in (dt, phi) space    clusters = []
    for measurement in measurements:
        plt.scatter(measurement[0], measurement[1], color='black', alpha=1.0)

    # Highlight best cluster and plot it as a plus sign with error bars
    if best_cluster is not None:
        
        # Calculate cluster center and variance for error bars
        best_center = calculate_center_from_cluster(best_cluster)
        
        # Plot as plus signs with error bars
        plt.scatter(best_center[0], best_center[1], 
                    marker='+', s=150, edgecolor='blue',
                    label='Best Cluster')
    
    # Highlight best measurement
    if best_measurement is not None:
        plt.scatter(best_measurement[0], best_measurement[1], marker = '+', color='red', s=100, label='Best Measurement', edgecolor='black')

    plt.xlabel('Delay Time (dt)')
    plt.ylabel('Fast Axis Orientation (phi)')
    plt.title('Shear-Wave Splitting Clusters')
    plt.xlim(0, 20)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.grid()
    plt.show()

# Now we bring it all together: calculate stopping criteria, and perform hierarchical clustering from M = N to our stopping criteria.
# Then we calculate overall cluster variance for each remaining cluster and select the cluster with the minimum overall variance as the best cluster.

def cluster_splitting_measurements(measurements, N_c_min=2):
    """
    Perform hierarchical clustering on shear-wave splitting measurements and select the best cluster.
    Parameters:
    -----------
    measurements: A list of tuples representing shear-wave splitting measurements as (delay_time, fast_axis_orientation)
    N_c_min: Minimum number of data points in a cluster to be considered valid (default: 2)

    Returns:
    --------
    best_cluster: The best cluster selected based on minimum overall variance
    best_measurement: The best measurement from the best cluster
    """
    # Initialize clusters with each measurement as its own cluster
    clusters = [[measurement] for measurement in measurements]
    cluster_centers = [measurement for measurement in measurements]

    N = len(measurements)
    
    # Calculate initial overall center and M_opt
    overall_center = calculate_center_from_cluster(measurements)

    # Calculate M_opt just once at the start
    M_opt = optimum_num_clusters(clusters, cluster_centers, overall_center, N)
    print(f"Optimum number of clusters determined: M_opt = {M_opt}")
    
    # Hierarchical clustering loop
    while len(clusters) > 1:
        
        # Output progress
        print(f"Current number of clusters: {len(clusters)}, Optimum number of clusters: {M_opt}")
        
        # Check stopping criterion
        if len(clusters) <= M_opt:
            break
        
        # Calculate distances between all cluster pairs
        min_distance = float('inf')
        merge_i, merge_j = 0, 1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Calculate Euclidean distance between cluster centers
                dt_diff = cluster_centers[i][0] - cluster_centers[j][0]
                phi_diff = cluster_centers[i][1] - cluster_centers[j][1]
                distance = np.sqrt(dt_diff**2 + phi_diff**2)
                
                if distance < min_distance:
                    min_distance = distance
                    merge_i, merge_j = i, j
        
        # Merge the closest pair of clusters
        merged_cluster = clusters[merge_i] + clusters[merge_j]
        
        # Recalculate center for merged cluster
        merged_center = calculate_center_from_cluster(merged_cluster)
        
        # Update clusters and cluster_centers
        new_clusters = []
        new_centers = []
        for idx in range(len(clusters)):
            if idx != merge_i and idx != merge_j:
                new_clusters.append(clusters[idx])
                new_centers.append(cluster_centers[idx])
        new_clusters.append(merged_cluster)
        new_centers.append(merged_center)
        
        clusters = new_clusters
        cluster_centers = new_centers

    # Filter clusters based on N_c_min
    valid_clusters = [cluster for cluster in clusters if len(cluster) >= N_c_min]

    if not valid_clusters:
        return None, None  # No stable solution

    # Calculate overall variance for each valid cluster and select the best one
    min_variance = float('inf')
    best_cluster = None
    best_cluster_center = None

    for cluster in valid_clusters:
        center = calculate_center_from_cluster(cluster)
        variance = overall_cluster_variance(cluster, center)

        if variance < min_variance:
            min_variance = variance
            best_cluster = cluster
            best_cluster_center = center

    # Select the best measurement from the best cluster as the measurement with the smallest distance to cluster center
    best_measurement = min(best_cluster, key=lambda point: 
                           np.sqrt((point[0] - best_cluster_center[0])**2 +
                                   (point[1] - best_cluster_center[1])**2))

    return best_cluster, best_measurement

# Finally, write a function that calls setup_windows, split_all_windows, and cluster_splitting_measurements in sequence
# We will apply run this function to a given event, inside of perform_splitting_on_organized_waveforms in splitting_functions.py

def teanby_clustering_analysis(event_data,
                              T_beg_1, T_end_0,
                              dT_beg, dT_end,
                              N_beg, N_end,
                              min_lag=0, max_lag=60,
                              Nlags=60, Nangles=90,
                              flag_adapt_window=True,
                              flag_adapt_maxlag=True,
                              min_thres=0.5,
                              min_numbers=2,
                              plot_results=False,
                              output_dir=None,
                              N_c_min=2):
    """
    Perform Teanby-style clustering analysis on shear-wave splitting measurements for a given event.
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms
    T_beg_1 : float
        Start time of the first window
    T_end_0 : float
        End time of the last window
    dT_beg : float
        Duration of the beginning windows
    dT_end : float
        Duration of the ending windows
    N_beg : int
        Number of beginning windows
    N_end : int
        Number of ending windows
    min_lag : int
        Minimum delay time in samples
    max_lag : int
        Maximum delay time in samples
    Nlags : int
        Number of lag values to test
    Nangles : int
        Number of angle values to test
    flag_adapt_window : bool
        Whether to adapt window based on dominant period
    flag_adapt_maxlag : bool
        Whether to adapt max_lag based on dominant period
    min_thres : float
        Threshold for minimum selection (0-1)
    min_numbers : int
        Maximum number of minima to keep
    plot_results : bool
        Whether to generate diagnostic plots (default: False)
    output_dir : str or None
        Directory to save plots (if None and plot_results=True, displays interactively)
    N_c_min : int
        Minimum number of data points in a cluster to be considered valid (default: 2)

    Returns:
    --------
    best_cluster: The best cluster selected based on minimum overall variance
    best_measurement: The best measurement from the best cluster
    """

    # Step 1: Set up windows
    windows = setup_windows(T_beg_1, T_end_0, dT_beg, dT_end, N_beg, N_end)
    # Step 2: Perform splitting analysis for all windows
    splitting_results = split_all_windows(event_data = event_data,
                                        windows = windows,
                                        N_beg=N_beg, N_end=N_end,
                                        min_lag=min_lag,
                                        max_lag=max_lag,
                                        Nlags=Nlags,
                                        Nangles=Nangles,
                                        flag_adapt_window=flag_adapt_window,
                                        flag_adapt_maxlag=flag_adapt_maxlag,
                                        min_thres=min_thres,
                                        min_numbers=min_numbers,
                                        plot_results=plot_results,
                                        output_dir=output_dir)
    
    # Step 3: Extract measurements from results
    measurements = []
    for result in splitting_results:
        window_id, result_dict = result
        if result_dict is not None:
            dt = result_dict['dt_samples']
            phi = result_dict['phi_rad']
            measurements.append((dt, phi))

    # Step 4: Perform clustering on measurements
    best_cluster, best_measurement = cluster_splitting_measurements(measurements, N_c_min=N_c_min)

    # Calc best_cluster center for printing
    if best_cluster is not None:
        best_cluster_center = calculate_center_from_cluster(best_cluster)
        print()
        print(f"Best cluster center: δt = {best_cluster_center[0]} samples, φ = {best_cluster_center[1]} rad")

    print()
    print(f"Best cluster: δt = {best_cluster_center[0]} samples, φ = {best_cluster_center[1]} rad, with {len(best_cluster)} points")
    print()
    print(f"Best measurement from best cluster: δt = {best_measurement[0]} samples, φ = {best_measurement[1]} rad")
    print()

    # Step 5: Wrap results into a results dictionary output format, like that of perform_splitting_analysis_baillard
    # We want to keep all the other event information from perform_splitting_analysis_baillard as well
    # Pull the results_dict from splitting_results for the window_id of best_measurement

    if best_measurement is not None:
        for result in splitting_results:
            window_id, result_dict = result
            if result_dict is not None:
                dt = result_dict['dt_samples']
                phi = result_dict['phi_rad']
                if (dt, phi) == best_measurement:
                    results = result_dict
                    break

    # Step 6: Plot measurements and clusters
    plot_measurements(measurements, best_cluster, best_measurement)

    # Step 7: Return results
    return results

# End of teanby_clustering.py
# We will implement teanby_clustering_analysis inside of perform_splitting_on_organized_waveforms in splitting_functions.py