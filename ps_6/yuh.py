import numpy as np
import astropy
import astropy.io.fits
import matplotlib.pyplot as plt
import time

hdu_list = astropy.io.fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

galaxy_indices = [0, 1, 2, 3, 4, 5]
print('Part A: plotting galaxy spectra...')
for idx in galaxy_indices:
    plt.plot(logwave, flux[idx], label=f'Galaxy {idx+1}')
    plt.xlabel('Log wavelength (Å)')
    plt.ylabel('Flux ($10^{−17} erg s^{−1}cm^{−2} Å^{-1}$)')
    plt.title(f'Part A: wavelength vs. Flux for galaxy # {idx+1}')
    plt.show()
    print(f'galaxy {idx + 1} plotted!')

# Part B: Normalize flux
print('Part B: normalizing flux and checking with plot: ')
flux_sum = np.sum(flux, axis=1)
# Normalize each galaxy's flux by dividing by its total flux
flux_normalized = flux / np.tile(flux_sum, (np.shape(flux)[1], 1)).T

# Check that data is normalized for all galaxies
plt.plot(np.sum(flux_normalized, axis=1))
plt.xlabel('Galaxy index')  # Label the x-axis
plt.ylabel('Normalized Total Flux')
plt.ylim(-0.5,2)
plt.title('Normalized Total Flux for each galaxy')
plt.show()

# Part C
print("part C: centering the spectra... plotting first galaxy to check.")
means_norm = np.mean(flux_normalized, axis=1)
# Subtract the mean spectrum from the normalized spectra
flux_0_mean = flux_normalized - np.tile(means_norm, (np.shape(flux)[1], 1)).T
print('flux shape ',flux_0_mean.shape)

# Plot the 0-mean normalized flux for the first galaxy just to check if its centered 
plt.plot(logwave, flux_0_mean[0, :])
plt.ylabel('Normalized 0-mean Flux')
plt.xlabel('log Wavelength [$Angstrom$]')
plt.title('Part C: 0-Mean Normalized Flux of the First Galaxy')
plt.show()


#Part D perform PCA (i used the first 500 galaxies)
def sorted_eigs(r, return_eigvalues=False):
    """
    Calculate the eigenvectors and eigenvalues of the correlation matrix of r
    -----------------------------------------------------
    """
    corr = r.T @ r #/r.shape[0]
    eigs = np.linalg.eig(corr)  # calculate eigenvectors and values of original
    arg = np.argsort(eigs[0])[::-1]  # get indices for sorted eigenvalues
    eigvec = eigs[1][:, arg]  # sort eigenvectors
    eig = eigs[0][arg]  # sort eigenvalues

    print('corr shape: ',corr.shape)
    if return_eigvalues:
        return eig, eigvec
    else:
        return eigvec


#Use data for only the first 500 galaxies
print('part D: Performing PCA \n getting eigen vectors...')
r_subset = flux_0_mean[:500,:] #create a subset of the first 500 rows of flux 
logwave_subset = logwave #this stays the same


#Time Eigenvalue Decomposition (pca)
start_time_pca = time.time()
eigvals, eigvecs = sorted_eigs(r_subset, return_eigvalues=True)
end_time_pca = time.time()

pca_time = end_time_pca - start_time_pca
print(f"Time taken for Principle Component Analysis: {pca_time} seconds")
print(f'plotting the first five eigen vectors found from pca: ')
#Plot the first five eigenvectors from EVD
for i in range(5):
    plt.plot(logwave_subset, eigvecs[:, i], label=f'Eigenvector_pca {i+1}', color = 'blue')
   # plt.plot(logwave_subset, eigvecs_svd[:, i], label=f'Eigenvector_svd {i+1}', color = 'red' )
    plt.legend()
    plt.ylabel('Normalized 0-mean Flux', fontsize=16)
    plt.xlabel('log Wavelength [$Angstrom$]', fontsize=16)
    plt.title('Part D: First Five Eigenvectors (PCA)', fontsize=16)
    plt.show()

print('Part E: Using SVD to find eigen vectors')
#PArt e doing svd and comparing our eigen vectors 
#Time Singular Value Decomposition (SVD)
start_time_svd = time.time()
U, S, V = np.linalg.svd(r_subset, full_matrices=True)
end_time_svd = time.time()

#timing SVD
svd_time = end_time_svd - start_time_svd
print(f"Time taken for Singular Value Decomposition (SVD): {svd_time} seconds")
# Print time comparisons
print(f"PCA Time: {pca_time} seconds")
print(f"SVD Time: {svd_time} seconds")


# rows of Vh are eigenvectors
eigvecs_svd = V.T
eigvals_svd = S**2
svd_sort = np.argsort(eigvals_svd)[::-1]
eigvecs_svd = eigvecs_svd[:, svd_sort]
eigvals_svd = eigvals_svd[svd_sort]
print("plotting svd eigen vectors over pca eigen vectors...")
for i in range(5):
    plt.plot(logwave_subset, eigvecs[:, i], label=f'Eigenvector_pca {i+1}', color = 'blue',)
    plt.plot(logwave_subset, eigvecs_svd[:, i], label=f'Eigenvector_svd {i+1}', color = 'red',ls = ":" )
    plt.legend()
    plt.ylabel('Normalized 0-mean Flux', fontsize=16)
    plt.xlabel('log Wavelength [$Angstrom$]', fontsize=16)
    plt.title('Part D: First Five Eigenvectors (PCA)', fontsize=16)
    plt.show()
'''
for i in range(5):
    plt.plot(logwave_subset, eigvecs_svd[:, i], label=f'Eigenvector {i+1}')
    plt.legend()
    plt.ylabel('Normalized 0-mean Flux')
    plt.xlabel('log Wavelength [$Angstrom$]')
    plt.title('Part D: First Five Eigenvectors (svd)')
    plt.show()
'''
'''
# Plot eigenvalues comparison
plt.plot(eigvals_svd, eigvals[:500], 'o')
plt.xlabel('SVD eigenvalues')
plt.ylabel('PCA eigenvalues')
plt.title("Comparing SVD and PCA eigenvalues")
plt.show()
'''

#part g
print('part g the reduced pca part. \n getting reduced data (coefficients)')
def PCA(l, r, project=True):
    """
    Perform PCA dimensionality reduction.
    --------------------------------------------------------------------------------------
    l: Number of principal components to keep
    r: Data matrix (e.g., galaxy spectra)
    project: If True, return the reconstructed spectra; if False, return the reduced data.
    """
    eigvector = sorted_eigs(r)
    eigvec = eigvector[:, :l]  # Keep only the first l eigenvectors (principal components)
    reduced_wavelength_data = np.dot(eigvec.T, r.T)  # Project the data onto the components
    if project:
        return np.dot(eigvec, reduced_wavelength_data).T  # Reconstruct data in the reduced space
    else:
        return reduced_wavelength_data.T  # Return the reduced data (PCA representation)

# Example usage to reduce to 5 principal components
nc = 5  # Number of principal components
reduced_data = PCA(nc, flux_0_mean, project=False) #coefficients shape (500,5)
print ('reduced data shape (shape of coefficient)', reduced_data.shape)

c0 = reduced_data [:,0]
c1 = reduced_data [:,1]
c2 = reduced_data [:,2]
print("plotting c0 vs c1")
plt.scatter(c0, c1, label='c_0 vs c_1', alpha=0.5)
plt.xlabel('c_0')
plt.ylabel('c_1')
plt.title('c_0 vs c_1')
plt.legend()
plt.grid(True)
plt.show()
print("plotting c0 vs c2")
# Create a scatter plot for c0 vs c2
plt.scatter(c0, c2, label='c_0 vs c_2', alpha=0.5)
plt.xlabel('c_0')
plt.ylabel('c_2')
plt.title('c_0 vs c_2')
plt.legend()
plt.grid(True)
plt.show()

print('reconstructing galaxy 1 spectrum using one eigenvector')
# Using only the first eigenvector 
plt.plot(logwave_subset, PCA(1,r_subset)[0,:], label = 'l = 1')
plt.plot(logwave_subset, r_subset[0,:], label = 'original data')
plt.ylabel('normalized 0-mean flux', fontsize = 16)
plt.xlabel('wavelength [$A$]', fontsize = 16)
plt.title('Spectrum of Galaxy 1 approximated with 1 eigenvector')
plt.legend()
plt.show()

print('reconstructing galaxy 1 using 5 eigenvectors')
# Using only the first 5 eigenvector 
plt.plot(logwave_subset, PCA(5,r_subset)[0,:], label = 'l = 5')
plt.plot(logwave_subset, r_subset[0,:], label = 'original data')
plt.ylabel('normalized 0-mean flux', fontsize = 16)
plt.xlabel('wavelength [$A$]', fontsize = 16)
plt.title('Spectrum of Galaxy 1 approximated with 5 eigenvectors')
plt.legend()
plt.show()
print('calculating residuals for galaxy one with varying eigenvectors')
#part i 

def calculate_squared_residuals(r_original, r_approx):
    """
    Calculate the squared residuals between the original and approximated spectra.
    """
    # Compute the squared differences between original and approximate spectra
    squared_diff = np.sum((r_original - r_approx) ** 2, axis=0)
    # Compute the squared sum of the original spectra
    squared_original_sum = np.sum(r_original ** 2, axis=0)
    # Calculate the squared residuals
    squared_residual = squared_diff / squared_original_sum
    return squared_residual

Nc_values = np.arange(1, 21)
squared_residuals = []

#Iterate over values of Nc (number of principal components)
for Nc in Nc_values:
    #Approximate spectra using Nc principal components
    approx_spectra = PCA(Nc, r_subset, project=True)[0,:]
    
    #Calculate the squared residuals
    residuals = calculate_squared_residuals(r_subset[0,:], approx_spectra)
    
    #Calculate and store the root mean squared (RMS) residual for this Nc
    rms_residual = np.sqrt(np.mean(residuals))
    squared_residuals.append(rms_residual)

#Plot the RMS residuals as a function of Nc
plt.plot(Nc_values, squared_residuals, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Principal Components (Nc)', fontsize=14)
plt.ylabel('Root-Mean-Squared Residual', fontsize=14)
plt.title('RMS Residuals vs Number of Principal Components', fontsize=16)
plt.grid(True)
plt.show()

#Print the RMS residual for Nc = 20
print(f'Root-Mean-Squared Residual for Nc = 20: {squared_residuals[-1]:.6f}')
