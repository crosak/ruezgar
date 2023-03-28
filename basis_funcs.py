  # Import libraries 
import numpy as np
from scipy import constants
from scipy import special

def h_ml(alpha, c_ml, m, l, omega_drag, theta, phi):
    '''
    The h_ml(theta, phi) basis functions as described in Morris et.al.(2021)
    
    Parameters
    ----------
    alpha: float
        Dimensionless fluid number constructed from Rossby, Reynolds and Prandtl numbers
    c_ml: np.ndarray
        Array filled with spherical harmonic coefficients
    m,l : int
        Indices of the basis functions (Analogous to the quantum numbers for spherical harmonics)
    omega_drag: float
        Dimensionless drag frequency (Normalized by twice the angular velocity of rotation?)
    theta: float
        Latitude
    phi: float 
        Longitude
    
    Returns
    -------
    h_ml: float
        Basis function for a single combination of m and l values
    '''
    mu = np.cos(theta)
    mu_tilde = alpha * mu
    c1 = (c_ml[l][m] / (omega_drag**2 * alpha**4 + m**2)) * np.exp(-0.5*mu_tilde**2)
    c2 = mu * m * special.eval_hermite(l,mu_tilde)*np.cos(m*phi)
    c3 = alpha * omega_drag * (2*l*special.eval_hermite(l-1,mu_tilde) - \
         mu_tilde * special.eval_hermite(l,mu_tilde))*np.sin(m*phi)
    return (c1 * (c2 + c3))

def T(theta, phi, c_ml, l_max, alpha, omega_drag, f, T_star, R_star, a, A_B = 0.0):
    '''
    2D temperature maps as described in Morris et.al.(2021)
    
    Parameters
    ----------
    theta: float
        Latitude 
    phi: float
        Longitude
    l_max: int
        Maximum number of l-terms (Analogous to resolution)
    alpha: float
        Dimensionless fluid number constructed from Rossby, Reynolds and Prandtl numbers
    omega_drag: float
        Dimensionless drag frequency (Normalized by twice the angular velocity of rotation?)
    f: float
        Greenhouse factor (There is also a geometric interpretation?)
    T_star: float
        Effective stellar temperature
    R_star: float
        Radius of the star
    a: float
        Semi-major axis
    A_B: float
        Bond albedo (Set to be zero as per convention)
    
    Returns
    -------
    T: float
        Temperature value at given a point on the solid angle(?)
    '''
    sum_term = 0.0
    T_tilde = T_star * np.sqrt(R_star/a) * f * (1 - A_B)**0.25
    for l in range(1, l_max+1):
        for m in range(-l, l+1):
            sum_term += h_ml(alpha, c_ml, m, l, omega_drag, theta, phi)
    return (T_tilde * (1.0 + sum_term))

def indexer(l, m):
    C_ml = [[0],
            [0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    C_ml[l][m] = 1
    return C_ml

def indexer_new(l_max, l, m):
    '''
    A generalized version of Brett's indexer with no l_max limitation.
    
    Parameters
    ----------
    l_max: int
    l: int
    m: int
    
    Returns
    -------
    C_ml: list
    '''
    C_ml = []
    for m_curr in range(l_max + 1):
        C_ml.append(list(np.zeros(2 * m_curr + 1)))
        
    C_ml[l][m] = 1
    return C_ml

def v_theta_ml(alpha, c_ml, m, l, theta, phi, prior):
    '''
    The v_theta_ml(theta, phi) basis functions as described in Heng et.al.(2014) formulated 
    using the same conventions as Morris et.al.(2021)
    
    Parameters
    ----------
    alpha: float
        Dimensionless fluid number constructed from Rossby, Reynolds and Prandtl numbers
    c_ml: np.ndarray
        Array filled with spherical harmonic coefficients
    m,l : int
        Indices of the basis functions (Analogous to the quantum numbers for spherical harmonics)
    theta: float
        Latitude
    phi: float 
        Longitude
    prior: float 
        A bias that we introduce to weigh parts of the map differently
    
    Returns
    -------
    v_theta_ml: float
        Basis function for a single combination of m and l values
    '''
    mu = np.cos(theta)
    mu_tilde = alpha * mu
    c1 = prior * (c_ml[l][m] / np.sin(theta)) * np.exp(-0.5*mu_tilde**2)
    c2 = special.eval_hermite(l,mu_tilde)*np.sin(m*phi)
    return c1*c2

def v_theta_ml_wosin(alpha, c_ml, m, l, theta, phi):
    '''
    The v_theta_ml(theta, phi) basis functions as described in Heng et.al.(2014) formulated 
    using the same conventions as Morris et.al.(2021)
    
    Parameters
    ----------
    alpha: float
        Dimensionless fluid number constructed from Rossby, Reynolds and Prandtl numbers
    c_ml: np.ndarray
        Array filled with spherical harmonic coefficients
    m,l : int
        Indices of the basis functions (Analogous to the quantum numbers for spherical harmonics)
    theta: float
        Latitude
    phi: float 
        Longitude
    prior: float 
        A bias that we introduce to weigh parts of the map differently
    
    Returns
    -------
    v_theta_ml: float
        Basis function for a single combination of m and l values
    '''
    mu = np.cos(theta)
    mu_tilde = alpha * mu
    c1 = c_ml[l][m] * np.exp(-0.5*mu_tilde**2)
    c2 = special.eval_hermite(l,mu_tilde)*np.sin(m*phi)
    return c1*c2

def v_phi_ml(alpha, c_ml, m, l, omega_drag, theta, phi, prior):
    '''
    The v_phi_ml(theta, phi) basis functions as described in Heng et.al.(2014) formulated 
    using the same conventions as Morris et.al.(2021)
    
    Parameters
    ----------
    alpha: float
        Dimensionless fluid number constructed from Rossby, Reynolds and Prandtl numbers
    c_ml: np.ndarray
        Array filled with spherical harmonic coefficients
    m,l : int
        Indices of the basis functions (Analogous to the quantum numbers for spherical harmonics)
    omega_drag: float
        Dimensionless drag frequency (Normalized by twice the angular velocity of rotation?)
    theta: float
        Latitude
    phi: float 
        Longitude
    prior: float 
        A bias that we introduce to weigh parts of the map differently
    
    Returns
    -------
    v_phi_ml: float
        Basis function for a single combination of m and l values
    '''
    mu = np.cos(theta)
    mu_tilde = alpha * mu
    c1 = prior * (c_ml[l][m] / ((omega_drag**2 * alpha**4 + m**2)*np.sin(theta))) * np.exp(-0.5*mu_tilde**2)
    c2 = omega_drag**2 * alpha**4 * special.eval_hermite(l,mu_tilde)*np.sin(m*phi)
    c3 = alpha * m * (2*l*special.eval_hermite(l-1,mu_tilde) - \
         mu_tilde * special.eval_hermite(l,mu_tilde))*np.cos(m*phi)
    return (c1 * (c2 + c3))

def v_phi_ml_wosin(alpha, c_ml, m, l, omega_drag, theta, phi):
    '''
    The v_phi_ml(theta, phi) basis functions as described in Heng et.al.(2014) formulated 
    using the same conventions as Morris et.al.(2021)
    
    Parameters
    ----------
    alpha: float
        Dimensionless fluid number constructed from Rossby, Reynolds and Prandtl numbers
    c_ml: np.ndarray
        Array filled with spherical harmonic coefficients
    m,l : int
        Indices of the basis functions (Analogous to the quantum numbers for spherical harmonics)
    omega_drag: float
        Dimensionless drag frequency (Normalized by twice the angular velocity of rotation?)
    theta: float
        Latitude
    phi: float 
        Longitude
    prior: float 
        A bias that we introduce to weigh parts of the map differently
    
    Returns
    -------
    v_phi_ml: float
        Basis function for a single combination of m and l values
    '''
    mu = np.cos(theta)
    mu_tilde = alpha * mu
    c1 = (c_ml[l][m] / (omega_drag**2 * alpha**4 + m**2)) * np.exp(-0.5*mu_tilde**2)
    c2 = omega_drag**2 * alpha**4 * special.eval_hermite(l,mu_tilde)*np.sin(m*phi)
    c3 = alpha * m * (2*l*special.eval_hermite(l-1,mu_tilde) - \
         mu_tilde * special.eval_hermite(l,mu_tilde))*np.cos(m*phi)
    return (c1 * (c2 + c3))

def hml_linear_solve(P, w, eps, m):
    '''
    Solver for the fitting procedure described in Morris et.al. (2021) 
    
    Parameters
    ----------
    P: ~numpy.array
        Design matrix
    w: ~numpy.array
        Weight matrix 
    eps: float
        Small positive constant
    m: ~numpy.array
        Ground truth
    
    Returns
    -------
    betas: ~numpy.array
        Fit coefficients
    '''
    
    # Solve 
    PTSinv = P.T * (w ** 2)[None, :]
    print(f"{P.T.shape} @ {(w ** 2)[None, :].shape} = {PTSinv.shape} : PTSinv")
    Q = np.linalg.solve(PTSinv @ P + eps * np.eye(P.shape[1]), PTSinv)
    betas = Q @ m.flatten()
    return betas

def bin_centers(x):
    '''
    Function for calculating the binned grid from its edges
    '''
    return 0.5 * (x[1:] + x[:-1])

## Experimental stuff

def angular_transformation(v_theta, v_phi):
    '''
    An experimental transformation function to get the orientations of each vector and end up with a map
    which does not give zero weight to zonal flows.
    
    Parameters
    ----------
    v_theta: ~numpy.array
        Meridional component of the velocity vector
    v_phi: ~numpy.array
        Zonal component of the velocity vector
    
    Returns
    -------
    experiment: ~numpy.array
        Transformed angular map 
    '''
    experiment = (np.arctan2(v_theta, v_phi) - np.pi/2) 
    experiment[experiment < -np.pi] += 2 * np.pi
    return experiment


def divergence(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    Parameters
    ----------
    f: List of ~numpy.arrays
        Every item of the list is one dimension of the vector field
    Returns
    -------
    Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])