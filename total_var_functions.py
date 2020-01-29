#!/usr/bin/env python
"""
DESCRIPTION

    Functions for the total variation of atmospheric data

AUTHOR

   Nicholas Hamilton
   nicholas.hamilton@nrel.gov

   Date:

   29 January 2020
"""

import numpy as np
import pandas as pd
import inspect
import scipy as sp


#######################################
def standardize_data(data, axis=0):
    '''
    Standardize data - calculate z-score.
    '''
    dnorm = (data - data.mean(axis=axis)) / data.std(axis=axis)
    return dnorm


# #######################################
# def total_variation(data):
#     '''
#     calculate the total variation of a data as the vector norm of the eigenvalues of the covariance matrix of the input data.

#     Parameters
#     ----------
#     data : pd.dataframe, np.ndarray
#         data

#     Returns
#     -------
#     tvar: float
#         total variation of the input data
#     V: np.array
#         eigenvalues of the input data's covariance matrix
#     S: np.array
#         eigenvectors of the input data's covariance matrix

#     '''
#     if 'pandas' in str(type(data)):
#         data = data.values
#     cov = np.cov(data.T)
#     V, S = np.linalg.eig(cov)
#     dpro = np.prod(V)

#     return tvar


#######################################
def covdet(data):
    '''
    calculate the total variation of a data as the determinant of the covariance matrix.

    Parameters
    ----------
    data : pd.dataframe, np.ndarray
        data

    Returns
    -------
    tvar: float
        total variation of the input data
    V: np.array
        eigenvalues of the input data's covariance matrix
    S: np.array
        eigenvectors of the input data's covariance matrix

    '''
    if 'pandas' in str(type(data)):
        data = data.values
    cov = np.cov(data.T)
    return np.linalg.det(cov)


#######################################
def mahalanobis_distance(data):
    '''
    Calculate the distance in standard deviations from the center of the data.
    Assumes data is a normally distrubuted multi-dimensional random variable.

    Parameters
    ----------
    data : np.array [ndata x ndims]
        input data array

    Returns
    -------
    MD : np.array [ndata]
        Mahalanobis distance
    '''
    # inverse of covariance matrix
    VI = np.linalg.inv(np.cov(data.T))

    # estimate of center of data
    center = np.repeat(data.mean(axis=0)[np.newaxis, :], data.shape[0], axis=0)

    MD = np.diag(
        np.sqrt(np.dot(np.dot((data - center), VI), (data - center).T)))

    return MD


#######################################
def linearfit(x, a, b):
    '''
    linear fit function
    '''
    return a * x + b


#######################################
def quadfit(x, a, b, c, x0):
    '''
    quadriatic fit function
    '''
    return a * (x - x0)**2 + b * (x - x0) + c


#######################################
def sinefit(x, a, b, c, x0):
    '''
    sine fit function
    '''
    return a * np.sin((x - x0) * 2 * np.pi * b) + c


#######################################
def invtanfit(x, a, b, c, x0):
    '''
    inverse tangent fit function
    '''
    return a * np.arctan(b * (x - x0)) + c


#######################################
def parse_fitfunc(detrend):
    '''
    [summary]

    Parameters
    ----------
    detrend : str
        detrend type (objective function flavor)

    Returns
    -------
    fitfunc : function
        objective function

    param_names : list
        names of each parameter
    '''
    if detrend is not None:

        # fit and remove linear trend
        if detrend.lower() in 'linear':
            fitfunc = linearfit
            param_names = ['slope', 'offset']

        # fit and remove sinusoidal
        if detrend.lower() in 'sinusoidal' or detrend.lower() in 'sine':
            fitfunc = sinefit
            param_names = ['amplitude', 'frequency', 'offset', 'phase']

        # fit and remove inverse tangent trend
        if detrend.lower() in 'inverse tangent':
            fitfunc = invtanfit
            param_names = ['amplitude', 'frequency', 'offset', 'phase']

    return fitfunc, param_names


#######################################
def parse_init_fitvals(detrend, ydat):
    '''
    [summary]

    Parameters
    ----------
    detrend : str
        detrend type (objective function flavor)
    ydat : np.ndarray
        data against which to fit

    Returns
    -------
    p0 : np.array, None
        inital values for fitting

    '''
    if detrend is not None:
        if detrend.lower() in 'linear':
            p0 = [(ydat[-1] - ydat[0]) / len(ydat), ydat.mean()]
        if detrend.lower() in 'sinusoidal' or detrend.lower() in 'sine':
            p0 = [
                1, 1 / (np.pi * 2 * np.abs(np.argmax(ydat) - np.argmin(ydat))),
                ydat.mean(), 0
            ]
        if detrend.lower() in 'inverse tangent':
            p0 = [1, 1, 1, len(ydat) / 2]
    return p0


#######################################
def find_outliers(data, threshold=3, searchtype=None):
    '''
    [summary]

    Parameters
    ----------
    data : [type]
        [description]
    threshold : int, optional
        [description] (the default is 3, which [default_description])

    Returns
    -------
    [type]
        [description]
    '''
    data_std = data.std(axis=0)

    outliers = np.zeros((0, 2))
    outlier_index = []

    if searchtype is None:
        for ii in range(data.shape[1]):
            tmp = np.abs(data[:, ii]) > threshold * data_std[ii]
            outlier_index.append([i for i, x in enumerate(tmp) if x])
            tmp = data[tmp]
            outliers = np.vstack([outliers, tmp])
        outlier_index = [item for sublist in outlier_index for item in sublist]
        clean_data = np.delete(data, outlier_index, axis=0)

    elif searchtype.lower() in 'mahalanobis':
        MD = mahalanobis_distance(data)
        tmp = MD > threshold
        outlier_index.append([i for i, x in enumerate(tmp) if x])
        tmp = data[tmp]
        outliers = np.vstack([outliers, tmp])
        outlier_index = [item for sublist in outlier_index for item in sublist]
        clean_data = np.delete(data, outlier_index, axis=0)

    return clean_data, outliers, outlier_index


# #######################################
# def total_variation(data,
#                     blocksize=60,
#                     detrend=None,
#                     column=None,
#                     window='slide',
#                     fitcurve=None,
#                     normalize_method='normal',
#                     metric='tvar'):
#     '''
#     Calculate the total variation of a dataset of block of data.

#     Parameters
#     ----------
#     data: df.DataFrame, np.ndarray
#         input data of format (timeseries x datachannels)

#     blocksize: int
#         size of blocks for which to calculate total variation
#         blocksize reflects the duration of each period in minutes

#     detrend: None, list
#         list of variables to detrend (linear)

#     Returns
#     --------
#     totalvar: df.Series
#         Series of the total variation for each time period.
#     '''

#     if window == 'slide':
#         timeind = data.index
#     elif window == 'block':
#         # make new time index (hourly)
#         timeind = pd.DatetimeIndex(start=data.index[0],
#                                    freq='{}T'.format(blocksize),
#                                    end=data.index[-1])

#     nblocks = len(timeind)

#     # allocate space for totalvar
#     totalvar = np.zeros(nblocks)

#     # if detrending data, parse objective functino, and the number of required arguments
#     if detrend is not None:
#         fitfunc, param_names = parse_fitfunc(detrend)
#         param_names.append('residual')
#         tmp = inspect.getfullargspec(fitfunc)
#         nargs = len(tmp.args)
#         fits = np.zeros((nblocks, nargs))  # slope, offset, residual
#         x = np.arange(blocksize)

#     nskip = 0
#     timedelay = pd.Timedelta('{}m'.format(blocksize - 1))

#     # loop over data blocks
#     for ii in range(nblocks - 1):

#         startind = timeind[ii]
#         endind = startind + timedelay

#         block = data[startind:endind].dropna(how='any').copy()
#         # block = data.iloc[startind:startind + blocksize].copy()

#         if (len(block) < blocksize) | (any(block.std() == 0)) | (
#             (np.abs(block.wdir.diff()).max() > 60)):
#             nskip += 1
#             continue

#         if detrend is not None:
#             p0 = parse_init_fitvals(detrend, block[column].values)

#             # try to make a good fit. Sometimes it just doesn't go well...
#             try:
#                 fittest = sciop.curve_fit(fitfunc, x, block[column], p0)
#                 fitparams, _ = fittest
#                 fitcurve = fitfunc(x, *fitparams)
#                 residual = np.linalg.norm(block[column] -
#                                           fitfunc(x, *fitparams))**2
#             except:
#                 # print('excepted')
#                 fitparams = p0
#                 fitcurve = fitfunc(x, *fitparams)
#                 residual = np.nan
#                 # fits[ii, :] = np.nan
#                 # continue

#             block[column] -= (fitcurve + block[column].mean())
#             fits[ii, :-1] = fitparams
#             fits[ii, -1] = np.linalg.norm(residual)

#         # normalize data
#         if normalize_method is 'normal':
#             block = normalize_data(block)
#         elif normalize_method is 'standard':
#             block = standardize_data(block)

#         if metric is 'tvar':
#             # total variation of the chunk
#             totalvar[ii], V, _ = pca_var(block)
#         elif metric is 'dpro':
#             totalvar[ii] = pca_dpro(block)
#         elif metric is 'covnorm':
#             totalvar[ii] = covnorm(block)
#         elif metric is 'covdet':
#             totalvar[ii] = covdet(block)

#     # make dataframe for total variation
#     totalvar = pd.DataFrame(data=totalvar, index=timeind, columns=['totalvar'])

#     if detrend is not None:

#         fitcols = {'_'.join([column, x]): np.array([]) for x in param_names}
#         fits = pd.DataFrame(index=totalvar.index, data=fits, columns=fitcols)
#         totalvar = totalvar.join(fits)

#     # replace 0.0 with np.nan
#     totalvar.replace(0, np.nan, inplace=True)
#     # Drop all nan values
#     totalvar.dropna(inplace=True, how='any')

#     return totalvar