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

        # initial fit values for linear trend
        if detrend.lower() in 'linear':
            p0 = [(ydat[-1] - ydat[0]) / len(ydat), ydat.mean()]

        # initial fit values for sinusoidal trend
        if detrend.lower() in 'sinusoidal' or detrend.lower() in 'sine':
            p0 = [
                1, (np.pi * 2 * np.abs(np.argmax(ydat) - np.argmin(ydat))),
                ydat.mean(), 0
            ]

        # initial fit values for inverse tangent trend
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
