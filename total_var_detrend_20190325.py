#!/usr/bin/env python
"""
DESCRIPTION

    Load met mast data
    QC
    normalize
    Calculate Total variation in sliding window
    remove standard trends
        linear
        wave
        direction change
    save data

AUTHOR

   Nicholas Hamilton
   nicholas.hamilton@nrel.gov

   Date: 20190319
"""

import sys, os, traceback, optparse
import datetime
import numpy as np
# import scipy as sp
import pandas as pd

import total_var_functions as TV

###########  Main stuff
today = datetime.date.today().strftime('%Y%m%d')

figpath = os.path.abspath('../figs/')
datadir = os.path.abspath('../../data')

#### read pre-processed data.
metdf = pd.read_csv(os.path.join(datadir, 'metdata_2009.csv'))
metdf.dropna(how='any', inplace=True)
metdf.index = pd.DatetimeIndex(metdf['time'])
metdf.drop(labels=['time'], axis=1, inplace=True)
metdf.ti *= 100
# keep a normalized version around
metdfnorm = (metdf - metdf.mean()) / (metdf.std())

#### Limit metdf to a subset for testing
# metdfnorm = metdfnorm.iloc[0:12000]

##### calculate total variation in a sliding window
# NO objective function => quiescent conditions
slidedf = TV.total_variation(metdfnorm, blocksize=120, window='slide')
slidedf.to_csv(os.path.join(datadir, 'tvar_quiescent_{}.csv'.format(today)))

#### calculate total variation in a sliding window
# LINEAR objective function => wind speed ramp
tvar_linear = TV.total_variation(metdfnorm,
                                 blocksize=120,
                                 detrend='linear',
                                 column='wspd')
tvar_linear.to_csv(os.path.join(datadir, 'tvar_linear_{}.csv'.format(today)))

##### calculate total variation in a sliding window
# LINEAR objective function => wind speed wave
tvar_wave = TV.total_variation(metdfnorm,
                               blocksize=120,
                               detrend='sine',
                               column='wspd')
tvar_wave.to_csv(os.path.join(datadir, 'tvar_wave_{}.csv'.format(today)))

# Testing the inverse tangent fit fixing the phase to len(x)/2
##### calculate total variation in a sliding window
# LINEAR objective function => wind direction change
tvar_dchg = TV.total_variation(metdfnorm,
                               blocksize=120,
                               detrend='inverse',
                               column='wdir')
tvar_dchg.to_csv(os.path.join(datadir, 'tvar_dchg_{}.csv'.format(today)))