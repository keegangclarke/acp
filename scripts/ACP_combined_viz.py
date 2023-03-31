# %%

# Imports
# Basic
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from os import listdir
import re
import gc
# Numbers
import pandas as pd
import numpy as np
# Graphic
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib import patheffects as pe
import seaborn as sns
# Models
import pmdarima as pm
from arch.univariate import ZeroMean, GARCH, Normal # 3 components of GARCH model
import tensorflow as tf
from tensorflow import keras
# Statistics
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS, GLS
# Special
from playsound import playsound

# %%
def ps(a=0):
    if a==0:
        playsound('C:/Windows/Media/Windows Notify.wav')
    elif a==1:
        for i in range(3):
            playsound('C:/Windows/Media/Ring06.wav')

# %%
# Data retrival
# LOAD NAMES
dmkt = "C:/Users/Keegan/OneDrive/1 Studies/2021 - 2022/5003W/3 - Dissertation/5-Data/cleaned/univariate/market_indices/market_names.csv"
mkt_names = pd.read_csv(dmkt).values.tolist()
mkt_names = [item for sublist in mkt_names for item in sublist]

# LOAD DATA
dd = "C:/Users/Keegan/OneDrive/1 Studies/2021 - 2022/5003W/3 - Dissertation/5-Data/cleaned/univariate/market_indices/"
dirs = []
data = OrderedDict()
for i in range(len(mkt_names)):
    # concat strings to get directories
    nm = mkt_names[i]
    directory = dd+nm+".csv"
    # store
    dirs.append(directory)
    data[nm] = pd.read_csv(directory)
    del directory, nm

unfinished = [
    'CAC.Index',
    'IMOEX.Index',
    'NIFTY.Index',
    'TFTSEMIB.Index'
    ]

# get AE data
d = "C:/Users/Keegan/OneDrive/1 Studies/2021 - 2022/5003W/3 - Dissertation/5-Data/results/fanomal/stats_models/ae_results/folder for mse/"
mse_filenames = listdir("C:/Users/Keegan/OneDrive/1 Studies/2021 - 2022/5003W/3 - Dissertation/5-Data/results/fanomal/stats_models/ae_results/folder for mse/")
ae_data = OrderedDict()
for i in range(len(mse_filenames)):
    nm = mse_filenames[i]
    nmc = str(nm).replace('_mse.csv','').upper()+".Index"
    directory = d+nm
    # store
    ae_data[nmc] = pd.read_csv(directory).rename(columns={'date':'date','0':'mse'})
    del directory, nm

# Constant specifications
# SCALING FACTOR
SCALE = 100
DESCALE = 1/SCALE

# GARCH
# Specify forecast horizon
H = 10

# AE datastructures
LAGS = 64

# Date parameters
E1 = pd.to_datetime("2020-01-13")
E2 = pd.to_datetime("2020-01-24")
E3 = pd.to_datetime("2020-02-24")
E4 = pd.to_datetime("2020-03-09")

CUT1 = pd.to_datetime("2019-04-29")
CUT2 = pd.to_datetime("2020-06-01")

EXCLUSION = pd.to_datetime("2010-01-01")

# Plot parameters
SCATTER = 25
ALPHA = 0.8
LINEWIDTH = 1.5

TITLE_FONT = {
        'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30
        }

AXIS_FONT = {
        'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22
        }

DATE_FORM  = DateFormatter('%Y-%m-%d')

MINIMUM = pd.to_datetime('2008-01-04')

# Notebook Settings
# set data splitting parameters
EVENT = E1

# Toggle Settings
store_stats_results = True
save_ae_results = True
show_plots = False
save_figs = True
verbose = False
run_ma = False

# Returns
var_mode = 'log_prices' # alternative = 'log_returns'

# Bounded Scaling
rescale = False

# Cutting out of sample
exclude = True

# Plot settings
plt.style.use('seaborn')
plt.rc('font', **AXIS_FONT)
plt.rc('xtick', labelsize=15)  
plt.rc('ytick', labelsize=15)

# For debugging
d_complete = 'C:/Users/Keegan/OneDrive/1 Studies/2021 - 2022/5003W/3 - Dissertation/5-Data/results/fanomal/'
Path(d_complete).mkdir(parents=True, exist_ok=True)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

mkt_chunks = [*chunks(mkt_names,11)]
# Ran into memory error
last_few = [
     'IBEX.Index',
     'INDU.Index',
     'MERVAL.Index'
     ]

# %%
# Data Wrangling
for name in mkt_names: #last_few #mkt_chunks
    # select data
    MKT = name #'SPX.Index' 
    dataset = data[MKT].copy(deep=True)
    ae = ae_data[MKT].copy(deep=True)
    ae['mse'] = ae['mse'].pow(0.5)
    ae['date'] = pd.to_datetime(ae['date'])
    ae = ae.rename(columns={'date':'date','mse':'rmse'})
    ae = ae.set_index('date')
    print("Beginning ACP on "+MKT)
    # format dates
    dataset['date'] = pd.to_datetime(dataset['date'].values)
    # change colname for ease of use
    dataset = dataset.rename(columns={'date':'date', MKT:'val'})
    # keep prices for ML
    prices = dataset.copy(deep=True).set_index('date')

    # Leave text to indicate that loop started for MKT
    with open(d_complete+'ACP completed indices'+'.txt', 'a') as f:
        f.write('\n\n'+MKT+' started.')

    if var_mode == 'log_returns':
        # extract price level, calc log_returns, store as dataset
        log_returns = np.log(dataset.val.shift(-1) / dataset.val)
        log_returns = log_returns[~np.isnan(log_returns)]

        dataset.val = log_returns
        del log_returns

    elif var_mode == 'log_prices':
        # extract price level, calc log_prices, store as dataset
        log_prices = np.log(np.copy(dataset.val.values))
        log_prices = log_prices[~np.isnan(log_prices)]

        dataset.val = log_prices
        del log_prices
    else:
        Exception("Variable calculation not specified. Please specify var_mode with one of the following: ['log_returns', 'log_prices']")

    # split dataset
    Y = dataset.val
    X = pd.concat([pd.to_datetime(dataset.date), pd.Series(range(len(Y)))], axis=1).rename({'date':'date',0:'index'},axis=1)
    # X_in = X.index[X.date < EVENT]
    # X_out = X.index[X.date >= EVENT]
    train = dataset[(dataset["date"] < EVENT)] # & (dataset['date'] > pd.to_datetime("2016-01-01")) 
    test = dataset[(dataset["date"] >= EVENT)]
    # Inspect
    if verbose:
        print(
            'TABULAR DATA:', '\n',
            'train.shape: ', train.shape, '\n',
            'test.shape: ', test.shape
            )

    # retain date seqs
    dtrain, dtest = pd.to_datetime(train.date), pd.to_datetime(test.date)
    # format structures (and dtypes) for arima
    Y, train, test = Y.values, train.val.values, test.val.values
    # Inspect and compare
    if verbose:
        print(
            'DATA VECTORS:', '\n',
            'train.shape: ', train.shape, '\n',
            'test.shape: ', test.shape
            )

    # %%
    # Storage Directories for Results
    d_stage1 = "C:/Users/Keegan/OneDrive/1 Studies/2021 - 2022/5003W/3 - Dissertation/5-Data/results/fanomal/stats_models/stage1_comparison/"
    d_stage1_arima = d_stage1+'arima/'
    d_garch_results = "C:/Users/Keegan/OneDrive/1 Studies/2021 - 2022/5003W/3 - Dissertation/5-Data/results/fanomal/stats_models/garch_results/all_garch_vols/"
    d_stage2 = "C:/Users/Keegan/OneDrive/1 Studies/2021 - 2022/5003W/3 - Dissertation/5-Data/results/fanomal/stats_models/stage2_comparison/"
    d_stage2_cu = "C:/Users/Keegan/OneDrive/1 Studies/2021 - 2022/5003W/3 - Dissertation/5-Data/results/fanomal/stats_models/stage2_comparison_close_up/"
    d_stage2_merged = "C:/Users/Keegan/OneDrive/1 Studies/2021 - 2022/5003W/3 - Dissertation/5-Data/results/fanomal/stats_models/stage2_comparison_merged/"
    # Ensure appropriate nested directories exit
    Path(d_stage1).mkdir(parents=True, exist_ok=True)
    Path(d_stage1_arima).mkdir(parents=True, exist_ok=True)
    Path(d_garch_results).mkdir(parents=True, exist_ok=True)
    Path(d_stage2).mkdir(parents=True, exist_ok=True)
    Path(d_stage2_cu).mkdir(parents=True, exist_ok=True)
    Path(d_stage2_merged).mkdir(parents=True, exist_ok=True)

    # %% [markdown]
    # .______________________________.
    # ARIMA Modelling
    ORDER = (1,1,1)
    # construct arima model of order ORDER, keeping data out of sample
    arima = pm.ARIMA(order=ORDER, out_of_sample_size=int(test.shape[0]))
    fit = arima.fit(Y)
    # predict in sample
    pred_in = fit.predict_in_sample(start=1, end=Y.shape[0], dynamic=False, return_conf_int=True)
    # extract upper & lower confidence intervals of predictions
    lw = pd.DataFrame({'date':dataset.date.values, 'lower':[pred_in[1][i][0] for i in range(pred_in[1].shape[0])]}).set_index('date')
    up = pd.DataFrame({'date':dataset.date.values, 'upper':[pred_in[1][i][1] for i in range(pred_in[1].shape[0])]}).set_index('date')
    # Build pd.DataFrames to make plotting easier
    predicted = pd.DataFrame({
        'date':dataset.date.values,
        # 'actual':Y,
        'predicted':pred_in[0],
        'lower':[pred_in[1][i][0] for i in range(pred_in[1].shape[0])],
        'upper':[pred_in[1][i][1] for i in range(pred_in[1].shape[0])]
        }).set_index('date')
    train_test = {
        'train':pd.DataFrame({'date':dtrain,'train':train}).set_index('date'),
        'test':pd.DataFrame({'date':dtest,'test':test}).set_index('date')
    }
    actual = dataset.set_index('date')

    # Put ARIMA resids into dataframe to make life easier
    arima_resids = pd.DataFrame(
        {'date':dataset.date.values[1:], 'arima_resids':fit.resid()[1:]*SCALE},
        copy=True).set_index('date')

    # In-sample Model
    garch = ZeroMean(arima_resids)
    garch.volatility = GARCH(p=ORDER[0],o=1,q=ORDER[2])
    garch.distribution = Normal(seed=1)
    garch_fit = garch.fit(last_obs = E1)
    # Out-sample Analytic Forecasts
    f = garch_fit.forecast(horizon = H, start=E1)

    # Conditional Volatilities
    # 1-period forward forecast conditional volatility
    fvol = pd.DataFrame({'date':f.variance['h.01'].index, 'cond_vol':f.variance['h.01'].pow(0.5)}).set_index('date')
    # sample estimated conditional volatility
    svol = pd.DataFrame({'date':garch_fit.conditional_volatility.index,'cond_vol':garch_fit.conditional_volatility}).set_index('date')
    # All conditional volatility from GARCH model
    cvol = pd.concat([svol.dropna(), fvol.dropna()])

    # Calculate Y_vol
    y_vol = pd.DataFrame(
        {'date':dataset.date.values[1:], 'y_vol':fit.resid()[1:]},
        copy=True).set_index('date').multiply(SCALE).pow(2).pow(0.5) #.multiply(SCALE)

    # Calculate GARCH residuals
    garch_resids = pd.DataFrame({'date':y_vol.index, 'garch_resids':(y_vol.y_vol.values - cvol.cond_vol.values)}).set_index('date')

    # Standardise length
    sy_vol =  y_vol[(y_vol.index>=MINIMUM)].divide(SCALE)
    scvol =  cvol[(cvol.index>=MINIMUM)].divide(SCALE)
    ssvol =  svol[(svol.index>=MINIMUM)].divide(SCALE)
    sfvol =  fvol[(fvol.index>=MINIMUM)].divide(SCALE)
    sae = ae[(ae.index>=MINIMUM)]
    sprices = prices[(prices.index>=MINIMUM)]
    # Fit auto_arima on squared error
    atrain, atest = pm.model_selection.train_test_split(sy_vol, test_size=int(test.shape[0]))
    arima2 = pm.auto_arima(atrain, seasonal=False)
    fit2 = arima2.fit(sy_vol)
    predict2 = fit2.predict_in_sample(start=1, end=sy_vol.shape[0], dynamic=False, return_conf_int=True)

    comp = pd.DataFrame({"date":sy_vol.index,"arima_resid":round(np.power(np.power(sy_vol.y_vol.values-predict2[0],2),0.5),3)}).set_index("date")

    # %%
    # Prep data for close-up of arima
    arima_results = pd.DataFrame(
        {
            'date':prices.index,
            'price':prices.val.values,
            'predicted':np.exp(predicted.predicted.values),
        }
    )
    arima_results = pd.merge(
        arima_results,
        pd.DataFrame(arima_resids.arima_resids).reset_index(),
        on='date',
        how='outer'
        )
    
    arima_results = pd.merge(
         arima_results,
         pd.DataFrame(
         np.power(np.square(arima_resids.arima_resids),0.5)
         ).reset_index(),
         on='date',
         how='outer'
    ).rename(columns={'date':'date','price':'price','predicted':'predicted','arima_resids_x':'arima_resids','arima_resids_y':'root_squared_error'})

    # data prep 
    # NOTE: not sure if I am still using this
    # unlog the logged data
    predicted['predicted'] = np.exp(predicted['predicted'])
    predicted['lower'] = np.exp(predicted['lower'])
    predicted['upper'] = np.exp(predicted['upper'])

    # %%
    # ARIMA Plots
    MIN_VAL = 1 + 2980
    MAX_VAL = MIN_VAL + test.shape[0] - 450
    # Plot actual test vs. forecasts:
    fig, ax = plt.subplots()
    ax.scatter(
        arima_results.date.values[MIN_VAL:MAX_VAL],
        arima_results.price.values[MIN_VAL:MAX_VAL],
        marker = '.',
        s = 300,
        alpha = 1,
        color = 'black',
        label = str.capitalize(arima_results.columns[1])
        )
    ax.plot(
        arima_results.date.values[MIN_VAL:MAX_VAL], 
        arima_results.predicted.values[MIN_VAL:MAX_VAL], #.iloc
        marker = '.',
        markersize = 20,
        alpha = 0.5,
        linewidth = LINEWIDTH,
        color = 'navy',
        label = str.capitalize(arima_results.columns[2])
        )
    ax2 = plt.twinx(ax)
    ax2.plot(
        arima_results.date.values[MIN_VAL:MAX_VAL], 
        arima_results.root_squared_error.values[MIN_VAL:MAX_VAL], #.iloc
        alpha = 0.5,
        linewidth = LINEWIDTH,
        color = 'darkred',
        label = str.capitalize(str.replace(arima_results.columns[4],'_',' ')),
    )

    ax.set_facecolor('whitesmoke')
    ax.fmt_xdata = DATE_FORM
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Date',fontdict=AXIS_FONT)
    ax.set_ylabel('Price',fontdict=AXIS_FONT)
    ax2.set_ylabel('Root Squared Error',fontdict=AXIS_FONT)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(visible=True, which='both', axis='both', color='white', alpha=1, linewidth=1.5)
    ax2.grid(False)
    fig.legend(loc=8, bbox_to_anchor=(0.5, -0.03), ncol=3, prop={'size':15})
    # fig.autofmt_xdate()
    fig.set_size_inches(18.5, 10.5)
    plt.title('ARIMA(1,1,1) fitted: '+str(MKT).replace('.',' '), fontdict=TITLE_FONT)

    if show_plots:
        fig.show()
    else:
        print("Not showing plot.")

    if save_figs:
        fig.savefig((d_stage1_arima+MKT+'_ARIMA'+str(ORDER)+'_full-fit_CLOSE_UP.png'), dpi=600, facecolor='white', edgecolor='white',bbox_inches = 'tight', pad_inches = 0.1)
    else:
        print('Not storing plot.')

    # %%
    # DISPERSION, ATYPICALITY & DISPERSION TRI-GRAPH
    # Some formatting
    PE = [pe.Stroke(linewidth=1.5, foreground='black'), pe.Normal()]
    MA_WINDOW = 20

    # Plot conditional volatilities
    fig, ax = plt.subplots(3)
    ax1, ax2, ax3 = ax[0], ax[1], ax[2]

    # Insert Title
    fig.suptitle('Dispersion & Atypicality Full Sample: '+str(MKT).replace('.',' '), y = 0.98, size=TITLE_FONT['size'])
    fig.subplots_adjust(hspace=0.4, top=0.90)

    # Axis 1
    # ax1.plot(sy_vol.rolling(window=MA_WINDOW).mean(), label='MA(20)', linewidth=2, alpha=1, color='violet', zorder=0)
    ax1.plot(sy_vol.index[:len(predict2[0][1:])], predict2[0][1:], label='ARIMA'+str(arima2.order)+'[RHS]', linewidth=2, alpha=1, color='violet', zorder=5)
    
    ax1.plot(scvol, label='GJR-GARCH (1, 1, 1)', linewidth=2, path_effects=PE, color='darkgreen', zorder=10)
    # ax1.plot(ssvol, label='GJR-GARCH (training)', linewidth=2, path_effects=PE, color='darkgreen', zorder=10)
    # ax1.plot(sfvol, label='GJR-GARCH (testing)', linewidth=2, path_effects=PE, color='limegreen', zorder=9)
    ax1.scatter(x=sy_vol.index, y=sy_vol.values, label='Squared ARIMA Error', s=20, alpha=0.2, color='navy', zorder=0)
    # ax1 settings
    ax1.set_title('Dispersion Estimator', fontdict=AXIS_FONT)
    ax1.axvspan(xmin=E1, xmax=E4, alpha=0.2, color='black')
    ax1.set_facecolor('whitesmoke')
    ax1.margins(x=0)
    ax1.xaxis.set_major_formatter(DATE_FORM)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1.set_xlabel('Date', fontdict=AXIS_FONT)
    ax1.set_ylabel('Squared Error', fontdict=AXIS_FONT)
    ax1.legend(prop={'size': 15}, loc='upper center')

    # Axis 2
    ax2b = plt.twinx(ax2)
    # ax2b.plot(sy_vol.rolling(window=MA_WINDOW).mean(), label='MA(20) [RHS]', linewidth=2, alpha=1, color='violet', zorder=3)
    # ax2.plot(sy_vol.index[:len(predict2[0][1:])], np.power(predict2[0][1:],0.5), label='ARIMA'+str(arima2.order)+'[LHS]', linewidth=2, alpha=1, zorder=5, color='violet')
    ax2.plot(comp, label='ARIMA'+str(arima2.order)+' Absolute Error [LHS]', linewidth=2, alpha=0.7, zorder=5, color='violet')
    # ax2.scatter(x=sy_vol.index, y=np.power(sy_vol.values,0.5), label='Root-Squared Stage 1 Error [LHS]', s=20, alpha=0.2, zorder=0, color='navy')

    ax2b.plot(sae, label='AE RMSE [RHS]', alpha=1, linewidth=2, path_effects=PE, zorder=100, color='firebrick')
    # fix order of plots
    # ax2b.set_zorder(0)
    ax2b.set_zorder(10)
    ax2b.patch.set_visible(False)

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()

    # ax2 settings
    ax2.set_facecolor('whitesmoke')
    ax2.margins(x=0)
    ax2.xaxis.set_major_formatter(DATE_FORM)
    ax2b.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.set_xlabel('Date',fontdict=AXIS_FONT)
    ax2b.set_ylabel('RMSE',fontdict=AXIS_FONT)
    ax2.set_ylabel('Root-Squared Error',fontdict=AXIS_FONT)
    ax2.axvspan(xmin=E1,xmax=E4, alpha=0.2, color='black')
    ax2.set_title('Atypicality Estimator', fontdict=AXIS_FONT)
    ax2.legend(lines + lines2, labels + labels2, prop={'size': 15}, loc='upper center')
    ax2.set_xlim(scvol.index[0], cvol.index[-1])

    # Axis 3
    ax3.plot(sprices, label=str(MKT).replace(".", " "), alpha=1, zorder=2, color="goldenrod")
    # ax3 settings
    ax3.set_facecolor('whitesmoke')
    ax3.margins(x=0)
    ax3.xaxis.set_major_formatter(DATE_FORM)
    ax3.set_xlabel('Date', fontdict=AXIS_FONT)
    ax3.set_ylabel('Price', fontdict=AXIS_FONT)
    ax3.axvspan(xmin=E1,xmax=E4, alpha=0.2, color='black')
    ax3.set_title(str(MKT).replace('.',' '), fontdict=AXIS_FONT)

    fig.set_size_inches(23.38, 15)
    plt.grid(visible=True, which='both', axis='both', color='white', alpha=1, linewidth=1.5)
    ax2b.grid(False)

    if show_plots:
        fig.show()
    elif verbose:        
            print("Not showing plot.")
        
    if save_figs:
        fig.savefig(d_stage2+MKT+' Dispersion & Atypicality Full Sample.png', dpi=600, facecolor='white', edgecolor='white', bbox_inches = 'tight', pad_inches = 0.1)
    elif verbose:
            print('Not storing plot.')
    # %%%
    # Run some autocorr tests for the PACF
    # from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    # plot_pacf(sy_vol,lags=100)
    # plot_acf(sy_vol,lags=100)

    # %%%
    # Run some autocorr tests for the PACF
    # from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    # plot_pacf(scvol,lags=100)
    # plot_acf(scvol,lags=100)
    
    # %%
    if run_ma:
        # MA test for DISPERSION LINE
            # Some formatting
        PE = [pe.Stroke(linewidth=1.5, foreground='black'), pe.Normal()]
        MA_WINDOW = 20

        # Plot conditional volatilities
        fig, ax = plt.subplots(1)
        ax1 = ax #[0], ax[1]

        # Insert Title
        fig.suptitle('Dispersion MA Full Sample test plot: '+str(MKT).replace('.',' '), y = 0.98, size=TITLE_FONT['size'])
        fig.subplots_adjust(hspace=0.4, top=0.90)

        # Axis 1
        # ax1.plot(sy_vol.rolling(window=MA_WINDOW).mean(), label='C2C MA(20)', linewidth=2, alpha=1, color='violet', zorder=0)
        # ax1.plot(scvol.rolling(window=MA_WINDOW).mean(), label='Conditional Vol MA(20)', linewidth=2, alpha=1, color='limegreen')
        # extra MAs
        ax1.plot(scvol.rolling(window=MA_WINDOW-15).mean(), label='MA(5)', linewidth=2, alpha=1)
        ax1.plot(scvol.rolling(window=MA_WINDOW-10).mean(), label='MA(10)', linewidth=2, alpha=1)
        ax1.plot(scvol.rolling(window=MA_WINDOW).mean(), label='MA(20)', linewidth=2, alpha=1, color='limegreen')
        ax1.plot(scvol.rolling(window=MA_WINDOW+40).mean(), label='MA(60)', linewidth=2, alpha=1)
        ax1.plot(scvol.rolling(window=MA_WINDOW+60).mean(), label='MA(80)', linewidth=2, alpha=1)
        ax1.plot(scvol.rolling(window=MA_WINDOW+80).mean(), label='MA(100)', linewidth=2, alpha=1)
        ax1.plot(scvol.rolling(window=MA_WINDOW+180).mean(), label='MA(200)', linewidth=2, alpha=1)

        # ax1 settings
        ax1.set_facecolor('whitesmoke')
        ax1.margins(x=0)
        ax1.xaxis.set_major_formatter(DATE_FORM)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax1.set_xlabel('Date',fontdict=AXIS_FONT)
        ax1.axvspan(xmin=E1,xmax=E4, alpha=0.2, color='black')
        ax1.set_title('Atypicality Estimator', fontdict=AXIS_FONT)
        ax1.legend(prop={'size': 15})
        ax1.set_xlim(scvol.index[0],cvol.index[-1])

        fig.set_size_inches(23.38, 16.54)
        plt.grid(visible=True, which='both', axis='both', color='white', alpha=1, linewidth=1.5)

        if show_plots:
            fig.show()
        elif verbose:        
                print("Not showing plot.")
            
        if save_figs:
            fig.savefig(d_stage2+MKT+' Dispersion MA Full Sample test plot.png', dpi=600, facecolor='white', edgecolor='white',bbox_inches = 'tight', pad_inches = 0.1)
        elif verbose:
                print('Not storing plot.')
            
    # %%%
    # Run some autocorr tests for the PACF
    # from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    # plot_pacf(sae,lags=100)
    # plot_acf(sae,lags=100)
    
    # %%
    if run_ma:
         
        # MA test for ATYPICALITY LINE
            # Some formatting
        PE = [pe.Stroke(linewidth=1.5, foreground='black'), pe.Normal()]
        MA_WINDOW = 20

        # Plot conditional volatilities
        fig, ax = plt.subplots(1)
        ax2 = ax #[0], ax[1]

        # Insert Title
        fig.suptitle('Atypicality MA Full Sample test plot: '+str(MKT).replace('.',' '), y = 0.98, size=TITLE_FONT['size'])
        fig.subplots_adjust(hspace=0.4, top=0.90)

        # Axis 2
        ax2.plot(sae, label='AE RMSE', alpha=1, zorder=2, color="indianred", linewidth =2)
        # extra MAs
        ax2.plot(sae.rolling(window=MA_WINDOW-15).mean(), label='MA(5)', linewidth=2, alpha=1)
        ax2.plot(sae.rolling(window=MA_WINDOW-10).mean(), label='MA(10)', linewidth=2, alpha=1)
        ax2.plot(sae.rolling(window=MA_WINDOW).mean(), label='MA(20)', linewidth=2, alpha=1, color='limegreen')
        ax2.plot(sae.rolling(window=MA_WINDOW+40).mean(), label='MA(60)', linewidth=2, alpha=1)
        ax2.plot(sae.rolling(window=MA_WINDOW+60).mean(), label='MA(80)', linewidth=2, alpha=1)
        ax2.plot(sae.rolling(window=MA_WINDOW+80).mean(), label='MA(100)', linewidth=2, alpha=1)
        ax2.plot(sae.rolling(window=MA_WINDOW+180).mean(), label='MA(200)', linewidth=2, alpha=1)

        # ax2 settings
        ax2.set_facecolor('whitesmoke')
        ax2.margins(x=0)
        ax2.xaxis.set_major_formatter(DATE_FORM)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax2.set_xlabel('Date',fontdict=AXIS_FONT)
        ax2.axvspan(xmin=E1,xmax=E4, alpha=0.2, color='black')
        ax2.set_title('Atypicality Estimator', fontdict=AXIS_FONT)
        ax2.legend(prop={'size': 15})
        ax2.set_xlim(scvol.index[0],cvol.index[-1])

        fig.set_size_inches(23.38, 16.54)
        plt.grid(visible=True, which='both', axis='both', color='white', alpha=1, linewidth=1.5)

        if show_plots:
            fig.show()
        elif verbose:        
                print("Not showing plot.")
            
        if save_figs:
            fig.savefig(d_stage2+MKT+' Atypicality MA Full Sample test plot.png', dpi=600, facecolor='white', edgecolor='white',bbox_inches = 'tight', pad_inches = 0.1)
        elif verbose:
                print('Not storing plot.')

    # %%
    # Some formatting
    # Plot conditional volatilities
    fig, ax = plt.subplots(2)
    ax1, ax2, ax3 = ax[0], ax[0], ax[1]

    # Insert Title
    fig.suptitle('Dispersion & Atypicality Full Sample: '+str(MKT).replace('.',' '), y = 0.98, size=TITLE_FONT['size'])
    fig.subplots_adjust(hspace=0.4, top=0.90)

    # Axis 1
    ax1.plot(scvol, label='GJR-GARCH (LHS)', linewidth=1, zorder=9, color='darkgreen') 
    
    # ax1 settings
    ax1.set_title('Dispersion & Atypicality Estimators', fontdict=AXIS_FONT)
    ax1.axvspan(xmin=E1, xmax=E4, alpha=0.2, color='black')
    ax1.set_facecolor('whitesmoke')
    ax1.margins(x=0)
    ax1.xaxis.set_major_formatter(DATE_FORM)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.set_xlabel('Date',fontdict=AXIS_FONT)
    ax1.set_ylabel('Squared Error',fontdict=AXIS_FONT)
    ax1.grid(False)

    # Axis 2
    ax2 = ax1.twinx()
    ax2.plot(ae, label='AE RMSE (RHS)', alpha=1, zorder=2, color="firebrick")

    # ax2 settings
    ax2.set_facecolor('whitesmoke')
    ax2.margins(x=0)
    ax2.xaxis.set_major_formatter(DATE_FORM)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.set_xlabel('Date', fontdict=AXIS_FONT)
    ax2.axvspan(xmin=E1, xmax=E4, alpha=0.2, color='black')
    ax2.set_ylabel('Root-Squared Error', fontdict=AXIS_FONT)
    ax2.grid(False)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, prop={'size': 15}, loc='upper center')

    # Axis 3
    ax3.plot(prices, label=str(MKT).replace(".", " "), alpha=1, zorder=2, color="goldenrod")
    # ax3 settings
    ax3.set_facecolor('whitesmoke')
    ax3.margins(x=0)
    ax3.xaxis.set_major_formatter(DATE_FORM)
    # ax3.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax3.set_xlabel('Date',fontdict=AXIS_FONT)
    ax3.axvspan(xmin=E1,xmax=E4, alpha=0.2, color='black')
    ax3.set_title(str(MKT).replace('.',' '), fontdict=AXIS_FONT)
    ax3.set_ylabel('Price', fontdict=AXIS_FONT)
    fig.set_size_inches(23.38, 16.54)


    if show_plots:
        fig.show()
    elif verbose:        
            print("Not showing plot.")
        
    if save_figs:
        fig.savefig(d_stage2_merged+MKT+' Dispersion & Atypicality merged.png', dpi=600, facecolor='white', edgecolor='white',bbox_inches = 'tight', pad_inches = 0.1)
    elif verbose:
            print('Not storing plot.')
    
    # %%
    # NOTE: I HAVE NOT MODIFIED THESE TO HAVE
    # a. The auto arima
    # b. The right y axis titles
    # CLOSE UP PLOT ####
    # Some formatting
    PE = [pe.Stroke(linewidth=3, foreground='black'), pe.Normal()]

    # Slices
    LOWER = '2019-01-01'
    UPPER = '2021-01-01'
    
    ae_slice = ae[(ae.index>LOWER)&(ae.index<UPPER)]
    cvol_slice = cvol[(cvol.index>LOWER)&(cvol.index<UPPER)]
    price_slice = prices[(prices.index>LOWER)&(prices.index<UPPER)]
    vol_slice = y_vol[(y_vol.index>LOWER)&(y_vol.index<UPPER)]

    # Plot conditional volatilities
    fig, ax = plt.subplots(3)
    ax1, ax2, ax3 = ax[0], ax[1], ax[2]

    # Insert Title
    fig.suptitle('Dispersion & Atypicality Close Up: '+str(MKT).replace('.',' '), y = 0.98, size=TITLE_FONT['size']) #fontsize=TITLE_FONT['size']
    fig.subplots_adjust(hspace=0.4, top=0.90)

    # Axis 1
    ax1.plot(y_vol.rolling(window=40).mean()[(y_vol.index>LOWER)&(y_vol.index<UPPER)], label='C2C MA(40)', linewidth=2, alpha=1, color='violet')
    ax1.plot(cvol.rolling(window=20).mean()[(cvol.index>LOWER)&(cvol.index<UPPER)], label='Conditional Vol MA(20)', linewidth=2, alpha=1, color='limegreen')

    ax1.plot(svol[(svol.index>LOWER)&(svol.index<UPPER)], label='Train Conditional Vol', linewidth=1, path_effects = PE)
    ax1.plot(fvol[(fvol.index>LOWER)&(fvol.index<UPPER)], label='Forecast Conditional Vol', linewidth=1, path_effects = PE)
    ax1.scatter(x=vol_slice.index, y=vol_slice.values, label='Actual Vol estimate', s=20, alpha=0.5, zorder=1)
    # Vertical lines
    ax1.vlines(E1, ymin=vol_slice.min()*0.9, ymax=vol_slice.max()*1.05, color='black', linewidth=1)
    ax1.vlines(E2, ymin=vol_slice.min()*0.9, ymax=vol_slice.max()*1.05, color='black', linewidth=1)
    ax1.vlines(E3, ymin=vol_slice.min()*0.9, ymax=vol_slice.max()*1.05, color='black', linewidth=1)
    ax1.vlines(E4, ymin=vol_slice.min()*0.9, ymax=vol_slice.max()*1.05, color='black', linewidth=1)
    ax1.text(E1, vol_slice.max()*1.05, 'E1', horizontalalignment='right', verticalalignment='top', fontsize=15, fontweight='semibold')
    ax1.text(E2, vol_slice.max()*1.05, 'E2', horizontalalignment='right', verticalalignment='top', fontsize=15, fontweight='semibold')
    ax1.text(E3, vol_slice.max()*1.05, 'E3', horizontalalignment='right', verticalalignment='top', fontsize=15, fontweight='semibold')
    ax1.text(E4, vol_slice.max()*1.05, 'E4', horizontalalignment='right', verticalalignment='top', fontsize=15, fontweight='semibold')
    # ax1 settings
    ax1.set_title('Dispersion Estimator', fontdict=AXIS_FONT)
    ax1.axvspan(xmin=E1,xmax=E4, alpha=0.2, color='black')
    ax1.set_facecolor('whitesmoke')
    ax1.margins(x=0)
    ax1.xaxis.set_major_formatter(DATE_FORM)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1.set_xlabel('Date',fontdict=AXIS_FONT)
    ax1.legend( prop={'size': 15})

    # Axis 2
    ax2.plot(ae_slice, label='AE RMSE', alpha=1, zorder=10, color="indianred")
    ax2.plot(ae.rolling(window=20).mean()[(ae.index>LOWER)&(ae.index<UPPER)], label='RMSE-MA(20)', alpha=1, zorder=0, color="dodgerblue")
    # Vertical lines
    ax2.vlines(E1, ymin=ae_slice.min()*0.001, ymax=ae_slice.max()*1.1, color='black', linewidth=1)
    ax2.vlines(E2, ymin=ae_slice.min()*0.001, ymax=ae_slice.max()*1.1, color='black', linewidth=1)
    ax2.vlines(E3, ymin=ae_slice.min()*0.001, ymax=ae_slice.max()*1.1, color='black', linewidth=1)
    ax2.vlines(E4, ymin=ae_slice.min()*0.001, ymax=ae_slice.max()*1.1, color='black', linewidth=1)
    ax2.text(E1, ae_slice.max()*1.1, 'E1', horizontalalignment='right', verticalalignment='top', fontsize=15, fontweight='semibold')
    ax2.text(E2, ae_slice.max()*1.1, 'E2', horizontalalignment='right', verticalalignment='top', fontsize=15, fontweight='semibold')
    ax2.text(E3, ae_slice.max()*1.1, 'E3', horizontalalignment='right', verticalalignment='top', fontsize=15, fontweight='semibold')
    ax2.text(E4, ae_slice.max()*1.1, 'E4', horizontalalignment='right', verticalalignment='top', fontsize=15, fontweight='semibold')
    # ax2 settings
    ax2.set_facecolor('whitesmoke')
    ax2.margins(x=0)
    ax2.xaxis.set_major_formatter(DATE_FORM)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.set_xlabel('Date',fontdict=AXIS_FONT)
    ax2.axvspan(xmin=E1,xmax=E4, alpha=0.2, color='black')
    ax2.set_title('Atypicality Estimator', fontdict=AXIS_FONT)
    ax2.legend(prop={'size': 15})

    # Axis 3
    ax3.plot(price_slice, label=str(MKT).replace(".", " "), alpha=1, zorder=2, color="goldenrod")
    # ax3 settings
    ax3.set_facecolor('whitesmoke')
    ax3.margins(x=0)
    ax3.xaxis.set_major_formatter(DATE_FORM)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax3.set_xlabel('Date',fontdict=AXIS_FONT)
    ax3.axvspan(xmin=E1,xmax=E4, alpha=0.2, color='black')
    ax3.set_title(str(MKT).replace('.',' '), fontdict=AXIS_FONT)
    # Vertical lines
    ax3.vlines(E1, ymin=price_slice.min()*0.9, ymax=price_slice.max()*1.1, color='black', linewidth=1)
    ax3.vlines(E2, ymin=price_slice.min()*0.9, ymax=price_slice.max()*1.1, color='black', linewidth=1)
    ax3.vlines(E3, ymin=price_slice.min()*0.9, ymax=price_slice.max()*1.1, color='black', linewidth=1)
    ax3.vlines(E4, ymin=price_slice.min()*0.9, ymax=price_slice.max()*1.1, color='black', linewidth=1)
    ax3.text(E1, price_slice.max()*1.1, 'E1', horizontalalignment='right', verticalalignment='top', fontsize=15, fontweight='semibold')
    ax3.text(E2, price_slice.max()*1.1, 'E2', horizontalalignment='right', verticalalignment='top', fontsize=15, fontweight='semibold')
    ax3.text(E3, price_slice.max()*1.1, 'E3', horizontalalignment='right', verticalalignment='top', fontsize=15, fontweight='semibold')
    ax3.text(E4, price_slice.max()*1.1, 'E4', horizontalalignment='right', verticalalignment='top', fontsize=15, fontweight='semibold')

    ax3.plot(
        predicted[['upper','lower']][(predicted[['upper','lower']].index>LOWER)&(predicted[['upper','lower']].index<UPPER)],
        alpha = 0.5,
        linewidth = LINEWIDTH,
        label = ['UpperCI','LowerCi']
        )
    ax3.legend( prop={'size': 15})

    fig.set_size_inches(23.38, 16.54)
    plt.grid(visible=True, which='both', axis='both', color='white', alpha=1, linewidth=1.5)

    if show_plots:
        fig.show()
    elif verbose:        
            print("Not showing plot.")
        
    if save_figs:
        fig.savefig(d_stage2_cu+MKT+' Dispersion & Atypicality Close Up.png', dpi=600, facecolor='white', edgecolor='white',bbox_inches = 'tight', pad_inches = 0.1)
    elif verbose:
            print('Not storing plot.')

    with open(d_complete+'ACP completed indices'+'.txt', 'a') as f:
        f.write('\n'+MKT+' finished.')

    # %%
    # DATA PREP FOR CLOSE UP PLOT
    # constants
    UPPER = '2021-01-01'
    LOWER = '2019-01-01'
    EVENT_LIST = [E1, E2, E3, E4]
    PE = [pe.Stroke(linewidth=1.5, foreground='black'), pe.Normal()]

    # select slices to plot
    price_slice = pd.DataFrame({"date":arima_results.date[(arima_results.date>LOWER)&(arima_results.date<UPPER)],
                                "price":arima_results.price[(arima_results.date>LOWER)&(arima_results.date<UPPER)]}
                                ).set_index('date')
    x_slice = arima_results.date[(arima_results.date>LOWER)&(arima_results.date<UPPER)]
    sy_slice = sy_vol[(sy_vol.index>LOWER)&(sy_vol.index<UPPER)]
    sae_slice = sae[(sae.index>LOWER)&(sae.index<UPPER)]
    disp_slice = scvol[(scvol.index>LOWER)&(scvol.index<UPPER)]
    arima_slice = arima_results.predicted[(arima_results.date>LOWER)&(arima_results.date<UPPER)]

    comp_model = pd.DataFrame({"date":arima_results.date,"squared_error":predict2[0],"root_squared_error":np.power(predict2[0],0.5)})
    comp_model = comp_model[(comp_model.date>LOWER)&(comp_model.date<UPPER)]

    comp_slice = comp[(comp.index>LOWER)&(comp.index<UPPER)]
    
    # %%
    # CLOSE UP
    # generate axes
    fig, axs = plt.subplots(3, sharex=True, constrained_layout=True)
    ax1, ax2, ax3 = axs[0], axs[1], axs[2]
    ax2b = ax2.twinx()

    # Insert heading
    # fig.suptitle('Covid Period: '+str(MKT).replace('.',' '), y = 0.9, fontdict=TITLE_FONT)    
    # fig.subplots_adjust(hspace=0.4, top=0.85)
    
    # Axis 1
    ax1.plot(disp_slice, alpha=1, label='GJR-GARCH(1, 1, 1)', linewidth=2, zorder=12)
    ax1.scatter(x=sy_slice.index, y=sy_slice.values, label='Squared Stage 1 Error', s=20, alpha=0.2, color='navy', zorder=0)
    ax1.plot(comp_model.date, comp_model.squared_error, label='ARIMA'+str(arima2.order), linewidth=2, alpha=1, color='violet', zorder=5)
    # Vertical lines
    i = 1
    for e in EVENT_LIST:
        ax1.vlines(e, ymin=0, ymax=sy_slice.max(), color='black', linewidth=1, zorder=9)
        ax1.text(e, sy_slice.max(), 'E'+str(i), horizontalalignment='right', verticalalignment='top', fontsize=10, fontweight='semibold')
        i += 1
        
    # Settings
    # Subtitle
    # ax1.set_title('Dispersion Estimator', fontdict=AXIS_FONT)
    ax1.axvspan(xmin=E1, xmax=E4, alpha=0.2, color='black')
    ax1.set_facecolor('whitesmoke')
    ax1.margins(x=0)
    ax1.xaxis.set_major_formatter(DATE_FORM)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax1.set_xlabel('Date', fontdict=AXIS_FONT)
    ax1.set_ylabel('Squared Error', fontdict=AXIS_FONT)
    ax1.legend(prop={'size': 15}, loc='upper left', frameon=True, framealpha=0.5, shadow=False, borderpad=1)

    # Axis 2
    # ax2.plot(comp_model.date, comp_model.root_squared_error, label='ARIMA'+str(arima2.order)+'[RHS]', linewidth=2, alpha=0.7, color='violet', zorder=5)
    # ax2.scatter(x=sy_slice.index, y=np.power(sy_slice.values,0.5), label='Root-Squared Stage 1 Error [LHS]', s=20, alpha=0.2, color='navy', zorder=0)
    ax2.plot(comp_slice, label='ARIMA'+str(arima2.order)+' Absolute Error [LHS]', linewidth=2, alpha=1, zorder=5, color='violet')  
    ax2b.plot(sae_slice, label='AE RMSE [RHS]', alpha=1, linewidth=2, path_effects=PE, zorder=20, color='firebrick')

    ax2b.set_zorder(10)
    ax2b.patch.set_visible(False)

    # Vertical lines
    i = 1
    for e in EVENT_LIST:
        # ax2.vlines(e, ymin=0, ymax=np.power(sy_slice.values,0.5).max(), color='black', linewidth=1)
        # ax2.text(e, np.power(sy_slice.values,0.5).max(), 'E'+str(i), horizontalalignment='right', verticalalignment='top', fontsize=10, fontweight='semibold')

        ax2.vlines(e, ymin=0, ymax=comp_slice.max(), color='black', linewidth=1, zorder=9)
        ax2.text(e, comp_slice.max(), 'E'+str(i), horizontalalignment='right', verticalalignment='top', fontsize=10, fontweight='semibold')

        i += 1

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()

    # Settings
    ax2.set_facecolor('whitesmoke')
    ax2.margins(x=0)
    ax2.xaxis.set_major_formatter(DATE_FORM)
    ax2b.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax2.set_xlabel('Date', fontdict=AXIS_FONT)
    ax2b.set_ylabel('RMSE', fontdict=AXIS_FONT)
    ax2b.set_ylim(ymax=float(round(sae_slice.max(),3)))
    ax2.set_ylabel('Root-Squared Error', fontdict=AXIS_FONT)
    ax2.axvspan(xmin=E1, xmax=E4, alpha=0.2, color='black')
    ax2.set_title('Atypicality Estimator', fontdict=AXIS_FONT)
    ax2b.legend(lines2 + lines, labels2 + labels, prop={'size': 15},
                loc='upper left', frameon=True, framealpha=0.5, shadow=False, borderpad=1
                ).set_zorder(100)
    ax2.set_xlim(sy_slice.index[0], sy_slice.index[-1])
    # align the two y-axes
    ax2b.set_yticks(np.linspace(ax2b.get_yticks()[0], ax2b.get_yticks()[-1], len(ax2.get_yticks())))

    # Axis 3
    ax3.plot(price_slice, label=str(MKT).replace(".", " "), alpha=1, zorder=2, color="goldenrod")
    ax3.axvspan(xmin=E1, xmax=E4, alpha=0.1, color='black')
    # Vertical lines
    i = 1
    for e in EVENT_LIST:
        ax3.vlines(e, ymin=0, ymax=price_slice.max(), color='black', linewidth=1)
        ax3.text(e, price_slice.max(), 'E'+str(i), horizontalalignment='right', verticalalignment='top', fontsize=10, fontweight='semibold')
        i += 1
        
    # Settings
    ax3.set_title(str(MKT).replace('.',' ')+' Price Level', fontdict=AXIS_FONT)
    ax3.set_ylim(ymin=price_slice.min()[0])
    ax3.set_facecolor('whitesmoke')
    ax3.xaxis.set_major_formatter(DATE_FORM)
    # ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.set_xlabel('Date', fontdict=AXIS_FONT)
    ax3.set_ylabel('Price', fontdict=AXIS_FONT)
    # ax3.legend()

    # General Fig settings
    fig.set_size_inches(15, 12)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.grid(visible=True, which='both', axis='both', color='white', alpha=1, linewidth=1.5)
    ax2b.grid(False)

    if show_plots:
        plt.show()
    else:
        if verbose:
            print("Not showing plot.")
        
    if save_figs:
        fig.savefig(d_stage2_cu+MKT+'_covid-period__close-up.png',
                    dpi=600, facecolor='white', edgecolor='white', bbox_inches='tight', pad_inches=0.1)
    else:
        if verbose:
            print('Not storing plot.')


# %%
ps(1)