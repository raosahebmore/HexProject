# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:39:08 2018

@author: avi
"""

#import pandas_datareader.data as web
#import matplotlib.pyplot as plt
from datetime import datetime
from nsepy import get_history
import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series
from pandas import concat
from pandas import Series
from pandas import DataFrame
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import seaborn as sns

print("Below are the list of methods which we can do the stock analysis : ")
print("      0: RSI Method  ")
print("      1: Moving Average Convergence Divergence  ")
print("      2: Moving average for 14 days  ")
print("      3: Bollinger band Method  ")
print("      4: Fast Stochastic Method  ")
print("      5: Slow Stochastic Method  ")
print("      6: Rate of change (ROC)  ")
print("      7: Williams R value  ")
print("      8: Commodity Channel Index ")
print("      9: Force Index")
#print("      10: Average Directional Movement Index")
#print("      11: Autoregression Time Series prediction of stock :")
print("      10: Exit")

var =input("Enter the value(0,1,2,3,4,5...) which you want perform the analysis on the stock :")
if(var =="12"):
        print("Exit")
        print("Thank Y")
        clear = lambda: os.system('cls')
        clear()
else:
    
    start =datetime.datetime.strptime(input("Enter the stock price Start date in DD-MON-YYYY e.g 1-jan-2018 :"),"%d-%b-%Y")
    '''inputstartDate = input("Enter the stock price Start date in format DD-MON-YYYY : ")
    day,month,year = inputstartDate.split('-')
    isValidDate = True
    try :
        datetime.datetime(int(day),int(month),int(year))
    except ValueError :
        isValidDate = False
    if(isValidDate) :
        start=datetime.datetime.strptime(inputstartDate)
        print ("Input date is valid ..")
    else :
        print ("Input date is not valid..")
        clear = lambda: os.system('cls')
        clear()'''
    end =datetime.datetime.strptime(input("Enter the stock price End date in DD-MON-YYYY 10-jun-2018:"),"%d-%b-%Y")
    
    stock =input("Enter the stock code e.g.: Tata Motors : TATAMOT   :") 
    ext_stock = get_history(stock,start,end)
def callme():
    print("Below are the list of methods which we can do the stock analysis : ")
    print("      0: RSI Method  ")
    print("      1: Moving Average Convergence Divergence  ")
    print("      2: Moving average for 14/21 days  ")
    print("      3: Bollinger band Method  ")
    print("      4: Fast Stochastic Method  ")
    print("      5: Slow Stochastic Method  ")
    print("      6: Rate of change (ROC)  ")
    print("      7: Williams R value  ")
    print("      8: Commodity Channel Index ")
    print("      9: Force Index")
 #   print("      10: Autoregression Time Series prediction of stock :")
    print("      10: Exit")
    var =input("Enter the value(0,1,2,3,4,5...) which you want perform the analysis on the stock :")
 #    print("      11: Average Directional Movement Index")     
    if(var =="12"):
        print("Exit")
        clear = lambda: os.system('cls')
        clear()
    else:
        if var == "0":
           print( "RSI smart process evaluation started ..... ")
           rsi=RSI_method(ext_stock,14)
           print(rsi)
           callme()
        elif var == "1":
           print( " Moing average convergence divergence smart process evaluation started ..... ")
           MACD=Moving_avg_con_div(ext_stock,9,26)
           print(MACD)
           callme()
           
        elif var == "2":
           print ("Moving average for 14 days evaluation started ..... ")
           ma=Moving_Avg_MAPE(ext_stock,14)
           print(ma)
           callme()
           
        elif var == "3":
           print ("Bollinger Method evaluation !")
           bb=Bollinger_band(ext_stock,14)
           print(bb)
           callme()
           
        elif var =="4":
           print ("Fast Stochastic Method evaluation !")
           fs=fast_stochastic(ext_stock,14,3)
           print(fs) 
           callme()
           
        elif var =="5":
           print ("Slow Stochastic Method evaluation !")
           print(ss) 
           callme()
           
        elif var =="6":
           print ("Rate of change evaluation !")
           roc=ROC(ext_stock,8)
           print(roc) 
           callme()
           
        elif var =="7":
           print ("William Average evaluation !")
           william=williams_r(ext_stock,14)
           print(william)
           callme()
           
        elif var =="8":
           print ("Commodity Channel Index")
           cciv=CCI(ext_stock,20)
           print(cciv)
           callme()
           
        elif var =="9":
           print ("Force Index")
           Forcei=Force_Index(ext_stock,1)
           print(Forcei)
           callme()
        elif var=="10":
            print("Thank You")
        else:
           print ("Select the aprroriate method type : ")
"""        elif var =="10":
           print ("ADX")
           adx=AR_X(ext_stock)
           print(adx)
           callme()          
"""           


"""     elif var =="11":
           print ("ADX")
           adx=AD_X(ext_stock,14,14)
           print(adx)
           callme()  """           

#var =input("Enter the value(0,1,2,3,4,5...) which you want perform the analysis on the stock :")

# Enter the start date and end along with stock code
#start =datetime.datetime.strptime(input("Enter the stock price Start date in DD-MON-YYYY :"),"%d-%b-%Y")
#end =datetime.datetime.strptime(input("Enter the stock price End date in DD-MON-YYYY :"),"%d-%b-%Y")
#stock =input("Enter the stock code e.g.: Tata Motors : TATAMOT   :")
# Extrating data
#ext_stock = get_history(stock,start,end)
#print(ext_stock[['Close']])

# Rate of Change (ROC)
def ROC(df,n):
     N = df['Close'].diff(n)
     D = df['Close'].shift(n)
     df['ROC'] = pd.Series(N/D,name='Rate of Change')
     #df = df.join(ROC)
     df.plot(y=['Close'])
     plt.xticks(rotation='vertical')
     plt.show()
     #df.plot(y=['ROC'])
     f1, ax11 = plt.subplots(figsize = (8,4))
     #ax11.plot(df.index,df['Close'], color = 'blue', lw=2, label='Close')
     ax11.plot(df.index,df['ROC'], color ='red', lw=1, label='ROC')
     ax11.set(title = 'Rate of Change', ylabel = 'Price',xlabel='Date')
     ax11.legend(loc='upper right')
     plt.xticks(rotation='vertical')
     plt.show()
     return df 

def RSI_method(df, n):
    """Calculate Relative Strength Index(RSI) for given data.
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    diff=[]
    gain=[]
    loss=[]
    mav_gain=[]
    mav=[]
    df1 = pd.DataFrame(df)
    print("Close  High")
    i=0
    prev_close_price=0
    diff_close=0
    for index,row in df1.iterrows() :
        if i>0:
            diff_close=row['Close']-prev_close_price
            diff.append(diff_close)
        else:
            diff.append(0)            
     #   print(row['Close'],row['High'])
        i=i+1
        if (diff_close>=1):
            gain_val=diff_close
        else:
            gain_val=0
            
        if(diff_close<1):
            loss_val=abs(diff_close)
        else:
            loss_val=0
        gain.append(gain_val)
        loss.append(loss_val)
        prev_close_price=row['Close']
    df1['diff'] = diff
    df1['gain'] = gain
    df1['loss'] = loss
    df1['MAV_gain'] = df1['gain'].rolling(n).mean()
    df1['MAV_loss'] = df1['loss'].rolling(n).mean()
    df1['MAV_gain_d_loss'] = df1['MAV_gain']/df1['MAV_loss']
    df1['RSI'] =100-(100/(1+df1['MAV_gain_d_loss']))
    df2=pd.DataFrame(df1,columns=['Date','Symbol','Close','diff','gain','loss','MAV_gain','MA_loss','MAV_gain_d_loss','RSI'])
    df2["Signal"] =np.where(((df2["RSI"]>1) & (df2["RSI"]<=30)),'BUY',np.where(((df2["RSI"]>30) & (df2["RSI"]<=60)),'Neutral',np.where(np.isnan(df2["RSI"]),'NA','Sell')))
    print(df2)
    groupdata=(df2.groupby(["Signal"]).size())  
    newgdata=pd.DataFrame(groupdata,columns=["count"])
    print(newgdata["count"])
    labels = 'BUY', 'NA', 'Neutral', 'Sell'
    sizes = newgdata.values
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
   # explode = (0.1, 0, 0,0)  # explode 1st slice
 
# Plot
  #  plt.pie(sizes, explode=None, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=160)
  #  plt.axis('equal')
    f1, ax11 = plt.subplots(figsize = (8,4))
    ax11.plot(df.index, df['Close'], color = 'black', lw=2, label='Close')
    ax11.plot(df.index, df1['RSI'], color ='red', lw=1, label='RSI')
    ax11.set(title = 'Relative Strength Index', ylabel = 'Price',xlabel='Date')
    ax11.legend(loc='upper right')
    plt.xticks(rotation='vertical')
    plt.show()
    return df2

def Moving_avg_con_div(df, n_fast, n_slow):
    print("Moving avg con div!")
    """Calculate MACD, MACD Signal and MACD difference
    
    :param df: pandas.DataFrame
    :param n_fast: 
    :param n_slow: 
    :return: pandas.DataFrame
    """
    EMAfast = pd.Series(df['Close'].ewm(span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df['Close'].ewm(span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
   # df = df.join(EMAfast)
    #df = df.join(EMAslow)
    # set up MACD parameters
    #slow_period = 26
    #fast_period = 12
    #signal_period = 9
    # compute the MACD datapoints
    #emaslow, emafast, macd_line = macd(df['Close'], slow_period, fast_period)
    #signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    # draw the MACD lines and histogram
    f1, ax1 = plt.subplots(figsize = (8,4))
    ax1.plot(df.index, df['Close'], color = 'black', lw=2, label='Close')
    ax1.plot(df.index, EMAslow, color ='blue', lw=1, label='EMA(26)')
    ax1.plot(df.index, EMAfast, color ='red', lw=1, label='EMA(12)')
    ax1.legend(loc='upper right')
    ax1.set(title = 'Stock Price', ylabel = 'Price',xlabel='Date')
    plt.xticks(rotation='vertical')
    plt.show()    
    
    f2, ax2 = plt.subplots(figsize = (8,4))
    ax2.plot(df.index, MACD, color='red', lw=1,label='MACD Line(26,12)')
    ax2.plot(df.index, MACDsign, color='purple', lw=1, label='Signal Line(9)')
    ax2.fill_between(df.index, MACD - MACDsign, color = 'gray', alpha=0.5, label='MACD Histogram')
#    ax2.fill_between(df.index, MACD - MACDsign, color = 'gray', alpha=0.5, label='MACD Histogram')
    ax2.set(title = 'MACD(26,12,9)', ylabel='MACD')
    ax2.legend(loc = 'upper right')
    ax2.grid(False)
    plt.xticks(rotation='vertical')
    plt.show()
    return df
    
def Moving_Avg_MAPE(df,n):
    """Calculate the moving average for the given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    df["MA"] = pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
    df["MA_7"] = pd.Series(df['Close'].rolling(n+7, min_periods=n+7).mean(), name='MA_' + str(n+7))    
    #df = df.join(MA)
    #df.plot(y=['Close'])
    #df.plot(y=['MA'])
    f1, ax12 = plt.subplots(figsize = (8,4))
    ax12.plot(df.index, df['Close'], color = 'black', lw=2, label='Close')
    ax12.plot(df.index, df['MA'], color ='red', lw=1, label='MA_14')
    ax12.plot(df.index, df['MA_7'], color ='red', lw=1, label='MA_21')
    ax12.set(title = 'Moving Average ', ylabel = 'Price',xlabel='Date')
    ax12.legend(loc='upper right')
    plt.xticks(rotation='vertical')
    plt.show()
    return df

def Bollinger_band(df,n):
    print("Bollinger Band  Method Analysis!")
    """    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean())
    MSD = pd.Series(df['Close'].rolling(n, min_periods=n).std())
    df['Mean'] = MA
    df['B1'] = MA + (MSD * 2)
    df['B2'] = MA - (MSD * 2)
    """
    b1 = 4 * MSD / MA
    df['B1'] = pd.Series(b1, name='BollingerB_' + str(n))
    #df = df.join(B1)
    b2 = (df['Close'] - MA + (2 * MSD)) / (4 * MSD)
    df['B2'] = pd.Series(b2, name='Bollinger%b_' + str(n))
    #df = df.join(B2)
    """
    f1, ax11 = plt.subplots(figsize = (8,4))
    ax11.plot(df.index, df['Close'], color = 'black', lw=2, label='Close')
    ax11.plot(df.index, df['B1'], color ='red', lw=1, label='High')
    ax11.plot(df.index, df['B2'], color ='red', lw=3, label='Low')
    ax11.set(title = 'Bollinger Band  Method', ylabel = 'Price',xlabel='Date')
    ax11.legend(loc='upper right')
    plt.xticks(rotation='vertical')
    plt.show()
    return df

def fast_stochastic(df, period=14, smoothing=3):
    """ calculate slow stochastic
    Fast stochastic calculation
    %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K
    """
    df1 = pd.DataFrame(df)
    low_min =  df1['Low'].rolling(period).min()
    high_max = df1['Low'].rolling(period).max()
    df1['k_fast'] = 100 * (df1['Close'] - low_min)/(high_max - low_min)
   # df1['k_fast'] = df1['k_fast'].dropna()
    df1['d_fast'] = df1['k_fast'].rolling(smoothing).mean()
    #print(df1)
    #df.plot(y=['Close'])
    #df.plot(y=['d_fast'])
    f1, ax11 = plt.subplots(figsize = (8,4))
    ax11.plot(df.index, df['Close'], color = 'black', lw=2, label='Close')
    ax11.plot(df.index, df1['d_fast'], color ='red', lw=1, label='Fast Stochastic')
    ax11.set(title = 'Fast stochastic calculation', ylabel = 'Price',xlabel='Date')
    ax11.legend(loc='upper right')
    plt.xticks(rotation='vertical')
    plt.show()
    return df1



def slow_stochastic(df, period=14, smoothing=3):
    """ calculate slow stochastic
    Slow stochastic calculation
    %K = %D of fast stochastic
    %D = 3-day SMA of %K
    """
    df1 = pd.DataFrame(df)
    low_min =  df1['Low'].rolling(period).min()
    high_max = df1['Low'].rolling(period).max()
    df1['d_fast'] = 100 * (df1['Close'] - low_min)/(high_max - low_min)
   # df1['k_fast'] = df1['k_fast'].dropna()
    df1['k_fast'] = df1['d_fast'].rolling(smoothing).mean()
    #print(df1)
    #df.plot(y=['Close'])
    #df.plot(y=['k_fast'])
    f1, ax_ss = plt.subplots(figsize = (8,4))
    ax_ss.plot(df.index, df['Close'], color = 'black', lw=2, label='Close')
    ax_ss.plot(df.index, df1['k_fast'], color ='red', lw=1, label='Slow Stochastic')
    ax_ss.set(title = 'Slow stochastic calculation', ylabel = 'Price',xlabel='Date')
    ax_ss.legend(loc='upper right')
    plt.xticks(rotation='vertical')
    plt.show()
    return df1


def williams_r(data, periods):
    data['williams_r'] = 0.
    for (line_number, (index, row)) in enumerate(data.iterrows()):
        print(line_number,row)
        if line_number > periods:
            data.set_value(line_number, 'williams_r', ((max(data['High'][line_number-periods:line_number]) - row['Close']) /  (max(data['High'][line_number-periods:line_number]) - min(data['Low'][line_number-periods:line_number]))))        
    data.plot(y=['High'])
    data.plot(y=['williams_r'])
    plt.xticks(rotation='vertical')
   # f1, ax11 = plt.subplots(figsize = (8,4))
    #ax11.plot(data.index,df1['High'], color = 'black', lw=2, label='High')
    #ax11.plot(data.index,df1['williams_r'], color ='red', lw=1, label='williams_r')
    #ax11.set(title = 'Williams R', ylabel = 'Price',xlabel='Date')
    #ax11.legend(loc='upper right')
    #plt.xticks(rotation='vertical')
    plt.show()
    return data
# Commodity Channel Index 
'''def CCI(data, ndays): 
    TP1 = (data['High'] + data['Low'] + data['Close']) / 3 
    TP = int(float(TP1))
    print(TP)
    nma=pd.Series(data['Close'].rolling(TP,ndays).mean())
    nsd=pd.Series(data['Close'].rolling(TP,ndays).std())
    CCI = pd.Series((TP - nma) / (0.015 * nsd),name = 'CCI') 
    data = data.join(CCI) 
    return data'''
    
# Commodity Channel Index 
def CCI(df, n):
    """Calculate Commodity Channel Index for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    df['CCI'] = pd.Series((PP - PP.rolling(n, min_periods=n).mean()) / PP.rolling(n, min_periods=n).std(),
                    name='CCI_' + str(n))
    #df = df.join(CCI)
    df.plot(y=['High'])
    df.plot(y=['CCI'])
    #f1, ax12 = plt.subplots(figsize = (8,4))
    #ax12.plot(df.index, df['High'], color = 'black', lw=2, label='High')
    #ax12.plot(df.index, df['CCI'], color ='red', lw=1, label='CCI')
    #ax12.set(title = 'Commodity Channel Index ', ylabel = 'Price',xlabel='Date')
    #ax12.legend(loc='upper right')
    plt.xticks(rotation='vertical')
    plt.show()
    return df

    
# Force Index 
def Force_Index(data, ndays): 
    data['FI'] = pd.Series(data['Close'].diff(ndays) * data['Volume'], name = 'ForceIndex') 
    #data = data.join(FI) 
    data.plot(y=['Close'])
    data.plot(y=['FI'])
    #f1, ax11 = plt.subplots(figsize = (8,4))
    #ax11.plot(data.index,data['Close'], color = 'blue', lw=2, label='Close')
    #ax11.plot(data.index,data['FI'], color ='red', lw=1, label='FI')
    #ax11.set(title = 'Force Index', ylabel = 'Price',xlabel='Date')
    #ax11.legend(loc='upper right')
    plt.xticks(rotation='vertical')
    plt.show()
    return data

def AD_X(df1, n, n_ADX):
    """Calculate the Average Directional Movement Index for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :param n_ADX: 
    :return: pandas.DataFrame
    """
    df = pd.DataFrame(df1)
    i = 1
    UpI = []
    DoI = []
    while i + 1 <= len(df):
        UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
        DoMove = df.at[i, 'Low'] - df.at[i + 1, 'Low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)

        print(DoMove)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    i = 1
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.at[i + 1, 'High'], df.at[i, 'Close']) - min(df.at[i + 1, 'Low'], df.at[i, 'Close'])
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span=n, min_periods=n).mean())
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean() / ATR)
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean() / ATR)
    ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)).ewm(span=n_ADX, min_periods=n_ADX).mean(),
                    name='ADX_' + str(n) + '_' + str(n_ADX))
    df = df.join(ADX)
    return df
def on_balance_volume(df, n):
    """Calculate On-Balance Volume for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] > 0:
            OBV.append(df.loc[i + 1, 'Volume'])
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] == 0:
            OBV.append(0)
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] < 0:
            OBV.append(-df.loc[i + 1, 'Volume'])
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(OBV.rolling(n, min_periods=n).mean(), name='OBV_' + str(n))
    df = df.join(OBV_ma)
    return df

def AR_X(df):
    #print(df.head())
    df1 = pd.DataFrame(df)
    df2=df1[["Close"]]    
    print(df2.head())
    df2.plot()
    plt.show()
    lag_plot(df2)
    plt.show()
    ##########check  Autocorrelation
    values = DataFrame(df2.values)
    dataframe = concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t-1', 't+1']
    result = dataframe.corr()
    print(result)
    #Autocorrelation Plots
    autocorrelation_plot(df2)
    plt.show()
    plot_acf(df2,lags=31)
    plt.show()
    ######
    ''' split dataset
    X = df2.values
    train, test = X[1:len(X)-7], X[len(X)-7:]
    # train autoregression
    model = AR(train)
    model_fit = model.fit()
    print('Lag: %s' % model_fit.k_ar)
    print('Coefficients: %s' % model_fit.params)
    # make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    for i in range(len(predictions)):
    	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot results
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()'''
    
    # split dataset
    X = df2.values
    train, test = X[1:len(X)-7], X[len(X)-7:]
    # train autoregression
    model = AR(train)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    # walk forward over time steps in test
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(20):
    	length = len(history)
    	lag = [history[i] for i in range(length-window,length)]
    	yhat = coef[0]
    	for d in range(window):
    		yhat += coef[d+1] * lag[window-d-1]
    	#obs = test[t]
    	predictions.append(yhat)
    	history.append(yhat)
    	#print('predicted=%f' % (yhat))
    #error = mean_squared_error(test, predictions)
    #print('Test MSE: %.3f' % error)
    # plot
    plt.plot(test)
    plt.plot(predictions, color='red')
    #f1, ax11 = plt.subplots(figsize = (8,4))
    #ax11.plot(df.index, test['Close'], color = 'blue', lw=2, label='Test')
    #ax11.plot(df.index, predictions['yhat'], color ='red', lw=1, label='Predictions')
   # plt.set(title = 'AR Prediction of Stock ', ylabel = 'Price',xlabel='Date')
    #plt.legend(loc='upper right')
    #plt.xticks(rotation='vertical')
    plt.show()
    print(pd.DataFrame(history))

if var == "0":
   print( "RSI smart process evaluation started ..... ")
   rsi=RSI_method(ext_stock,14)
   callme()
elif var == "1":
   print( " Moing average convergence divergence smart process evaluation started ..... ")
   MACD=Moving_avg_con_div(ext_stock,9,26)
   print(MACD)
   callme()
elif var == "2":
   print ("Moving average for 14 days evaluation started ..... ")
   ma=Moving_Avg_MAPE(callme.ext_stock,14)
   print(ma)
   callme()
   
elif var == "3":
   print ("Bollinger Method evaluation !")
   bb=Bollinger_band(ext_stock,14)
   print(bb)
   callme()
   
elif var =="4":
   print ("Fast Stochastic Method evaluation !")
   fs=fast_stochastic(callme.ext_stock,14,3)
   print(fs) 
   callme()
   
elif var =="5":
   print ("Slow Stochastic Method evaluation !")
   ss=slow_stochastic(ext_stock,14,3)
   print(ss)
   callme()
   
elif var =="6":
   print ("Rate of change evaluation !")
   roc=ROC(ext_stock,8)
   print(roc) 
   callme()
   
elif var =="7":
   print ("William Average evaluation !")
   william=williams_r(ext_stock,14)
   print(william)
   callme()
   
elif var =="8":
   print ("Commodity Channel Index")
   cciv=CCI(ext_stock,20)
   print(cciv)
   callme()
   
elif var =="9":
   print ("Force Index")
   Forcei=Force_Index(ext_stock,1)
   print(Forcei)
   callme()  
elif var=="10":
    print("Thank You")
else:
   print ("Select the apropriate method type : ")
   """
   elif var =="10":
   print ("AR")
   adr=AR_X(ext_stock)
   print(adr)
   callme()
   """
# Data for matplotlib finance plot
   #elif var =="10":
#   print ("ADX")
#   adx=AD_X(ext_stock,14,14)
#   print(adx)
#   callme()
 
"""
r=[]
r=rsi
rsi =rsi['RSI']
plt.rc('axes', grid=True)
plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)

textsize = 9
left, width = 0.1, 0.8
rect1 = [left, 0.7, width, 0.2]
rect2 = [left, 0.3, width, 0.4]
rect3 = [left, 0.1, width, 0.2]


fig = plt.figure(facecolor='white')
axescolor = '#f6f6f6'  # the axes background color

ax1 = fig.add_axes(rect1, axisbg=axescolor)  # left, bottom, width, height
ax2 = fig.add_axes(rect2, axisbg=axescolor, sharex=ax1)
ax2t = ax2.twinx()
ax3 = fig.add_axes(rect3, axisbg=axescolor, sharex=ax1)

fillcolor = 'darkgoldenrod'

ax1.plot(r.Date, rsi, color=fillcolor)
ax1.axhline(70, color=fillcolor)
ax1.axhline(30, color=fillcolor)
ax1.fill_between(r.Date, rsi, 70, where=(rsi >= 70), facecolor=fillcolor, edgecolor=fillcolor)
ax1.fill_between(r.Date, rsi, 30, where=(rsi <= 30), facecolor=fillcolor, edgecolor=fillcolor)
ax1.text(0.6, 0.9, '>70 = overbought', va='top', transform=ax1.transAxes, fontsize=textsize)
ax1.text(0.6, 0.1, '<30 = oversold', transform=ax1.transAxes, fontsize=textsize)
ax1.set_ylim(0, 100)
ax1.set_yticks([30, 70])
ax1.text(0.025, 0.95, 'RSI (14)', va='top', transform=ax1.transAxes, fontsize=textsize)
ax1.set_title('%s daily' % r.Symbol)


def switch_demo(method_val):
    switcher = {
                0: RSI_method(),
                1: Moving_Avg_MAPE(), 
                2: Moving_avg_con_div(),
                3: Bollinger_band()
    }

    return switcher.get(method_val,"Invalid Stock analysis method ")

print(switch_demo(method_val))
    
"""




