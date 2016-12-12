# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 17:45:14 2016

@author: ngoldbergerr
"""

import sys
sys.path.insert(0,"""F:\Quant Analysis\portfolioManager\RatesAnalysis""")
sys.path.insert(0,"""L:\Rates & FX\Quant Analysis\portfolioManager\Monitors""")
#import xlrd
import pandas as pd
import numpy as np
import datetime as dt
#import os.path
from tia.bbg import v3api
import matplotlib.pyplot as plt

class moneyMarket(object):

        def __init__(self, path = 'L:\Rates & FX\Quant Analysis\portfolioManager\Monitors'):
            self.XLpath = path
### Get data from Bloomberg
        def getHistDataFromBloomberg(self, tickers, init = dt.datetime.today()-dt.timedelta(weeks=104), end = dt.datetime.today()-dt.timedelta(days=1)):
            
            LocalTerminal = v3api.Terminal('localhost', 8194)        
            try:
                response = LocalTerminal.get_historical(tickers, ['PX_LAST'], ignore_security_error=1, ignore_field_error=1, start = init, end = end)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                return False
    
            bloombergData = response.as_frame()
            
            return bloombergData
            
        def getDataFromBloomberg(self,tickers,fields):
            LocalTerminal = v3api.Terminal('localhost', 8194)  
            response = LocalTerminal.get_reference_data(tickers, fields, ignore_security_error = 1, ignore_field_error = 1)
            return response.as_frame()
### Build graphs            
        def buildHistoricalDepoCurve(self):
            tickers = ['CLTN30DN Index','CLTN90DN Index','CLTN180N Index','CLTN360N Index']
            fields = ['SHORT_NAME','PX_LAST','CHG_NET_1M','CHG_NET_YTD']
            bloombergData = self.getDataFromBloomberg(tickers,fields)
            table = pd.DataFrame(bloombergData)
            table.index = [180,30,360,90]
            del table['SHORT_NAME']
            table = table/12
            table['CHG_NET_1M'] = table['PX_LAST']-table['CHG_NET_1M']
            table['CHG_NET_YTD'] = table['PX_LAST']-table['CHG_NET_YTD']
            table.columns = ['Last Price','One Month','Beginning of Year']
            curve = pd.DataFrame(table, index = [30,60,90,120,150,180,210,240,270,300,330,360], columns = table.columns)
            curve = curve.interpolate(method='linear')
#            fig, ax = plt.subplots(1,1)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.style.use('bmh') # fivethirtyeight, dark_background, bmh, grayscale
            
            ax.plot(curve.index, curve['Last Price'], '--', markersize=4)
            ax.plot(curve.index, curve['One Month'], '--', markersize=4)
            ax.plot(curve.index, curve['Beginning of Year'], '--', markersize=4)
            plt.legend(['Last Price','One Month','Beginning of Year'], fontsize=15, loc=4)     

            ax.plot(table.index, table['Last Price'], 'o', markersize=7, markerfacecolor = 'c')
            A = table.index            
            B = table['Last Price'].values

            for x, y in zip(A,B):
                ax.annotate('%s' % round(y,3), xy=(x-10,y+0.003), textcoords='data')
            
            ax.plot(table.index, table['One Month'], 'o', markersize=7, markerfacecolor = 'r')
            A = table.index            
            B = table['One Month'].values

            for x, y in zip(A,B):
                ax.annotate('%s' % round(y,3), xy=(x-10,y+0.003), textcoords='data')
                
            ax.plot(table.index, table['Beginning of Year'], 'o', markersize=7, markerfacecolor = 'purple')
            A = table.index            
            B = table['Beginning of Year'].values

            for x, y in zip(A,B):
                ax.annotate('%s' % round(y,3), xy=(x-10,y+0.003), textcoords='data')

            fig.set_size_inches(12, 8)
            plt.subplots_adjust(top=0.85)
            plt.xlabel('Tenor (dias)', fontsize=15)
            plt.xticks(fontsize=15)
            plt.ylabel('Tasa (%)', fontsize=15)
            plt.yticks(fontsize=15)
            plt.title('Evolucion Curva de Depositos', fontsize=18, y=1.05)
            plt.xticks(np.arange(min(table.index), max(table.index)+1, 30.0))
            
            plt.show()
            return curve.sort_index(ascending=True)
        
        def buildCLPDepoTimeSeries(self):
            tickers = ['CLTN30DN Index','CLTN180N Index','CLTN360N Index']
            ratesCLP = self.getHistDataFromBloomberg(tickers)
            ratesCLP.columns = ['Depositos CLP 360D','Depositos CLP 30D','Depositos CLP 180D']
            ratesCLP = self.set_column_sequence(ratesCLP,['Depositos CLP 30D','Depositos CLP 180D','Depositos CLP 360D'])
            ratesCLP.plot(figsize=(12,8),title='Evolucion Tasas de Depositos en CLP',fontsize=14)
            return ratesCLP
            
        def buildCLFDepoTimeSeries(self):
            tickers = ['PCRR180D Index','PCRR360D Index']
            ratesCLF = self.getHistDataFromBloomberg(tickers)
            ratesCLF.columns = ['Depositos UF 360D','Depositos UF 180D']
            ratesCLF.plot(figsize=(12,8),title='Evolucion Tasas de Depositos en UF',fontsize=14)
            return ratesCLF

        def buildSpreadTimeSeries(self):
            tickers = ['CLTN360N Index','CHSWP1 Curncy']
            rates_data = self.getHistDataFromBloomberg(tickers)
            spread = rates_data['CLTN360N Index'] - rates_data['CHSWP1 Curncy']
            spread.columns = ['Spread']
            spread['Promedio'] = spread['Spread'].mean()
            spread.plot(figsize=(12,8),title='Evolucion Swap CLP/Camara 1Y vs Deposito 360D',fontsize=14)
            return spread

        def buildImpliedInflationTimeSeries(self):
            tickers = ['CLTN360N Index','CLTN180N Index','PCRR360D Index','PCRR180D Index']
            rates_data = self.getHistDataFromBloomberg(tickers)
            rates_data.columns = ['CLTN180N Index','CLTN360N Index','PCRR180D Index','PCRR360D Index']
            inflation = pd.DataFrame()
            inflation['Inflacion 180D'] = rates_data['CLTN180N Index'] - rates_data['PCRR180D Index']
            inflation['Inflacion 360D'] = rates_data['CLTN360N Index'] - rates_data['PCRR360D Index']
            inflation.plot(figsize=(12,8),title='Evolucion Inflacion Implicita',fontsize=14)
            return inflation
            
### Support functions
        def set_column_sequence(self, dataframe, seq, front=True):
            '''Takes a dataframe and a subsequence of its columns,
               returns dataframe with seq as first columns if "front" is True,
               and seq as last columns if "front" is False.
            '''
            cols = seq[:] # copy so we don't mutate seq
            for x in dataframe.columns:
                if x not in cols:
                    if front: #we want "seq" to be in the front
                        #so append current column to the end of the list
                        cols.append(x)
                    else:
                        #we want "seq" to be last, so insert this
                        #column in the front of the new column list
                        #"cols" we are building:
                        cols.insert(0, x)
            return dataframe[cols]



if __name__ == '__main__':
    money = moneyMarket()
    curvaDepo = money.buildHistoricalDepoCurve()
    timeSeriesDepoCLP = money.buildCLPDepoTimeSeries()
    timeSeriesDepoCLF = money.buildCLFDepoTimeSeries()
    spread = money.buildSpreadTimeSeries()
    inflation = money.buildImpliedInflationTimeSeries()