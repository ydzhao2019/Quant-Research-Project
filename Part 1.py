import pandas as pd
import numpy as np 
import statsmodels.api as sm
import matplotlib.pyplot as plt 
#import stock divide ,delte 1880 and 2540 because of missing data
stocks_divide=pd.read_excel('Project_603.xlsx',header=0,index_col=0)
stocks_divide=stocks_divide.drop(['1880.csv','2540.csv'])
#pick stocks for Team7
stocks=stocks_divide.index[stocks_divide['Team']==7]
#constuct stock names for later use
stocks_name=[]
for i in stocks:
	stock_name=i[0:4]
	stocks_name.append(stock_name)
#pick close price and concate to dataframe，use stock names as columns
data=[]
for i in stocks:
	datai=pd.read_csv(i,header=0,index_col=0)
	data.append(datai['Close'])
data=pd.concat(data,axis=1,keys=stocks_name,sort=False)
#Tranfer index to date format
data.index=pd.to_datetime(data.index)
#pick data from 1999-04 to 2018-12
data=data.loc['1999-04-01':'2018-12-31',:]
#also can use: data=data.groupby(pd.Grouper(freq='BM')).last() 
#resample data to monthly price
data=data.resample('BM').last()
#compute monthly simple returns,divide into train set and test set
msr=((data/data.shift(1))-1).dropna()
trainmsr=msr.loc['1999-04-01':'2013-12-31',:]
testmsr=msr.loc['2014-01-02':'2018-12-31',:]
#import rf，transfer to decimal format
rf=pd.read_csv('FFF.csv',header=0,index_col=0)['RF'].loc['199905':'201812']/100
trainrf=rf.loc['199905':'201312']
testrf=rf.loc['201401':'201812']
#compute monthly simple excess returns
trainmser=trainmsr-np.tile(trainrf.values.reshape(-1,1),len(trainmsr.columns))
testmser=testmsr-np.tile(testrf.values.reshape(-1,1),len(testmsr.columns))

#import SP500 data
SP500=pd.read_csv('GSPC.csv',header=0,index_col=0)['Close']
#transfer index to date format
SP500.index=pd.to_datetime(SP500.index)
#pick data from 1999-04 to 2018-12
SP500=SP500.loc['1999-04-01':'2018-12-31']
#resample data to monthly price
SP500=SP500.resample('BM').last()
#compute monthly simple returns,divide into train set and test set
msr_SP=((SP500/SP500.shift(1))-1).dropna()
trainmsr_SP=msr_SP.loc['1999-04-01':'2013-12-31']
testmsr_SP=msr_SP.loc['2014-01-02':'2018-12-31']
#compute monthly simple excess returns
trainmser_SP=trainmsr_SP-trainrf.values
testmser_SP=testmsr_SP-testrf.values

#Regrssion
trainmser_SP1=sm.add_constant(trainmser_SP)
beta=pd.DataFrame([(sm.OLS(trainmser.iloc[:,i],trainmser_SP1).fit()).params[1] for i in range(len(trainmser.columns))],index=trainmser.columns,columns=['Beta'])
alpha=pd.DataFrame([(sm.OLS(trainmser.iloc[:,i],trainmser_SP1).fit()).params[0] for i in range(len(trainmser.columns))],index=trainmser.columns,columns=['Alpha'])

#Sort alpha from the largest to the smallest
alpha=alpha.sort_values(by='Alpha',ascending=False)
print(alpha)
longstocks=alpha.index[:10]
shortstocks=alpha.index[-10:]
print(longstocks)
print(shortstocks)

#construct portfolio's return
testrp=(testmsr[longstocks]-testmsr[shortstocks].values).mean(axis=1)
print(testrp)


#Treynor ratio of portfolio
testmser_P=testrp-testrf.values
testmser_SP1=sm.add_constant(testmser_SP)
reg3=sm.OLS(testmser_P,testmser_SP1).fit()
beta_P=reg3.params[1]
print(beta_P)
Treynor=(testmser_P.mean())/beta_P
print('Treynor ratio',Treynor)

#Jensen's alpha
alpha_P=reg3.params[0]
print('Alpha',alpha_P)

#Sharpe ratio
sigma_P=testmser_P.std()
Sharpe=(testmser_P.mean())/sigma_P
print('Sharpe ratio',Sharpe)

#M square
sigma_SP=testmser_SP.std()
M2=(testmser_P.mean())/sigma_SP*sigma_SP-testmser_SP.mean()
print('M2',M2)

#Information Ratio
testmser_P2=testrp-testmser_SP
sigma_P2=testmser_P2.std()
IR=testmser_P2.mean()/sigma_P2
print('IR',IR)


#Analysis
print(reg3.summary())
plt.plot(testmser_P,label='Portfolio')
plt.plot(testmser_SP,label='SP500')
plt.legend()
plt.show()
table=pd.concat([testmser_P,testmser_SP],axis=1)
corr=table.corr().iloc[0,1]
print('correlation',corr)
mean_P=testmser_P.mean()
print('mean_P',mean_P)
mean_SP=testmser_SP.mean()
print('mean_SP',mean_SP)
sigma_SP=testmser_SP.std()
Sharpe_SP=mean_SP/sigma_SP
print('Sharpe_SP',Sharpe_SP)
plt.plot(SP500)
plt.show()


    

























