import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

from scipy import stats, signal
from numpy.fft import fft, ifft
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
#%pip pmdarima

# Visualization libraries
import streamlit as st
import seaborn as sns
import seaborn; seaborn.set()
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title='GDP Forecasting',
    page_icon='https://static.vecteezy.com/system/resources/previews/021/210/715/original/forecasting-icon-style-free-vector.jpg'
    )

st.image('cover_st_project.webp',use_column_width=True)
st.title('GDP Forecasting')

st.write("""
This project focuses on the analysis and the forecasting of the total GDP (Gross Domestic Product) for the last decade (2010-2020) in five different countries: Canada, China, Italy, Japan, and the United States.
The dataset contains estimates of total GDP and its components across 220 countries, with 17 derived indicators, covering the period from 1970 to 2020.

### Objectives:

1. **Forecasting Models:** Using various models such as Linear Regression, ETS models (Error, Trend, Seasonality), and ARIMA (AutoRegressive Integrated Moving Average) for GDP prediction.
  
2. **Comparing the forecasting models based on two key metrics:** AIC (Akaike Information Criterion) and RMSE (Root Mean Squared Error). This evaluation will provide insights into the accuracy and goodness of fit for each model.

3. **Variable Analysis:** Beyond GDP, the project aims to analyze and visualize additional variables for each country, including Exports, Imports, Manufacturing (ISIC D), and Gross Capital Formation. These variables contribute to a holistic understanding of the economic landscape.

The comprehensive analysis and forecasting for these key economic indicators will contribute to a deeper understanding of the economic trends and dynamics in the specified countries, facilitating informed decision-making and strategic planning.
""")

# Additional Information
st.sidebar.title("Glossary")
st.sidebar.write("""
*  **Gross Domestic Product (GDP)** --> the total of all value added created in an economy.
*  **Gross capital formation** --> consists of outlays on additions to the fixed assets of the economy plus net changes in the level of inventories
*  **Imports of goods and services** --> represent the value of all goods and other market services received from the rest of the world.
*  **Exports of goods and services** --> represent the value of all goods and other market services provided to the rest of the world.
*  **Manufacturing (ISIC D)**	--> is the net output of a sector after adding up all outputs and subtracting intermediate inputs. """)


# Upload the dataset
gdp_df= pd.read_excel('/Users/GaiaTravaglini/Desktop/Project/GDPcountries.xlsx', header=2)

#Drop unuseful columns and rename NameIndicator
gdp_df.drop(['CountryID'], axis=1, inplace=True)
gdp_df.rename(columns={'IndicatorName':'Name_indicator'}, inplace=True)

# Melt the different date columns together to get a single column (Date)
gdp_df_melted = gdp_df.melt(id_vars =['Country', 'Name_indicator'],
                var_name = 'Date',
                value_name = 'Value').dropna()

#setting Date as index
gdp_df_melted['Date'] = pd.to_datetime(gdp_df_melted['Date'].astype(str).str.replace(r'\.0$', ''), format="%Y") #convert the Date variable into Datatime
gdp_df_melted= gdp_df_melted.set_index(gdp_df_melted ['Date'])
gdp_df_melted.drop(['Date'], axis=1, inplace=True) #Drop the date column since now it is an index
gdp_df_melted.head()

#Focusing on 5 different countries: Italy, Germany, United States, China, Japan
countries = ['Italy', 'Canada', 'United States', 'China', 'Japan']
gdp_sub_df = gdp_df_melted[(gdp_df_melted['Country'].apply(lambda x: x in countries))]

#Focusing on the 5 selected categories
categories = ['Gross Domestic Product (GDP)', 'Exports of goods and services', 'Imports of goods and services','Manufacturing (ISIC D)','Gross capital formation']
gdp_sub_df= gdp_sub_df[gdp_sub_df['Name_indicator'].apply(lambda x: x in categories)]

st.write('The Dataset:')
st.write(gdp_sub_df.head(5))
if st.checkbox('Show Data Description'):
    st.write('Size:',gdp_sub_df.shape)
    st.write('**Categorical Data**:', '2 attributes',  '_(Country, Name_indicator)_')
    st.write('**Quantitative Data**:', '1 attribute',  '_(Value)_')
    if gdp_sub_df.isna().sum().sum() == 0:
       st.write('There are no missing values')
    else:
        st.write('There are still', gdp_sub_df.isna().sum().sum(),'missing values' )
    
st.header('Data Analysis')
st.subheader('Plot of the different categories for each country')

fig = make_subplots()

for country in gdp_sub_df['Country'].unique():
    country_df = gdp_sub_df[gdp_sub_df['Country'] == country]

    for category in country_df['Name_indicator'].unique():
        category_df = country_df[country_df['Name_indicator'] == category]

        # Add scatter trace for the category
        trace = go.Scatter(x=category_df.index,
                           y=category_df['Value'],
                           mode='lines+markers',
                           name=category)
        fig.add_trace(trace)

# Adding buttons for each country and category
updatemenus = []
buttons = []

for i, country in enumerate(gdp_sub_df['Country'].unique()):
    visible = [False] * len(fig.data)
    visible[i * 5: (i + 1) * 5] = [True] * 5  # Show traces for the current country

    buttons.append(
        dict(args=[{'visible': visible},
                    {'title': f'Scatter Plot for {country}'}],
             label=country,
             method='update'))

# Set configurations for the buttons
updatemenus = [{'buttons': buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.175,
                'xanchor': 'right',
                'y': 1.10,
                'yanchor': 'top'}]

# Adding labels and title
fig.update_layout(
    title='Scatter Plot for the 5 Categories',
    xaxis_title='Date',
    yaxis_title='Value',
    updatemenus=updatemenus,
    height=600,
    width=800
)
st.write(fig)
st.caption('The graph allows for a more comprehensive view on the distribution in time of each category for each of the selected country.')
st.write('From the plot, we can see that almost every variable seems to be highly correlated with each other for every country.\n \n \n Then, we can verify it by definining the correlation function between those variables')

# Filter for one dataset
pivot_df = gdp_sub_df[gdp_sub_df['Country'] == 'Canada']
pivot = pd.pivot_table(data=pivot_df, index='Date',columns='Name_indicator', values='Value')

#scatter plot of Canadian exports and imports
plt.figure(figsize=(8, 4))
plt.scatter(pivot['Exports of goods and services'], pivot['Imports of goods and services'])
plt.grid()
plt.title("Canadian Exports and Imports")
plt.xlabel('Exports of goods and services \n(hundred billions)')
plt.ylabel('Imports of goods and services \n(hundred billions)')
st.pyplot()

def correlation(x,y):
  A = np.sum((x-np.mean(x))*(y-np.mean(y)))
  B = np.sum((x-np.mean(x))**2) *np.sum((y-np.mean(y))**2)
  corr = A/(B)**0.5
  return(corr)

#Correlation between Exports and Imports
x = pivot['Exports of goods and services']
y = pivot['Imports of goods and services']
corr = correlation(x,y)
st.text(f'The correlation value is:{corr}.')
st.text('Then, the two variables are highly positive correlated.')

st.write('In addition, we can check for the correlation of all the other variables through a correlation matrix')
fig, ax = plt.subplots(figsize=(8, 4))
corr_mat = pivot.corr()
sns.heatmap(corr_mat, annot=True)
plt.title('Correlation Matrix')
st.pyplot()
st.caption('As expected, all the variables present high correlation within each other')


st.subheader('Plot of the GDP Indicator')

fig = go.Figure()

for country in gdp_sub_df['Country'].unique():
    country_df = gdp_sub_df[(gdp_sub_df['Country'] == country) & (gdp_sub_df['Name_indicator'] == 'Gross Domestic Product (GDP)')]

    trace = go.Scatter(
        x=country_df.index,
        y=country_df['Value'],
        mode='lines+markers',
        name=country
    )
    fig.add_trace(trace)

# Adding labels and title
fig.update_layout(
    title='Scatter Plot for Gross Domestic Product (GDP)',
    xaxis_title='Date',
    yaxis_title='Value'
)
st.write(fig)
st.write('It is also useful to catch more information about the autocorrelation of the GDP for a specific country in different time steps.')

# Filter for the GDP of Canada
gdp_signal = pivot['Gross Domestic Product (GDP)']

# Lag plots for the Canadian GDP
fig, axes = plt.subplots(3,3, sharex=True, sharey=True, figsize=(8,8))

for i, ax in enumerate(axes.flatten()[:9]):
    pd.plotting.lag_plot(gdp_signal, lag=i+1, ax=ax, c='black')
    ax.set_xlabel("y(t)")
    ax.set_ylabel("y(t+"+str(i+1)+")")
st.pyplot()
st.caption('The lag plot suggests for strong positive autocorrelation')

st.write('In addition, we can consider both the manual implementation of the ACF and Automatic ACF:')
# Using both Automatic ACF and Manual ACF
signal_demean = gdp_signal - gdp_signal.mean() # Signal has to be demean in the manual version (this is automatically done in the plot_acf function)
n = np.array(signal_demean,dtype = float)

fig, ax = plt.subplots(1,2, figsize=(8,4))
plot_acf(gdp_signal, lags=9, ax = ax[0])
ax[0].set_title('Automatic ACF')

ax[1]=plt.acorr(n,normed = True, maxlags = 9)
ax[1] = plt.title('Manual ACF')
st.pyplot()
st.caption('Again, the autocorrelation of GDP seems to be high for all the lags, especially for the first ones.')

st.write('It is possibile to expand this insight about autocorrelation to understand if all the other variables present the same patten')
#ACF for all Canadian variables
n = 5
fig, axes = plt.subplots(n, int(len(pivot.columns)/n), figsize=(20, 18))
fig.tight_layout()

# Plot ACF for all Canadian variables
for i, ax in enumerate(axes.flatten()):
    variable_name = pivot.columns[i]
    plot_acf(pivot[variable_name], lags=10, ax=ax, title=variable_name)

st.pyplot()

st.subheader('Nominal GDP vs. GDP per Capita')
st.write('Besides the analysis of each country GDP, a more relevant analysis could be done by looking at the nominal GDP the Nominal GDP and the GDP per capita of each country.')

# Get GDP per Capita Dataset (Source: https://data.worldbank.org/indicator/NY.GDP.PCAP.CD )

gdp_capita_df= pd.read_csv('/Users/GaiaTravaglini/Desktop/Project/all_countries_gdp_per-capita.csv',  delimiter=',', header=2)
gdp_capita_df.drop(columns={'Country Code','Indicator Code', 'Indicator Name'}, axis=1, inplace=True)
gdp_capita_melted_df = gdp_capita_df.melt(id_vars =['Country Name'],
                var_name = 'Date',
                value_name = 'GDP per Capita').dropna()
gdp_capita_melted_df['Date']= pd.to_datetime(gdp_capita_melted_df['Date'], format= "%Y") #convert the Date variable into Datatime
gdp_capita_melted_df= gdp_capita_melted_df.set_index(gdp_capita_melted_df['Date'])
gdp_capita_melted_df.drop(['Date'], axis=1, inplace=True) #Drop the date column since now it is an index
gdp_capita_melted_df.head()

# Filtering
countries = ['Italy', 'Canada', 'United States', 'China', 'Japan']
gdp_capita_sub_df = gdp_capita_melted_df[gdp_capita_melted_df['Country Name'].apply(lambda x: x in countries)]
if st.checkbox('Show GDP per Capita'):
    st.write(gdp_capita_sub_df.head(5))

# Filter the original dataset for the indicator GDP
nominal_gdp = gdp_sub_df[gdp_sub_df['Name_indicator'] == 'Gross Domestic Product (GDP)']
# Merge the two dataset and apply the necessary transformation
merged_df = pd.merge(gdp_capita_sub_df, nominal_gdp[['Country','Value']], how='right', left_on=['Country Name',gdp_capita_sub_df.index ], right_on=['Country', nominal_gdp.index])
merged_df.rename(columns={'key_1':'Date','Value':'Nominal GDP'}, inplace= True)
merged_df.drop(['Country Name'], axis=1, inplace=True)
merged_df = merged_df[['Date', 'Country', 'Nominal GDP', 'GDP per Capita']]
merged_df = merged_df.reset_index()
merged_df.drop(['index'], axis=1, inplace=True)
merged_df= merged_df[merged_df['Date'] == '2020-01-01']

df = merged_df.sort_values('Nominal GDP')
new_gdp = merged_df.sort_values('GDP per Capita')

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=['GDP 2020', 'GDP per capita 2020'],column_widths=[0.5, 0.5])

# Add bar plot for Nominal GDP
fig.add_trace(go.Bar(x=df['Country'], y=df['Nominal GDP'], name='Nominal GDP'), row=1, col=1)
fig.update_xaxes(title_text='Country', row=1, col=1)
fig.update_yaxes(title_text='Nominal GDP', row=1, col=1)

# Add bar plot for GDP per Capita
fig.add_trace(go.Bar(x=new_gdp['Country'], y=new_gdp['GDP per Capita'], name='GDP per Capita'), row=1, col=2)
fig.update_xaxes(title_text='Country', row=1, col=2)
fig.update_yaxes(title_text='GDP per Capita', row=1, col=2)
fig.update_layout(showlegend=False, title_text='Comparison between Nominal GDP and GDP per Capita', height=500, width=1000)
st.write(fig)
st.caption('We can notice how the things change, for example, for China, which is the second country for GDP value. Concerning its GDP per Capita, it has a much lower rank.  '
         'This is due to the fact that GDP per capita, in general, is by design an indicator of the total income generated by economic activity in a country and is often used as a measure of people’s material well-being.')

st.header('Frequency Analysis')
#PSD using the raw periodogram
Fs = 1 #The unit of measure to consider is "years", thus 1 per year
f_per, Pxx_per = signal.periodogram(gdp_signal,Fs,detrend=None,window=None,return_onesided=True,scaling='density')
plt.figure(figsize=(8, 4)) 
plt.plot(f_per[1:],Pxx_per[1:])
plt.title('GDP Periodogram')
plt.ylabel('PSD')
plt.xlabel('Frequency [cycles/day]');
peaks = signal.find_peaks(Pxx_per[f_per >=0], prominence=100)[0]
peak_freq =  f_per[peaks]
peak_power = Pxx_per[peaks]
plt.plot(peak_freq, peak_power, 'ro');
signal = {'Freq': peak_freq, 'Period': 1/peak_freq, 'Power': peak_power}
signal_df = pd.DataFrame(signal)
st.write(signal_df)
st.pyplot()
st.caption('4 peaks:(6, 4, 3, 2 years) repetition of data')

from scipy import signal
signal_imp = pivot['Exports of goods and services']
Fs = 1 #our unit of measure is "years", thus 1 per year
f_per, Pxx_per = signal.periodogram(signal_imp,Fs,detrend=None,window=None,return_onesided=True,scaling='density')
plt.figure(figsize=(8, 4)) 
plt.plot(f_per[1:],Pxx_per[1:])
plt.title('Exports Periodogram')
plt.ylabel('PSD')
plt.xlabel('Frequency [cycles/day]');

peaks = signal.find_peaks(Pxx_per[f_per >=0], prominence=100)[0]
peak_freq =  f_per[peaks]
peak_power = Pxx_per[peaks]
plt.plot(peak_freq, peak_power, 'ro');

signal = {'Freq': peak_freq, 'Period': 1/peak_freq, 'Power': peak_power}
signal_df = pd.DataFrame(signal)
st.write(signal_df)
st.pyplot()
st.caption('5 peaks:(16, 6, 4, 3, 2 years) repetition of data')

st.header('Pattern Analysis of Canada')
st.subheader('GDP Decomposition')

pivot_df = gdp_sub_df[gdp_sub_df['Country'] == 'Canada']
pivot = pd.pivot_table(data=pivot_df, index='Date',columns='Name_indicator', values='Value')
canada_gdp = pivot_df[pivot_df['Name_indicator'] == 'Gross Domestic Product (GDP)'	]
canada_gdp = canada_gdp.drop(['Country',	'Name_indicator'], axis=1)
canada_gdp = canada_gdp.asfreq('AS') #annual indexing, 'AS' for start of year
decompose_data_add = seasonal_decompose(canada_gdp, model='additive')
decompose_data_add.plot()
st.pyplot()
st.write('From the first two plots, it seems that the TS is entirely taken as the trend component with no seasonality at all. \n In addition, also the residual plot shows zero. ')
st.write('This may be due to the fact that the classical decomposition was not able to separate the noise added from the linear trend. \n One possible cause seems to be the yearly frequency of the TS (period =1).')

st.subheader('Other Variables Decomposition')
st.write('Now, we can check if the same result is obtained for the other variables')
canada_exp = pivot['Exports of goods and services'].asfreq('AS')
canada_imp = pivot['Imports of goods and services'].asfreq('AS')
canada_gcf= pivot['Gross capital formation'].asfreq('AS')
canada_man = pivot['Manufacturing (ISIC D)'].asfreq('AS')
    
def plot_seasonal(res, axes, title):
    res.observed.plot(ax=axes[0], legend=False)
    axes[0].set_title(f'{title} - Observed')
    axes[0].set_xlabel('')

    res.trend.plot(ax=axes[1], legend=False)
    axes[1].set_title(f'{title} - Trend')
    axes[1].set_xlabel('')

    res.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_title(f'{title} - Seasonal')
    axes[2].set_xlabel('')

    res.resid.plot(ax=axes[3], legend=False)
    axes[3].set_title(f'{title} - Residual')
    axes[3].set_xlabel('')

decompose_exports = seasonal_decompose(canada_exp, model='additive')
decompose_imports = seasonal_decompose(canada_imp, model='additive')
decompose_gcf = seasonal_decompose(canada_gcf, model='additive')
decompose_man = seasonal_decompose(canada_man, model='additive')

#Create subplots for each variable
fig, axes = plt.subplots(nrows=4, ncols=4, sharey=True, figsize=(18, 12))
plot_seasonal(decompose_exports, axes[0, :], 'Exports')
plot_seasonal(decompose_imports, axes[1, :], 'Imports')
plot_seasonal(decompose_gcf, axes[2, :], 'Gross Capital Formation')
plot_seasonal(decompose_man, axes[3, :], 'Manufacturing')
plt.tight_layout()
st.pyplot()
st.write('Again, the same pattern exploited with the GDP decomposition holds. ')

st.subheader('MA Smoothing')
st.write('One possible solution could be apply **moving average smoothing** of order 5 to all the variables to decompose the **trend and cyclical pattern**')
exports_df=canada_exp.to_frame()
imports_df=canada_imp.to_frame()
gcf_df=canada_gcf.to_frame()
man_df=canada_man.to_frame()

canada_gdp['5-MA'] = canada_gdp['Value'].rolling(5, center=True).mean()
exports_df['5-MA'] = exports_df['Exports of goods and services'].rolling(5, center=True).mean()
imports_df['5-MA'] = imports_df['Imports of goods and services'].rolling(5, center=True).mean()
gcf_df['5-MA'] = gcf_df['Gross capital formation'].rolling(5, center=True).mean()
man_df['5-MA'] = man_df['Manufacturing (ISIC D)'].rolling(5, center=True).mean()
    
# Plot each original and smoothed variables to compare the TS
sns.set(style="whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(15, 10))

# GDP
sns.lineplot(data=canada_gdp, x=canada_gdp.index, y='Value', ax=axes[0, 0], label='Original data', linewidth=2)
sns.lineplot(data=canada_gdp, x=canada_gdp.index, y='5-MA', ax=axes[0, 0], label='5-MA', linewidth=2)
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('GDP')
axes[0, 0].legend()

# Exports
sns.lineplot(data=exports_df, x=exports_df.index, y='Exports of goods and services', ax=axes[0, 1], label='Original data', linewidth=2)
sns.lineplot(data=exports_df, x=exports_df.index, y='5-MA', ax=axes[0, 1], label='5-MA', linewidth=2)
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Exports')
axes[0, 1].legend()

# Imports
sns.lineplot(data=imports_df, x=imports_df.index, y='Imports of goods and services', ax=axes[1, 0], label='Original data', linewidth=2)
sns.lineplot(data=imports_df, x=imports_df.index, y='5-MA', ax=axes[1, 0], label='5-MA', linewidth=2)
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Imports')
axes[1, 0].legend()

# Gross Capital Formation
sns.lineplot(data=gcf_df, x=gcf_df.index, y='Gross capital formation', ax=axes[1, 1], label='Original data', linewidth=2)
sns.lineplot(data=gcf_df, x=gcf_df.index, y='5-MA', ax=axes[1, 1], label='5-MA', linewidth=2)
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Gross Capital Formation')
axes[1, 1].legend()

# Manufacturing
sns.lineplot(data=man_df, x=man_df.index, y='Manufacturing (ISIC D)', ax=axes[2, 0], label='Original data', linewidth=2)
sns.lineplot(data=man_df, x=man_df.index, y='5-MA', ax=axes[2, 0], label='5-MA', linewidth=2)
axes[2, 0].set_xlabel('Year')
axes[2, 0].set_ylabel('Manufacturing')
axes[2, 0].legend()
axes[2, 1].axis('off')
plt.tight_layout()
st.pyplot()

st.header('Forecasting')
st.subheader('Linear Regression')
st.write('Since we are dealing with a time series, linear regression cannot be directly applied as the time information should be considered. Then, we can introduce a time dummy representing the years. It counts the steps in the series from the beginning to end.')
canada_gdp['Year'] = range(canada_gdp.shape[0])
st.write(canada_gdp['Year'].head(5))
# Regression with the time dummy (Year)
x = canada_gdp['Year'].values
y = canada_gdp['Value'].values
x = x.reshape(len(x),1)
y = y.reshape(len(y),1)

model = LinearRegression().fit(x,y)
st.text(f'Intercept: {float(model.intercept_)}')
st.text(f'Slope: {float(model.coef_)}')

fitted_line = model.predict(x)
plt.plot(x, y, marker='o',label='Data')
plt.grid()
plt.title('Time Plot of Canadian GDP',weight='bold');
plt.plot(x,fitted_line,label='fitted line');
plt.xlabel('Time step')
plt.ylabel('GDP')
plt.grid()
plt.legend();
st.pyplot()

st.write('The evaluation metrics used to assess the fitting results:')
r2 = r2_score(y, fitted_line)
mse = mean_squared_error(y, fitted_line)
rmse = np.sqrt(mse)

N = len(y)
n_params = x.shape[1]
error = np.sum((y - fitted_line)**2)

# Calculate AIC for regression
s2 = error/N
LL  = -N/2*(1+np.log(2*np.pi)+np.log(s2))
AIC = -2*LL + 2*(n_params+1)

# Calculate BIC for regression
bic = -2*LL + (n_params+1)*np.log(N) # different penalty for the number of parameters compared to AIC

results_data = {
    'Metric': ['R2 Score', 'MSE', 'RMSE', 'Manual AIC', 'Manual BIC'],
    'Value': [r2, mse, rmse, float(AIC), float(bic)],
}

results_df = pd.DataFrame(results_data)

# Display the table
st.table(results_df)

st.subheader('Forecasting with Exponential Smoothing Models')
st.write(
    "We will use first **Simple exponential smoothing**, and then compare it with "
    "**Holt’s linear trend method (Double Exponential Smoothing)** with damped trend (using additive method)"
)
st.write('The first step is to split the dataset into Train and Test and then train the train dataset to get our forecasted values')
# Split between train and test dataset
canada_train_df = canada_gdp.Value[:-10]
canada_test_df = canada_gdp.Value[-10:]
st.write('Train Dataset:')
st.write(canada_train_df.tail())
st.write('Test Dataset:')
st.write(canada_test_df.head())

# Section for Simple Exponential Smoothing
st.subheader('Simple Exponential Smoothing')
st.write(
    "**Simple Exponential Smoothing**: This method is suitable for forecasting data with no clear trend or seasonal pattern. "
    "Forecasts are calculated using weighted averages, where the weights decrease exponentially as observations are distant in the past. "
    "(i.e., the smallest weights are associated with the oldest observations). SES has one forecast equation and a single smoothing equation for the level."
)

model = ETSModel(canada_train_df).fit()
predictions = model.forecast(10)

results_data = {
    'Metric': ['AIC', 'RMSE'],
    'Value': [model.aic, np.sqrt(model.mse)],
}

results_df = pd.DataFrame(results_data)
st.table(results_df)

# Forecasting in the future and confidence interval
pred = model.get_prediction(start='2010-01-01', end='2020-01-01')
df = pred.summary_frame()

# Model parameters
tab = {'Param Name': model.param_names, 'Values': model.params}
tab = pd.DataFrame(tab)

plt.plot(canada_train_df, color='green',label='Training data')
plt.plot(canada_test_df, color='black', label='Testing data')
plt.plot(predictions, color='red', label='Predictions')
plt.plot(model.fittedvalues,color='orange', label='one-step-ahead fitted values')
plt.fill_between(df.index, df['pi_lower'], df['pi_upper'], alpha=.1, color='crimson', label='95% CI')
plt.xlabel('Year')
plt.ylabel('Canadian GDP')
plt.legend();
st.pyplot()

# Section for Holt’s linear trend method
st.subheader("Holt’s linear trend method (Double Exponential Smoothing)")
st.write(
    "Extended SES method to allow the forecasting of data with a trend. "
    "This method involves a forecast equation and two smoothing equations (one for the level and one for the trend). "
    "To avoid over-forecast, we use a parameter that 'dampens' the trend to a flat line in the future."
)
# Train the model and predictions
Model = ETSModel(canada_train_df, trend='add', damped_trend=True).fit()
predictions = model.forecast(10)

results_data = {
    'Metric': ['AIC', 'RMSE'],
    'Value': [Model.aic, np.sqrt(Model.mse)],
}

results_df = pd.DataFrame(results_data)
st.table(results_df)

# Forecasting in the future and confidence interval
pred = Model.get_prediction(start='2010-01-01', end='2020-01-01')
df = pred.summary_frame()

# Model parameters
tab = {'Param Name': Model.param_names, 'Values': Model.params}
tab = pd.DataFrame(tab)

# Plot
plt.plot(canada_train_df, color='green',label='Training data')
plt.plot(canada_test_df, color='black', label='Testing data')
plt.plot(predictions, color='red', label='Predictions')
plt.plot(model.fittedvalues,color='orange', label='one-step-ahead fitted values')
plt.fill_between(df.index, df['pi_lower'], df['pi_upper'], alpha=.1, color='crimson', label='95% CI')
plt.xlabel('Year')
plt.ylabel('Canadian GDP')
plt.legend();
st.pyplot()

# Model Comparison
st.subheader('Model Comparison')
comparison_data = {
    'Model': ['Simple Exp Smoothing', 'Holt+damped trend'],
    'AIC': [model.aic, Model.aic],
    'RMSE': [np.sqrt(model.mse), np.sqrt(Model.mse)],
}

comparison_df = pd.DataFrame(comparison_data)
st.write(comparison_df)
st.write('By evaluating both models with AIC and RMSE, we can notice that the second one (the Holt Method with damped trend) is the best model to fit the data')

st.subheader('Forecasting with Arima Model')
st.write('The first step to implement the Arima Model is to check stationarity.') 

st.markdown('**Check stationarity with Augmented Dickey-Fuller test:**')
result_ADF = adfuller(canada_train_df)
st.write('ADF p-value:', result_ADF[1])
if result_ADF[1] <= 0.05:
    st.markdown('<p style="color:green;">The time series is stationary.</p>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:red;">The time series is not stationary.</p>', unsafe_allow_html=True)

# Add a separator
st.markdown('---')

# Check stationarity using KPSS test
st.markdown('**KPSS Test:**')
kpss_test = kpss(canada_train_df, regression='ct', nlags='auto', store=True)
st.write('KPSS p-value:', kpss_test[1])
if kpss_test[1] > 0.05:
    st.markdown('<p style="color:green;">The time series is stationary.</p>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:red;">The time series is not stationary.</p>', unsafe_allow_html=True)

st.write('Since the time series is not stationary, we can applied differenciation:')

warnings.filterwarnings("ignore")

data_diff_1 = canada_train_df.diff().dropna()

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), sharex=True)
axs[0].plot(canada_train_df)
axs[0].set_title('Original time series')
axs[1].plot(data_diff_1)
axs[1].set_title('Differenced of order 1')
st.pyplot()

st.markdown('**Check stationarity with ADF test after differenciation:**')
result_ADF = adfuller(data_diff_1)
st.write('ADF p-value:', result_ADF[1])
if result_ADF[1] < 0.05:
    st.markdown('<p style="color:green;">The time series is stationary.</p>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:red;">The time series is not stationary.</p>', unsafe_allow_html=True)

st.markdown('---')

st.markdown('**Check stationarity using KPSS test after differenciation:**')
kpss_test = kpss(data_diff_1, regression='ct', nlags='auto', store=True)
st.write('KPSS p-value: ', kpss_test[1])
if kpss_test[1] > 0.05:
    st.markdown('<p style="color:green;">The time series is stationary.</p>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:red;">The time series is not stationary.</p>', unsafe_allow_html=True)

st.markdown('---')
st.write('ACF and PACF plot')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
plot_acf(data_diff_1, ax=ax1);
plot_pacf(data_diff_1, ax=ax2);
st.pyplot()

st.markdown('---')
st.write('To choose the model parameters (in particular p and q), we can rely on Auto-ARIMA')
from pmdarima.arima import auto_arima

arima_model =  auto_arima(canada_train_df, start_p=0, d=0, start_q=0,
                          max_p=6, max_q=6, m=1, seasonal=False,
                          error_action='warn',trace = True,
                          supress_warnings=True,stepwise = False,
                          random_state=20,information_criterion='aicc')
best_params = arima_model.get_params()
st.text(f'Best ARIMA Model Parameters: {best_params}')

from statsmodels.tsa.arima.model import ARIMA

st.write('Then, we can implement the ARIMA model by using the following parameters:\n'
         '- p = 2\n'
         '- d = 1 (order of integration since differencing of order 1 makes the time series stationary)\n'
         '- q = 2')
mod_ARIMA = ARIMA(canada_train_df, order=(2,1,2)).fit() #order of integration d=1 since a differencing of order 1 allow to make the TS stationary.

warnings.filterwarnings("ignore")
checkbox_show_results = st.checkbox('Visualize the statistics and main results')
if checkbox_show_results:
    st.subheader('ARIMA Model Summary:')
    st.write(mod_ARIMA.summary())

    st.subheader('Estimated Coefficients:')
    st.write(mod_ARIMA.params)

st.write('Plot Diagnostics')
mod_ARIMA.plot_diagnostics(figsize=(12,9));
st.pyplot()


# Displaying residuals' diagnostics
st.markdown("**Residuals’ diagnostics of the ARIMA(2,1,2) model:**")
st.markdown("1. **Standardized Residual Plot:** The plot shows that the residuals have no trend with a variance that seems fairly constant over time, resembling the behavior of white noise.")
st.markdown("2. **Histogram Plot:** The plot shows the distribution of the residuals, which approaches a normal distribution, despite the unusual peak.")
st.markdown("3. **Q-Q plot:** The plot displays a fairly straight line that lies on y = x, even if this plot works best with a large number of observations.")
st.markdown("4. **Correlogram:** The plot shows no significant autocorrelation coefficients after lag 0, approximately looking like white noise.")
st.markdown("In conclusion, the model seems to be a good fit for the Canadian GDP time series.")

st.markdown('---')

st.write('Plot all the models performance')
fig, ax = plt.subplots(figsize=(10,6)) 
x = ['Linear Regression', 'Simple exponential smoothing', 'Holt+damped trend', 'ARIMA']
y = [float(AIC), model.aic, Model.aic, mod_ARIMA.aic]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Models')
ax.set_ylabel('AIC')
ax.set_ylim(0, 4000)

for index, value in enumerate(y):
    plt.text(x=index, y=value + 1, s=str(round(value,2)), ha='center')

plt.tight_layout()
st.pyplot()
st.write('The best model seems to be the Arima')

st.subheader('Forecasting for all countries with Double Exponential Smoothing')
gdp = gdp_sub_df[gdp_sub_df['Name_indicator'].str.contains('gdp'.upper())]
pivot_all = pd.pivot_table(data=gdp, index='Date', columns='Country', values='Value')

# Iterate over each country
for country in pivot_all.columns:
    st.markdown(f"<h3 style='text-align: center;'>{country} GDP</h3>", unsafe_allow_html=True)

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    piv = pivot_all[country].asfreq('AS')
    data_train = piv[:-10]
    data_test = piv[-10:]
    Mod = ETSModel(data_train, trend='add', damped_trend=True).fit()
    predictions = Mod.forecast(10)
    predic = Mod.get_prediction(start='2011-01-01', end='2020-01-01')
    df = predic.summary_frame()

    ax.plot(data_train, color='green', label='Training data')
    ax.plot(data_test, color='black', label='Testing data')
    ax.plot(predictions, color='red', label='Predictions')
    ax.plot(Mod.fittedvalues, color='orange', label='One-step-ahead fitted values')
    ax.fill_between(df.index, df['pi_lower'], df['pi_upper'], alpha=.1, color='crimson', label='95% CI')
    ax.set_xlabel('Date')
    ax.set_ylabel('GDP')
    ax.legend()
    
    # Display the plot
    st.pyplot(fig)

    # Summary Table
    tab = pd.DataFrame({'AIC': [Mod.aic], 'RMSE': [np.sqrt(Mod.mse)]}, index=[country])
    
    # Create an empty slot for the table
    table_slot = st.empty()
    
    # Fill the slot with the table
    table_slot.table(tab)