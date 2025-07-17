""" import of necessary modules and csv file as well as denote global parameters """
# import necessary module
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as TK

# import csv file for stocks and index
df_anz = pd.read_csv('ANZ.csv')
df_cba = pd.read_csv('CBA.csv')
df_nab = pd.read_csv('NAB.csv')
df_wbc = pd.read_csv('WBC.csv')
df_asx = pd.read_csv('ASX_200.csv')

# given risk-free rate is 3% (p.a.)
Rf = 0.03
T = 252

"""" data cleaning in ASX ("Notice that there's ',' in the numbers under 
'Close' where the entire field is determined as object by python") """

# function to convert columns with commas to float
def convert_to_float(column):
    return pd.to_numeric(column.str.replace(',', ''), errors='coerce')

# Apply to the 'Close' column (and any other relevant columns)
df_asx['Close'] = convert_to_float(df_asx['Close'])

""" data cleaning in Big4 banks and market data ("Notice that 'Date' type 
is not in the type 'datetime'") """

# function to convert raw date to datetime
def convert_to_date(column):
    return pd.to_datetime(column)

# Summarize my_portfolio and market_portfolio 
my_portfolio = [df_anz, df_cba, df_nab, df_wbc]
my_portfolio_s = []

for df in my_portfolio:
    df['Date'] = convert_to_date(df['Date'])
    df = df.sort_values(['Date'])  # Sort in ascending order (earliest to latest)
    my_portfolio_s.append(df)

df_asx['Date'] = convert_to_date(df_asx['Date'])
market_portfolio = df_asx.sort_values(['Date'])  # Sort in ascending order (earliest to latest)

# common function to compute daily return for each stocks and index
def compute_daily_return(df):
    df = df.copy()  # Avoid modifying original dataframe
    df['Daily Return'] = df['Close'].pct_change()  # More direct percentage change calculation
    return df

""" Preparation of return sequence """
# First calculate percentage returns for portfolio
big4_banks_return = []

for df in my_portfolio_s:
    df_r = compute_daily_return(df)
    # Keep only Date and Daily Return columns
    df_rr = df_r[['Date', 'Daily Return']].copy()
    
    # Set first return to 0 (no return on first day)
    df_rr.iloc[0, df_rr.columns.get_loc('Daily Return')] = 0   
    big4_banks_return.append(df_rr)

# Create equal-weighted portfolio returns
# Merge all banks on Date and calculate average return
portfolio_returns = big4_banks_return[0][['Date']].copy()
portfolio_returns['Daily Return'] = 0

for df in big4_banks_return:
    portfolio_returns = portfolio_returns.merge(df, on='Date', how='inner', suffixes=('', '_temp'))
    portfolio_returns['Daily Return'] += portfolio_returns['Daily Return_temp'] / 4
    portfolio_returns = portfolio_returns.drop('Daily Return_temp', axis=1)

my_portfolio_return = portfolio_returns.sort_values(['Date'])

# Convert portfolio percentage returns to cumulative dollar values starting from $10,000
my_portfolio_return_dollar = my_portfolio_return.copy()
my_portfolio_return_dollar['Daily Return'] = (1 + my_portfolio_return_dollar['Daily Return']).cumprod() * 10000

# Process individual bank returns in dollar for graph 2
big4_banks_return_dollar = []

for df in my_portfolio_s:
    df_r = compute_daily_return(df)
    df_rr = df_r[['Date', 'Daily Return']].copy()
    
    # Set first return to 0
    df_rr.iloc[0, df_rr.columns.get_loc('Daily Return')] = 0   
    
    # Convert to cumulative dollar value starting from $2500 per stock
    df_rr['Daily Return'] = (1 + df_rr['Daily Return']).cumprod() * 2500
    big4_banks_return_dollar.append(df_rr)

# Process market_portfolio_return (daily) in dollar
market_df = compute_daily_return(market_portfolio)
market_portfolio_return_dollar = market_df[['Date', 'Daily Return']].copy()

# Set first return to 0 and convert to cumulative dollar value starting from $10000
market_portfolio_return_dollar.iloc[0, market_portfolio_return_dollar.columns.get_loc('Daily Return')] = 0
market_portfolio_return_dollar['Daily Return'] = (1 + market_portfolio_return_dollar['Daily Return']).cumprod() * 10000
market_portfolio_return_dollar = market_portfolio_return_dollar.sort_values(['Date'])

# Process market_portfolio_return (daily) in percentage
market_portfolio_return = market_df[['Date', 'Daily Return']].copy()
market_portfolio_return.iloc[0, market_portfolio_return.columns.get_loc('Daily Return')] = 0
market_portfolio_return = market_portfolio_return.sort_values(by='Date')

""" Calculation of expected annualized return and annualised SD """
# Use percentage returns for calculations (excluding first day with 0 return)
portfolio_daily_returns = my_portfolio_return['Daily Return'].iloc[1:].reset_index(drop=True)  # Reset index to avoid alignment issues
market_daily_returns = market_portfolio_return['Daily Return'].iloc[1:].reset_index(drop=True)  # Reset index to avoid alignment issues

# Expected annualized return using simple multiplication by T (as per requirements)
expected_annualized_return_my_port = portfolio_daily_returns.mean() * T
expected_annualized_return_market = market_daily_returns.mean() * T

# Annualized standard deviation
annualized_sd_my_port = portfolio_daily_returns.std() * np.sqrt(T)
annualized_sd_market = market_daily_returns.std() * np.sqrt(T)

# Annualized variance for market (needed for beta calculation)
annualized_var_market = market_daily_returns.var() * T

# Generate summary dataframe
summary_risk_return = pd.DataFrame({
    'Measures': ['Market Portfolio', 'My Portfolio'], 
    'Expected Annualized Return (%)': [expected_annualized_return_market * 100, expected_annualized_return_my_port * 100],
    'Annualized Std Dev (%)': [annualized_sd_market * 100, annualized_sd_my_port * 100]
})

""" Beta, Jensen's Alpha, Sharpe Ratio, Information Ratio"""
# Annualized covariance
covariance_annualized = np.cov(portfolio_daily_returns, market_daily_returns)[0, 1] * T

# Beta
Beta = covariance_annualized / annualized_var_market

# Jensen's Alpha (annualized)
Alpha = expected_annualized_return_my_port - (Rf + Beta * (expected_annualized_return_market - Rf))

# Sharpe Ratio
SR_p = (expected_annualized_return_my_port - Rf) / annualized_sd_my_port

# Information Ratio
diff = portfolio_daily_returns - market_daily_returns  # Now both series have aligned indices
Tracking_error = diff.std() * np.sqrt(T)  # Annualized tracking error

IR_p = (expected_annualized_return_my_port - expected_annualized_return_market) / Tracking_error

summary_ratio = pd.DataFrame({
    'Measures': ['Beta', 'Jensen Alpha', 'Sharpe Ratio', 'Information Ratio'], 
    'Ratio': [Beta, Alpha, SR_p, IR_p]
})

# Print results for verification
print("Summary Risk Return:")
print(summary_risk_return)
print("\nSummary Ratios:")
print(summary_ratio)

""" Data Visualization"""
# Import date formatting tools
import matplotlib.dates as mdates

# Create the Tkinter window
root = TK.Tk()

# Create a figure for plotting
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

# Plot 1: Portfolio vs Market performance in dollars
axs[0].plot(my_portfolio_return_dollar['Date'], my_portfolio_return_dollar['Daily Return'], 
           label='Portfolio (Big 4 Banks)', color='blue', linewidth=1.2)

axs[0].plot(market_portfolio_return_dollar['Date'], market_portfolio_return_dollar['Daily Return'], 
           label='Market (ASX200)', color='orange', linewidth=1.2)

axs[0].set_title('Portfolio vs Market Performance (across 12 months, starting from $10,000 each)')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Portfolio Value ($)')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Format x-axis to show months
axs[0].xaxis.set_major_locator(mdates.MonthLocator())
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axs[0].tick_params(axis='x', rotation=45)

# Plot 2: Individual bank performance breakdown
banks = ['ANZ', 'CBA', 'NAB', 'WBC']
colors = ['purple', 'green', 'red', 'grey']
for i, df in enumerate(big4_banks_return_dollar):
    axs[1].plot(df['Date'], df['Daily Return'], label=banks[i], 
               color=colors[i], linewidth=1.2)

axs[1].set_title('Breakdown of Big 4 Banks Standalone Performance (across 12 months, starting from $2,500 each)')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Investment Value ($)')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# Format x-axis to show months for the second plot as well
axs[1].xaxis.set_major_locator(mdates.MonthLocator())
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axs[1].tick_params(axis='x', rotation=45)

# Table 1: Risk and Return Summary
axs[2].axis('tight')
axs[2].axis('off')
axs[2].set_title('Portfolio Summary - Risk and Return', fontsize=14, pad=20)
table_risk_return = axs[2].table(cellText=summary_risk_return.round(4).values, 
                                colLabels=summary_risk_return.columns, 
                                cellLoc='center', loc='center')
table_risk_return.auto_set_font_size(False)
table_risk_return.set_fontsize(9)
table_risk_return.scale(1.2, 1.5)

# Table 2: Financial Ratios
axs[3].axis('tight')
axs[3].axis('off')
axs[3].set_title('Portfolio Ratio Analysis', fontsize=14, pad=20)
table_ratios = axs[3].table(cellText=summary_ratio.round(4).values, 
                           colLabels=summary_ratio.columns, 
                           cellLoc='center', loc='center')
table_ratios.auto_set_font_size(False)
table_ratios.set_fontsize(9)
table_ratios.scale(1.2, 1.5)

# Adjust layout
plt.tight_layout(pad=2.0)

# Create canvas and embed in Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=TK.TOP, fill=TK.BOTH, expand=1)

# Add quit button
button = TK.Button(root, text='Quit', command=root.quit)
button.pack(side=TK.BOTTOM)

# Start the Tkinter main loop
TK.mainloop()