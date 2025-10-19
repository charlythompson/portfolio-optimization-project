'''
Project 4: Portfolio Optimization

This program calculates and visualizes the efficient frontier for a portfolio of two stocks.
It uses historical stock data to compute key metrics such as expected returns, risk, and Sharpe ratio.
Users can analyze default stocks (AAPL and WMT) or input custom stock tickers for analysis.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from weights import generate_weights # imported recursive function
import portfolio # importing the portfolio class 

# Current U.S. 1 Month Treasury Bill
RiskFreeRate = 0.043

# Creates a new column in the stock dataframe with the returns of the stock using percentage change
def process_stock_data(stock, column_name='Price', name='the stock'):
    stock_cleaned = stock.dropna(subset=[column_name])
    try:
        # Print cleaned data before and after cleaning
        answer = input(f'Do you want to see the cleaned data for {name}? (y/n): ').lower()
        if answer == 'y':
            print('Before cleaning:')
            print(stock[column_name])
            print()
            print('After cleaning:')
            print(stock_cleaned[column_name])
    except Exception as e:
        print(f"An error occurred while processing input: {e}")
    
    # Calculate returns, filling missing values with None
    stock['Returns'] = stock[column_name].pct_change(fill_method=None)
    # Drop the first row with NaN value in the 'Returns' column
    stock = stock.dropna(subset=['Returns'])
    return stock


def simulate_optimal_portfolio(mu, sigma, starting_value=1000 ,months=60, simulations=1000, seed=42, plot=True):
    np.random.seed(seed)
    
    monthly_return = mu / 12
    monthly_volatility = sigma / np.sqrt(12)
    
    sim = np.zeros((months, simulations))
    sim[0] = starting_value
    
    for t in range(1, months):
        random_returns = np.random.normal(loc=monthly_return, scale=monthly_volatility, size=simulations)
        sim[t] = sim[t - 1] * (1 + random_returns)

    average_value = sim[-1].mean()
    print(f"Final portfolio value in 5 years: ${average_value:.2f}")  

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(sim, alpha=0.1, color='blue')
        plt.title('Monte Carlo Simulations of Optimal Portfolio Returns')
        plt.xlabel('Months')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.show()

    return sim


# Function to calculate and plot portfolio metrics
def calculate_portfolio(stock1, stock2, stock1_name, stock2_name, RiskFreeRate):

    # To prevent issues from cleaing the code in  the process_stock_data function this aligns the indices of the 'Returns' columns
    stock1, stock2 = stock1.align(stock2, join='inner', axis=0)

    # Calculate expected returns, standard deviations, and variances
    ExpectedReturn1 = stock1['Returns'].mean() * 12
    ExpectedReturn2 = stock2['Returns'].mean() * 12
    ExpectedSD1 = stock1['Returns'].std() * np.sqrt(12)
    ExpectedSD2 = stock2['Returns'].std() * np.sqrt(12)
    variance1 = ExpectedSD1 ** 2
    variance2 = ExpectedSD2 ** 2

    # Calculate the correlation between the two stocks
    # corrcoef returns an array, so we need to access the [0][1] element for the correlation between stock1 and stock2
    stockCorrelation = np.corrcoef(stock1['Returns'], stock2['Returns'])[0][1]

    weights = generate_weights()  # Use the imported function
    
    # Initialize a dictionary to store results. 
    # Easier to store results in a dictionary and convert to DataFrame later
    results = {'weights': [], 'returns': [], 'risks': [], 'sharpes': []}
    # Loop through the weights and calculate the portfolio metrics using the Portfolio class
    for weight in weights:
        p = portfolio.Portfolio(weight[0], weight[1], ExpectedReturn1, ExpectedReturn2, variance1, variance2, stockCorrelation, ExpectedSD1, ExpectedSD2, RiskFreeRate)
        results['weights'].append(weight)
        results['returns'].append(p.expected_return())
        results['risks'].append(p.expected_std_dev())
        results['sharpes'].append(p.sharpe_ratio())

    # Convert results to a DataFrame for easier manipulation
    results_df = pd.DataFrame(results)
    # find the index of the minimum variance portfolio and maximum Sharpe ratio portfolio
    min_risk_index = results_df['risks'].idxmin()
    max_sharpe_index = results_df['sharpes'].idxmax()

    # Plotting the efficient frontier and extras to make it look nice
    plt.figure(figsize=(14, 8))
    plt.plot(results_df['risks'], results_df['returns'], label='Efficient Frontier', color='blue', linewidth=2)
    plt.scatter(results_df['risks'][min_risk_index], results_df['returns'][min_risk_index], color='red', label='Minimum Variance Portfolio (lowest risk)', s=150)
    plt.scatter(results_df['risks'][max_sharpe_index], results_df['returns'][max_sharpe_index], color='green', label='Optimal Portfolio', s=150)
    plt.text(results_df['risks'][min_risk_index], results_df['returns'][min_risk_index], 'MVP', fontsize=12, ha='right', color='red')
    plt.text(results_df['risks'][max_sharpe_index], results_df['returns'][max_sharpe_index], 'Optimal Portfolio', fontsize=12, ha='left', color='green')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Efficient Frontier of {stock1_name} and {stock2_name}', fontsize=16)
    plt.xlabel('Risk (Standard Deviation)', fontsize=14)
    plt.ylabel('Return', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Print the results
    print()
    print(f"Minimum Variance Portfolio: {results_df['weights'][min_risk_index][0] * 100:.2f}% {stock1_name}, {results_df['weights'][min_risk_index][1] * 100:.2f}% {stock2_name}")
    print(f"Optimal Portfolio: {results_df['weights'][max_sharpe_index][0] * 100:.2f}% {stock1_name}, {results_df['weights'][max_sharpe_index][1] * 100:.2f}% {stock2_name}")
    print()
    try:
        choice = input('Do you want to display the efficient frontier? (y/n): ').lower()  # Fixed typo
        if choice == 'y':
            plt.show()
        else:
            print()
    except Exception as e:
        print(f"An error occurred while processing input: {e}")
    
    try:
        print('Would you like to see the results in a CSV file? (y/n): ')
        choice = input().lower()
        if choice == 'y':
            results_df.to_csv('results.csv', index=False)
            print('Results saved to results.csv')
        else:
            print('Results not saved.')
    except Exception as e:
        print(f"An error occurred while processing input: {e}")

    try:
        print('Would you like to simulate future returns (y/n): ')
        choice = input().lower()
        if choice == 'y':
            value = int(input('How much would you like to invest?: '))
            # Make an instance of the optimal portfolio object:
            max_sharpe_portfolio = portfolio.Portfolio(
                results_df['weights'][max_sharpe_index][0], 
                results_df['weights'][max_sharpe_index][1], 
                ExpectedReturn1, 
                ExpectedReturn2, 
                variance1, 
                variance2, 
                stockCorrelation, 
                ExpectedSD1, 
                ExpectedSD2, 
                RiskFreeRate
            )
            mu = max_sharpe_portfolio.expected_return()
            sigma = max_sharpe_portfolio.expected_std_dev()

            simulate_optimal_portfolio(mu, sigma,value)
        else:
            print('Exiting')
    except Exception as e:
        print(f"An error occurred while processing input: {e}")


    

# Default stocks function using CSV files, as per the requirements
def defualtStocks(RiskFreeRate):
    # Load stock data from CSV files
    stock1 = pd.read_csv('AAPL data.csv')
    stock2 = pd.read_csv('WMT data.csv')
    # Process and clean the stock data
    stock1 = process_stock_data(stock1, 'Price', 'AAPL')
    stock2 = process_stock_data(stock2, 'Price', 'WMT')
    calculate_portfolio(stock1, stock2, 'AAPL', 'WMT', RiskFreeRate)

# Custom stocks function that takes user input for stock tickers and fetches data from Yahoo Finance
def customStocks(RiskFreeRate):
    try:
        stock1_ticker = input('Enter the first stock ticker: ').upper()
        stock2_ticker = input('Enter the second stock ticker: ').upper()
    except Exception as e:
        print(f"An error occurred while processing input: {e}")
        return
    # Fetch stock data from Yahoo Finance
    stock1 = yf.Ticker(stock1_ticker).history(period='5y',interval='1mo')
    stock2 = yf.Ticker(stock2_ticker).history(period='5y',interval='1mo')
    # Process  and clean the stock data
    stock1 = process_stock_data(stock1, 'Close', stock1_ticker)
    stock2 = process_stock_data(stock2, 'Close', stock2_ticker)

    calculate_portfolio(stock1, stock2, stock1_ticker, stock2_ticker, RiskFreeRate)


# Main menu
print('Select Option:')
print('1. AAPL and WMT')
print('2. Custom Stocks')
print('3. Exit')

# Get user input for the option
try:
    user_input = input()
    if user_input == '1':
        defualtStocks(RiskFreeRate)
    elif user_input == '2':
        customStocks(RiskFreeRate)
    elif user_input == '3':
        print('Exiting...')
        exit()
    else:
        print('Invalid input. Please enter a valid option.')
except Exception as e:
    print(f"An error occurred while processing input: {e}")