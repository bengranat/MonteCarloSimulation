import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf

# Define the SP100 index symbol
sp100_symbol = '^SP100'  # This symbol represents the S&P 100 index

# Define the number of Monte Carlo simulations and the timeframe in days
mc_sims = 400
T = 106

# Define the start and end dates for historical data retrieval
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=300)

# Function to retrieve SP100 index data
def get_sp100_data(symbol, start, end):
    sp100_data = yf.download(symbol, start=start, end=end)
    sp100_close = sp100_data['Close']
    return sp100_close

# Retrieve SP100 index prices
sp100_prices = get_sp100_data(sp100_symbol, start_date, end_date)
initial_sp100_price = sp100_prices.iloc[-1]

# Initialize arrays
sp100_simulated = np.full(shape=(T, mc_sims), fill_value=0.0)

# Monte Carlo Simulation for SP100
for m in range(mc_sims):
    daily_returns = np.random.normal(0, sp100_prices.pct_change().std(), T)
    simulated_prices = initial_sp100_price * np.cumprod(1 + daily_returns)
    sp100_simulated[:, m] = simulated_prices

# Calculate the average of simulated prices at each time step
average_prices = np.mean(sp100_simulated, axis=1)

# Plotting SP100 simulations and the average price path
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(sp100_simulated)
plt.ylabel('SP100 Index Value')
plt.xlabel('Days')
plt.title('MC simulation of SP100 index price path')

plt.subplot(2, 1, 2)
plt.plot(average_prices, color='red')
plt.ylabel('Average SP100 Index Value')
plt.xlabel('Days')
plt.title('Average SP100 Index Price Path')

plt.tight_layout()
plt.show()

# Prompt the user for a value t
t = int(input("Enter a value for t (time step): "))

# Calculate and print the average price at time step t for all Monte Carlo simulations
average_price_at_t = np.mean(sp100_simulated[t - 1, :])
print(f'Average SP100 Index Price at Time Step {t}: {average_price_at_t:.2f}')
