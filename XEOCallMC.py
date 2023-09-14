import yfinance as yf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy import stats

# Define the SPX index symbol
symbol = "^XEO"

# Fetch the options data for the SPX index
spx = yf.Ticker(symbol)

# Get the expiration dates available for options up to 2024
expiration_dates = spx.options
expiration_dates = [date for date in expiration_dates if int(date.split("-")[0]) <= 2023]

print("Available expiration dates up to 2023:")
for i, date in enumerate(expiration_dates):
    print(f"{i + 1}: {date}")

# Prompt the user to choose an expiration date
expiration_choice = int(input("Choose an expiration date (enter the corresponding number): ")) - 1
expiration_date = expiration_dates[expiration_choice]

# Fetch the options data for the chosen expiration date
options = spx.option_chain(expiration_date)

# Get the current SPX index price
current_price = spx.history(period="1d")["Close"].iloc[-1]
print(f"Current SPX index price: {current_price}")

# Retrieve real values for volatility (implied volatility) and strike prices
vol = options.calls['impliedVolatility'].values  # Array of implied volatilities
strike_prices = options.calls['strike'].values  # Array of strike prices
T = (datetime.datetime.strptime(expiration_date, "%Y-%m-%d") - datetime.datetime.now()).days / 365.0  # Time to expiration

# Prompt the user to enter a strike price
user_strike = float(input("Enter a strike price: "))

# Find the contract whose strike price is closest to the user input
closest_strike_idx = np.argmin(np.abs(strike_prices - user_strike))
K = strike_prices[closest_strike_idx]

# Get the bid and ask prices for the selected call option
bid_price = options.calls['bid'].values[closest_strike_idx]
ask_price = options.calls['ask'].values[closest_strike_idx]

# Check if the difference between bid and ask price is 0, and if so, use "last price" from Yahoo Finance
if ask_price - bid_price == 0:
    print("Bid-Ask price difference is 0. Using 'last price' from Yahoo Finance options.")
    last_price = options.calls['lastPrice'].values[closest_strike_idx]
    market_price = last_price
else:
    # Calculate the market price as the midpoint between bid and ask
    market_price = (bid_price + ask_price) / 2.0

print(f"Bid Price: {bid_price}")
print(f"Ask Price: {ask_price}")
print(f"Market price of the selected call option (Closest Strike Price: {K}): {market_price}")

# Define the relevant variables from XEO option data
S = current_price  # Current SPX index price
r = 0.01  # Risk-free rate (%)
N = 1  # Number of time steps
M = 1000000  # Number of simulations
strike_price = K  # Selected strike price

# Calculate constants
dt = T / N
nudt = (r - 0.5 * vol ** 2) * dt
volsdt = vol * np.sqrt(dt)
lnS = np.log(S)

# Monte Carlo Method
Z = np.random.normal(size=(N, M))
delta_lnSt = nudt + volsdt * Z
lnSt = lnS + np.cumsum(delta_lnSt, axis=0)
lnSt = np.concatenate((np.full(shape=(1, M), fill_value=lnS), lnSt))

# Compute Expectation and SE
ST = np.exp(lnSt)
CT = np.maximum(0, ST - strike_price)
C0 = np.exp(-r * T) * np.sum(CT[-1]) / M
sigma = np.sqrt(np.sum((CT[-1] - C0) ** 2) / (M - 1))
SE = sigma / np.sqrt(M)
print("Call value is ${0} with SE +/- {1}".format(np.round(C0, 2), np.round(SE, 2)))

# Calculate the total payoff including the difference between market price and call value
total_payoff = C0 - market_price

# Visualisation of Convergence
x1 = np.linspace(total_payoff - 3 * SE, total_payoff + 3 * SE, 100)
s1 = stats.norm.pdf(x1, total_payoff, SE)
plt.fill_between(x1, s1, color='cornflowerblue', label='Probability Density')
plt.plot([total_payoff, total_payoff], [0, max(s1) * 1.1], 'k', label='Total Payoff')
plt.plot([call_values[0], call_values[0]], [0, max(s1) * 1.1], 'r', label='Call Value (Theoretical)')
plt.plot([market_price, market_price], [0, max(s1) * 1.1], 'k--', label='Market Price')
plt.xlabel("Value")
plt.ylabel("Probability")
plt.title("Convergence of Total Payoff vs. Call Value vs. Market Price")
plt.legend()
plt.show()