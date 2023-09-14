import numpy as np
import yfinance as yf
import datetime

bond_symbol = "^TNX"

# Create a Ticker object for the Treasury bond
bond_ticker = yf.Ticker(bond_symbol)

# Get historical data for the Treasury bond
bond_data = bond_ticker.history(period="1d")

# Extract the most recent yield (interest rate) from the bond data
risk_free_rate = bond_data["Close"].iloc[-1] / 100.0 

# Define the symbol for the S&P 100 Index (XEO)
symbol = "^XEO"

# Create a Ticker object
ticker = yf.Ticker(symbol)

# Get key information about the index
index_info = ticker.info

# Extract the relevant parameters
S = (index_info["bid"] + index_info["ask"]) / 2.0  # Average between bid and ask prices

# Prompt the user to choose a European call option for XEO
symbol = "^XEO"
ticker = yf.Ticker(symbol)

# Display available expiration dates for the options
expiration_dates = ticker.options
print("Available expiration dates:")
for i, date in enumerate(expiration_dates):
    print(f"{i + 1}. {date}")

# Ask the user to select an expiration date
expiration_choice = int(input("Select an expiration date (1, 2, 3, ...): ")) - 1
chosen_expiration_date = expiration_dates[expiration_choice]
today_datetime = datetime.datetime.now()  # Use datetime.datetime to avoid conflicts
expiration_date_parts = [int(part) for part in chosen_expiration_date.split("-")]
expiration_datetime = datetime.datetime(expiration_date_parts[0], expiration_date_parts[1], expiration_date_parts[2])
T = (expiration_datetime - today_datetime).days

# Fetch call option data for the chosen expiration date
option_chain = ticker.option_chain(chosen_expiration_date)
call_options = option_chain.calls

print(call_options)

call_option_choice = int(input("Select a call option by number (0, 1, 2, ...): "))
selected_contract = call_options.iloc[call_option_choice]

# Extract the chosen call option's attributes
K = selected_contract['strike']
last_price = selected_contract['lastPrice']
vol = selected_contract['impliedVolatility']
r = risk_free_rate

# Display the gathered parameters
print(f"Current Index Level (S): {S}")
print(f"Time to Expiration (T): {T} years")
print(f"Risk-Free Interest Rate (r): {r}")
print(f"Volatility (vol): {vol}")

# Monte Carlo Parameters
N = 252  # Number of time steps
M = 10000  # Number of simulations

# Precompute constants
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
CT = np.maximum(0, ST - K)
C0 = np.exp(-r * T) * np.sum(CT[-1]) / M
sigma = np.sqrt(np.sum((CT[-1] - C0) ** 2) / (M - 1))
SE = sigma / np.sqrt(M)

# Print the results, including the chosen call option details
print(f"Chosen Call Option: Strike Price = {K}, Last Price = {last_price}")
print("Monte Carlo Results:")
print("Call value is ${0:.2f} with SE +/- {1:.2f}".format(C0, SE))
