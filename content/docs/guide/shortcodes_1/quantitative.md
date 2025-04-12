---
title: Quantitative Finance
linkTitle: 11- Quantitative Finance
---

#  What are the Different Types of Mathematics Found in Quantitative Finance?

The real-world subject of quantitative finance uses tools from many branches of mathematics. And financial modelling can be approached in a variety of different ways. For some strange reason the advocates of different branches of mathematics get quite emotional when discussing the merits and demerits of their methodologies and those of their ‘**opponents**.’ Is this a territorial thing? What are the pros and cons of martingales and differential equations? What is all this fuss, and will it end in tears before bedtime?
Here’s a list of the various approaches to modelling and a selection of useful tools. The distinction between a ‘**modelling approach**’ and a ‘**tool**’ will start to become clear.


<span style="color:red">**Modelling approaches**</span>

- **Probabilistic** : One of the main assumptions about the financial markets, at least as far as quantitative finance goes, is that asset prices are random. We tend to think of describing financial variables as following some random path, with parameters describing the growth of the asset and its degree of randomness. We effectively model the asset path via a specified rate of growth, on average, and its deviation from that average. This approach to modelling has had the greatest impact over the last 30 years, leading to the explosive growth of the derivatives markets.

- **Deterministic**: The idea behind this approach is that our model will tell us everything about the future. Given enough data, and a big enough brain, we can write down some equations or an algorithm for predicting the future. Interestingly, the subjects of dynamical systems and chaos fall into this cat- egory. And, as you know, chaotic systems show such sensitivity to initial conditions that predictability is in practice impossible. This is the ‘butterfly effect,’ that a butterfly flap- ping its wings in Brazil will ‘cause’ rainfall over Manchester. (And what doesn’t!) A topic popular in the early 1990s, this has not lived up to its promises in the financial world.

- **Discrete**: difference equations: Discrete means that asset prices and/or time can only be incremented in finite chunks, whether a dollar or a cent, a year or a day.

- **Continuous**: differential equations:  Continuous means that no such lower increment exists. The mathematics of continuous processes is often easier than that of discrete ones. But then when it comes to number crunching you have in any case to turn a continuous model into a discrete one.

For an important example: In discrete models we end up with difference equations. An example of this is the binomial model for option pricing. Time progresses in finite amounts, the time step. In continuous models we end up with differential equations. The equivalent of the binomial model in discrete space is the **Black–Scholes model**, which has continuous asset price and continuous time. Whether **Binomial** or **Black–Scholes**, both of these mod- els come from the probabilistic assumptions about the financial world.


<span style="color:red">**Usefull tools**</span>

- **Simulations**: If the financial world is random then we can experiment with the future by running simulations. For example, an asset price may be represented by its average growth and its risk, so let’s simulate what could happen in the future to this random asset. If we were to take such an approach we would want to run many, many simulations. There’d be little point in running just the one; we’d like to see a range of possible future scenarios. *Simulations can also be used for non-probabilistic problems. Just because of the similarities between mathematical equations, a model derived in a deterministic framework may have a probabilistic interpretation*.

- **Discretization methods**: The complement to simulation methods, and there are many types of these. The best known are the finite-difference methods which are discretizations of continu- ous models such as Black–Scholes.
- **Approximations**: In modelling we aim to come up with a solution representing something meaningful and useful, such as an option price. Unless the model is really simple, we may not be able to solve it easily. This is where approximations come in. A complicated model may have approximate solutions. And these approximate solutions might be good enough for our purposes.

- **Asymptotic analysis**: this is an incredibly useful technique, used in most branches of applicable mathematics, but until recently almost unknown in finance. The idea is simple: find approximate solutions to a complicated problem by exploiting parameters or variables that are either large
or small, or special in some way. For example, there are simple approximations for vanilla option values close to expiry.

- **Series solutions**: If your equation is linear (and they almost all are in quantitative finance) then you might be able to solve a particular problem by adding together the solutions of other problems. Series solutions are when you decompose the solu- tion into a (potentially infinite) sum of simple functions, such as sines and cosines, or a power series. This is the case, for example, with barrier options having two barriers, one below the current asset price and the other above.

- **Green’s functions** :  Green's functions are mathematical tools used in physics and engineering to solve differential equations that describe various physical phenomena, such as heat conduction, fluid flow, or electromagnetic fields.  This is a very special technique that only works in certain situations. The idea is that solutions to some difficult problems can be built up from solutions to special cases of a similar problem.

## Present value and future value of money

In finance, the present value (PV) and future value (FV) of money are essential concepts that help in evaluating the time value of money. The time value of money is the idea that money available at the present time is worth more than the same amount in the future, due to its potential earning capacity.

Present value (PV): This is the present value of $x with the consideration interest rate $r$ and the $x$ cash flow in the future with respect to the number of year. This defines how much of a future sum of money is worth today given a specific rate of interet
$$\frac{x}{(1+r)^{n}} $$

Future value: is the value of a current asset at a specified date in the future based on an assumed rate of growth over time

$$x(1+r)^{n}$$

If we have to deal with the continous mode with diferential equation. The interest I receive must be proportional to the actual **x(t)** amount I have and the **r** interest rate and the **dt** time step
$$x(t)=x(0)e^{rt}$$


```python 
from math import exp


def future_discrete_value(x, r, n):
    return x*(1+r)**n


def present_discrete_value(x, r, n):
    return x*(1+r)**-n


def future_continuous_value(x, r, t):
    return x*exp(r*t)


def present_continuous_value(x, r, t):
    return x*exp(-r*t)


if __name__ == '__main__':

    # value of investment in dollars
    x = 100
    # define the interest rate (r)
    r = 0.05
    # duration (years)
    n = 5

    print("Future value (discrete model) of x: %s" % future_discrete_value(x, r, n))
    print("Present value (discrete model) of x: %s" % present_discrete_value(x, r, n))
    print("Future value (continuous model) of x: %s" % future_continuous_value(x, r, n))
    print("Present values (continuous model) of x: %s" % present_continuous_value(x, r, n))

```
 

- Future value (discrete model) of x: 127.62815625000003
- Present value (discrete model) of x: 78.35261664684589
- Future value (continuous model) of x: 128.40254166877415
- Present values (continuous model) of x: 77.8800783071405


### Stock and shares 

Stock price rise and fall due to the fluctutation of supply and demand; if more/fewer people want to buy a given stocks the market price will  increase/ decrease .
There may be growth in the value of the stock that can be realized if you sell a given stock\
But still a risky given-so-called diividents.
there are so-called dividends. These are payments paid
out every quarter or every six months to the shareholders
the amount of dividend usually depends on the
profitability of the given company.




<span style="color:red">Meassuring the risk of stock - Volatility</span>

- Statistical measure of the dispersion of returns for a given security
which is the amount of uncertainty (or risk) about the size of
changes in the value of a given security (stock, bond etc.)
- We can measure volatility with standard deviation
or variance between return's of the same security
Higher the volality the risker the security
- We can use the Capital Asset Pricing Model (CAPM)
with the $\beta$ value to approximate volatility


## Commodities ##
Commodities are raw products such as **gold, oil** or **natural gas** .

Investing into commodities is not that simple so that commodieties such as oil is extremely volatile. This is why there are **future contracts**

Commodities prices are usually very similar to **Random walk**. Commodity prices rise and fall due to the fluctuation of the supply and demand. *If more people want to buy a given commodity then the market price will increase*

**Future contract** are made in an attempt tby producers and suppliers of commodities to advoid market volality. They negociate the price of a given commodity in the future.

The commodity prices typically rise when inflation is accelerating ( *Commodities such as oild or gold usually offer protection from the effect of inflation*). Thus, commodities may offer protection against the negative effect of inflation .

## Currencies and the Forex 

In finance an **exchange rate** is the rate at which one national currency will be exchanged for another

It tells you how much a given currency worth in another currency

*Governments and central banks can influence currencies and exchange rates*

| Country       | Currency         | Code | Exchange Rate (USD) |
|---------------|------------------|------|---------------------|
| United States | US Dollar        | USD  | 1.0000              |
| European Union| Euro             | EUR  | 0.8500              |
| United Kingdom| British Pound    | GBP  | 0.7300              |
| Japan         | Japanese Yen     | JPY  | 110.0000            |
| Canada        | Canadian Dollar  | CAD  | 1.2100              |
| Australia     | Australian Dollar| AUD  | 1.3000              |

This concise table provides a snapshot of the current exchange rates for some of the world's major currencies, including the US Dollar (USD), Euro (EUR), British Pound (GBP), Japanese Yen (JPY), Canadian Dollar (CAD), and Australian Dollar (AUD). These currencies play a significant role in the foreign exchange (forex) market, as they are among the most traded and liquid currencies globally.The table is organized with the country, currency name, currency code, and exchange rate relative to the US Dollar. By presenting the data in a tabular format, users can quickly compare the value of one currency to another and assess the relative strength or weakness of a particular currency.It is essential to note that the forex market is constantly fluctuating due to various factors, including economic data releases, geopolitical events, and changes in monetary policy. As a result, exchange rates in this table may change over time and should be regularly updated to reflect the latest market conditions.To sum up, this table serves as a handy reference for individuals and businesses engaged in international trade, travel, or investment. Keeping track of these key exchange rates can provide valuable insights into the global economy and help inform financial decisions.

 <span style="color:red">Why do exchange rates fluctuate ?</span>
 
 Exchange rates rise and fall due to the fluctuation of supply and demand. That is the reaon why the exchanged rates are usually very similar to a **Random Walk**. If more people want to buy a given currency then its market price will increase.

 Factors affecting exchange rates 
 - Interest rates: is a major factor that can be manipulated by the central bank of a country. Investors will lend money to the banks of the given country for higher returns.

- Money supply; created by the central bank by printing too much currency may trigger inflation. Investors do not like inflation so they will leave the currency that can push the value of a currency down.

- Fincacial stability and economic growth of a give country have a huge impact on the value of the exchange rate.
  
  
<span style="color:red">What is Arbitrage ?</span>

Arbitrage is making a sure profit in excess of the risk-free rate of return. In the language of quantitative finance we can say that an arbitrage opportunity is a portfolio of zero value today which is of positive value in the future with positive probability, and of negative value in the future with zero probability.
The assumption that there are no arbitrage opportunities in the market is fundamental to classical finance theory. This idea is popularly known as ‘there’s no such thing as a free lunch.

**Example**: An at-the-money European call option with a strike of $100 and an expiration of six months is worth $8. A European put with the same strike and expiration is worth $6. There are no dividends on the stock and a six-month zero-coupon bond with a principal of $100 is worth $97.

Buy the call and a bond, sell the put and the stock, which will bring in $(−8−97+6+100)=$1. At expiration this portfolio will be worthless regardless of the final price of the stock. You will make a profit of $1 with no risk. This is arbitrage. 

*The principle of no arbitrage is one of the foundations of classical finance theory. In derivatives theory it is assumed during the derivation of the binomial model option-pricing algorithm and in the Black–Scholes model.*

## Long and short positions

**Long Position** in a security means that you owns the security. Investors maintain long positions in the expectation that the stock will increase in the value in the future. *Investors can make profit by maitaining a long position*

**Short Position** in a security means that you sell the security.Investors maintain short positions in the expectation that the stock will decrease in the value in the future. *Investors can make profit by maitaining a short position*. Short selling meaning that you sell something you do not actually own

 <span style="color:red"> What are the risks with Short and Long positions ?</span>
 - Shorting is **much risker** than opening long positions
 - When you open long position then you maximum possible loss is 100% so you may lose your entire initial investment
 - With short selling there is no limit to how much you can lose because there is no limit for the given stock to increase in value

 


# Bond Theory

Bond theory in finance refers to the principles and frameworks used to analyze and evaluate bonds as a form of investment. Bonds are debt securities issued by entities such as governments or corporations to raise capital. Investors who purchase bonds are essentially lending money to the issuer in exchange for periodic interest payments (known as the coupon) and the return of the principal (face value) at the end of the bond's term (maturity date).

## Yields and yield to maturity
Yield refers to the annual return on investment that an investor can expect to earn from holding a bond. It takes into account the bond's purchase price, face value, coupon payments, and time to maturity. It defines how much money your investment is generating $$\frac{\text{annual coupon amount} }{\text{bond price} } $$

The yield to maturity of a bond is the internal rate of return (overall interest rate) earned by an investor who buys thebond
at **t** today at the **V** market price 

- We assume that the bond is held until **T** maturity.
- all **$C_{i}$** coupons and **P** principal payments are made on schedule

Therefore it is the yeild maturity **y** interest rate that will make the present value of the cash flows from the investment equal to the price(cost) of the investment
$$\text{v} =\sum^{N}_{i=1} C_{i}e^{-y\left( t_{i}-t\right)  }\  +\  Pe^{-y(T-t)}$$

- **v** is discounting everything back to the **t** present gives the current **v** price
- $\sum^{N}_{i=1}C_{i}e^{-y\left( t_{i}-t\right)  }$  : to calculate the present value of the $C_{i}$ coupon payments.
- $ Pe^{-y(T-t)} $  : to calculate the $P$ present value of the pricipal amount.

Longer bonds pays investors higher interest rate- investors expect more yield in return for loaning their money for a longer period of time


## Interest rates and bonds 
Bonds and market interest rates are negatively correlated when the cost of borrowing money rises bond prices usually fall and vice-versa. Of course if the market interest rate is high enough than it better to lend money to the bank rather than buying bonds

Coupon bonds and zero-coupon bonds are two types of bonds that differ in their payment structures. Here's an overview of each type and how to calculate their values:
- Coupon Bond: is a debt security that pays periodic interest payments (coupons) to the bondholder throughout its term. The issuer also repays the face (par) value of the bond upon maturity.
$$\sum^{n}_{i=1} \frac{c}{(1+r)^{i}} +\frac{x}{(1+r)^{n}} $$
- A zero-coupon bond is a debt security that does not pay any periodic interest payments. Instead, it is sold at a discount to its face value, and the bondholder receives the face value at maturity. The difference between the purchase price and the face value represents the interest earned on the bond:
$$\frac{x}{(1+r)^{n}} $$

- c the present value of coupon payment
- r interest rate
- n maturity (years)

## Bonds Implementation


```python 

class CouponBond:

    def __init__(self, principal, rate, maturity, interest_rate):
        self.principal = principal
        self.rate = rate / 100
        self.maturity = maturity
        self.interest_rate = interest_rate / 100

    def present_value(self, x, n):
        return x / (1+self.interest_rate)**n

    def calculate_price(self):

        price = 0

        # discount the coupon payments
        for t in range(1, self.maturity+1):
            price = price + self.present_value(self.principal * self.rate, t)

        # discount principle amount
        price = price + self.present_value(self.principal, self.maturity)

        return price


if __name__ == '__main__':

    bond = CouponBond(1000, 10, 3, 4)
    print("Bond price: %.2f" % bond.calculate_price())
```
Bond price: 1166.51


# Markowitz-Model (Modern Portfolio Theory)

### <span style="color:red"> Definition</span>
The Markowitz Model, also known as Modern Portfolio Theory (MPT) or Mean-Variance Optimization, is an investment model developed by Harry Markowitz in 1952. It is a mathematical framework that aims to maximize the expected return of a portfolio for a given level of risk, or equivalently, minimize risk for a given level of expected return. The model assumes that investors are rational and risk-averse, and that their investment decisions are solely based on the expected return and risk of the assets

 ### <span style="color:red"> Example</span>
Should you put all your money in a stock that has low risk
but also low expected return, or one with high expected
return but which is far riskier? Or perhaps divide your
money between the two. Modern Portfolio Theory addresses
this question and provides a framework for quantifying and
understanding risk and return.

### <span style="color:red"> Explanation</span>
In MPT the return on individual assets are represented by normal distributions with certain mean and standard devi- ation over a specified period. So one asset might have an annualized expected return of 5% and an annualized standard deviation (volatility) of 15%. Another might have an expected return of −2% and a volatility of 10%. Before Markowitz, one would only have invested in the first stock, or perhaps sold the second stock short. Markowitz showed how it might be possible to better both of these simplistic portfolios by taking into account the correlation between the returns on these stocks.

In the MPT world of N assets there are $2N+\frac{N(N-1)}{2} $ parameters: expected return, one per stock; standard deviation, one per stock; correlations, between any two stocks (choose two from N without replacement, order unimportant). To Markowitz all investments and all portfolios should be compared and contrasted via a plot of expected return versus risk, as measured by standard deviation. If we write $\mu_{A} $ to represent the expected return from investment or portfolio A (and similarly for B, C, etc.) and $\sigma_{B}$ for its standard deviation then investment/portfolio A is at least as good as B if
$$\mu_{A} \geq \mu_{B} \  \text{and} \  \sigma_{A} \leq \sigma_{B}   $$



The mathematics of risk and return is very simple. Consider a portfolio, $\prod $, of $N$ assets, with $W_i$ the fraction of wealth invested in the $i^{th}$ asset. The expected return is then

$$\mu_{\prod } =\sum^{N}_{i=1} W_{i}\mu_{i} $$

and the standard deviation of the return, the risk, is

$$\sigma_{\Pi } =\sqrt{\sum^{N}_{i=1} \sum^{N}_{j=1} W_{i}W_{j}\rho_{ij} \sigma_{i} \sigma_{j} } $$

where $\rho_{ij}$ is the correlation between the $i^{th}$ and $j^{th}$ investments, with $\rho_{ij}=1$



Markowitz showed how to optimize a portfolio by finding the $W’s$ giving the portfolio the greatest expected return for a prescribed level of risk. The curve in the risk-return space with the largest expected return for each level of risk is called the **efficient frontier**.

## <span style="color:red"> Markowitz-Model Implementation</span>

```python 
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

# on average there are 252 trading days in a year
NUM_TRADING_DAYS = 252
# we will generate random w (different portfolios)
NUM_PORTFOLIOS = 10000

# stocks we are going to handle
stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

# historical data - define START and END dates
start_date = '2010-01-01'
end_date = '2017-01-01'


def download_data():
    # name of the stock (key) - stock values (2010-1017) as the values
    stock_data = {}

    for stock in stocks:
        # closing prices
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(stock_data)


def show_data(data):
    data.plot(figsize=(10, 5))
    plt.show()


def calculate_return(data):
    # NORMALIZATION - to measure all variables in comparable metric
    log_return = np.log(data / data.shift(1))
    return log_return[1:]


def show_statistics(returns):
    # instead of daily metrics we are after annual metrics
    # mean of annual return
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)


def show_mean_variance(returns, weights):
    # we are after the annual return
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()
                                                            * NUM_TRADING_DAYS, weights)))
    print("Expected portfolio mean (return): ", portfolio_return)
    print("Expected portfolio volatility (standard deviation): ", portfolio_volatility)


def show_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov()
                                                          * NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)


def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()
                                                            * NUM_TRADING_DAYS, weights)))
    return np.array([portfolio_return, portfolio_volatility,
                     portfolio_return / portfolio_volatility])


# scipy optimize module can find the minimum of a given function
# the maximum of a f(x) is the minimum of -f(x)
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]


# what are the constraints? The sum of weights = 1 !!!
# f(x)=0 this is the function to minimize
def optimize_portfolio(weights, returns):
    # the sum of weights is 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # the weights can be 1 at most: 1 when 100% of money is invested into a single stock
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns
                                 , method='SLSQP', bounds=bounds, constraints=constraints)


def print_optimal_portfolio(optimum, returns):
    print("Optimal portfolio: ", optimum['x'].round(3))
    print("Expected return, volatility and Sharpe ratio: ",
          statistics(optimum['x'].round(3), returns))


def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], 'g*', markersize=20.0)
    plt.show()


if __name__ == '__main__':
    dataset = download_data()
    show_data(dataset)
    log_daily_returns = calculate_return(dataset)
    # show_statistics(log_daily_returns)

    pweights, means, risks = generate_portfolios(log_daily_returns)
    show_portfolios(means, risks)
    optimum = optimize_portfolio(pweights, log_daily_returns)

```
![uploads](/uploads/outputb.png)
![uploads](/uploads/outputc.png)

The dots represent different **w** weights of a given portfolio containing multiple stocks (So different porfolios ). 

The investor interested in 
- The maximum return (means) given a fixed risk level (so volatility)
- Minimum risk given a fixed return

These portfolios make up the so-called ** efficient -frontier**. This is the main feature of **Markowitz model** the investor can decide the risk of the expected return


# Capital Asset Pricing Model


### <span style="color:red"> Definition</span>
**The Capital Asset Pricing Model (CAPM)** relates the returns on individual assets or entire portfolios to the return on the market as a whole. It introduces the concepts of specific risk and systematic risk. **Specific risk** is unique to an individual asset, systematic risk is that associated with the market. In CAPM investors are compensated for taking **systematic risk** but not for taking specific risk. This is because specific risk can be diversified away by holding many different assets.


 ### <span style="color:red"> Example</span>
A stock has an expected return of 15% and a volatility of 20%. But how much of that risk and return are related to the market as a whole? The less that can be attributed to the behaviour of the market, the better will that stock be for diversification purposes.

### <span style="color:red"> Explanation</span>
CAPM simultaneously simplified Markowitz’s **Modern Portfolio Theory (MPT)**, made it more practical and introduced the idea of specific and systematic risk. Whereas MPT has arbitrary correlation between all investments, **CAPM**, in its basic form, only links investments via the market as a whole. CAPM is an example of an equilibrium model, as opposed to a no-arbitrage model such as **Black–Scholes**.

The mathematics of CAPM is very simple. We relate the random return on the ith investment, $R_i$, to the random return on the market as a whole (or some representative index),$R_M$ by
$$R_{i}=\alpha_{i} +\beta_{i} R_{M}+\varepsilon_{i}   $$

The $\varepsilon_{i}$ is random with zero mean and standard deviation $e_i$ and uncorrelated with the market return $R_M$ and the other $e_j$. There are three parameters associated with each asset $\alpha_{i}$, $\beta_{i}$ and $e_i$. In this representation we can see that the return on an asset can be decomposed into three parts: a constant drift; a random part common with the index; a random part uncorrelated with the index,$\varepsilon_{i}$. The random part $\varepsilon_{i}$ is unique to the ith asset. Notice how all the assets are related to the index but are otherwise completely uncorrelated

```python 
import numpy as np
import pandas as p
import yfinance as yf
import matplotlib.pyplot as plt

# market interest rate
RISK_FREE_RATE = 0.05
# we will consider monthly returns - and we want to calculate the annual return
MONTHS_IN_YEAR = 12

class CAPM:

    def __init__(self, stocks, start_date, end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        data = {}

        for stock in self.stocks:
            ticker = yf.download(stock, self.start_date, self.end_date)
            data[stock] = ticker['Adj Close']

        return pd.DataFrame(data)

    def initialize(self):
        stock_data = self.download_data()
        # we use monthly returns instead of daily returns
        stock_data = stock_data.resample('M').last()

        self.data = pd.DataFrame({'s_adjclose': stock_data[self.stocks[0]],
                                  'm_adjclose': stock_data[self.stocks[1]]})

        # logarithmic monthly returns
        self.data[['s_returns', 'm_returns']] = np.log(self.data[['s_adjclose', 'm_adjclose']] /
                                                       self.data[['s_adjclose', 'm_adjclose']].shift(1))

        # remove the NaN values
        self.data = self.data[1:]

    def calculate_beta(self):
        # covariance matrix: the diagonal items are the variances
        # off diagonals are the covariances
        # the matrix is symmetric: cov[0,1] = cov[1,0] !!!
        covariance_matrix = np.cov(self.data["s_returns"], self.data["m_returns"])
        # calculating beta according to the formula
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        print("Beta from formula: ", beta)

    def regression(self):
        # using linear regression to fit a line to the data
        # [stock_returns, market_returns] - slope is the beta
        beta, alpha = np.polyfit(self.data['m_returns'], self.data['s_returns'], deg=1)
        print("Beta from regression: ", beta)
        # calculate the expected return according to the CAPM formula
        # we are after annual return (this is why multiply by 12)
        expected_return = RISK_FREE_RATE + beta * (self.data['m_returns'].mean()*MONTHS_IN_YEAR
                                                   - RISK_FREE_RATE)
        print("Expected return: ", expected_return)
        self.plot_regression(alpha, beta)

    def plot_regression(self, alpha, beta):
        fig, axis = plt.subplots(1, figsize=(20, 10))
        axis.scatter(self.data["m_returns"].to_numpy(), self.data['s_returns'].to_numpy(),
                     label="Data Points")
        axis.plot(self.data["m_returns"].to_numpy(), beta * self.data["m_returns"].to_numpy() + alpha,
                  color='red', label="CAPM Line")
        plt.title('Capital Asset Pricing Model, finding alpha and beta')
        plt.xlabel('Market return $R_m$', fontsize=18)
        plt.ylabel('Stock return $R_a$')
        plt.text(0.08, 0.05, r'$R_a = \beta * R_m + \alpha$', fontsize=18)
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    capm = CAPM(['IBM', '^GSPC'], '2010-01-01', '2017-01-01')
    capm.initialize()
    capm.calculate_beta()
    capm.regression()


```

- [*********************100%***********************]  1 of 1 completed
- [*********************100%***********************]  1 of 1 completed
- Beta from formula:  0.7135097171981648
- Beta from regression:  0.7135097171981654
- Expected return:  0.09011312101583244


![uploads](/uploads/outputd.png)


# Random Behavior in Finance


## Types of Analysis

There are three main types of analysis which as fundamental analysis, there is technical analysis and finally, quantitative analysis.

- <span style="color:red"> Fundamental analysis</span> is about the in-depth study of a given company.
There are several factors to consider
such as the management teams, products and services, balance sheets, income statements
and ...We try to predict by analyzing these factors whether the stock is undervalued or not
based on the intrinsic value of the company.So, for example, if we are using machine learning approaches, for example, logistic
regression or support vector classifiers or deep neural networks, basically we are analysing
historical data. We are looking for patterns in the past that will repeat themselves in the future.
And if we come to the conclusion that the same pattern, then we can make a prediction based on historical data what's going to happen in the future.

-  <span style="color:red"> Technical analysis</span> which is the opposite of fundamental analysis.
This approach doesn't care about the company.
It assumes that all the information is contained within its stock.
So technical analysis is about analyzing historical data.

<span style="color:red"> Quantitative analysis</span> has an assuption: all finacial quantities such as stock price or interest rates have random behavior.We have to use **randomness** in our models so stochastic caculus and stochastic differential equations are needed. So, for example, the famous **Black-Scholes model** is a typical quantitative analysis
related approach where we use stochastic differential equations and we assume random behavior of
stock prices in order to calculate the value of a given option. And it is working quite fine.

## Random behavior

<span style="color:red"> why we have to include randomness in our model ?</span>

By analyzing the so-called daily returns.The daily return is the stock price to day minus the stock price yesterday divided by the stock price yesterday.
$$\frac{S(t)-S(t-1)}{S(t-1)} =R(t)$$



```python 
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Fetch stock data from Yahoo Finance
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2021-12-31'

stock_data = yf.download(ticker, start=start_date, end=end_date)

# Calculate daily returns
stock_data['Daily_Return'] = (stock_data['Close'] - stock_data['Close'].shift(1)) / stock_data['Close'].shift(1)

# Drop missing values
stock_data.dropna(inplace=True)

# Plot histogram of daily returns
plt.hist(stock_data['Daily_Return'], bins=50, alpha=0.75, edgecolor='black')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.title(f'Daily Return Histogram for {ticker}')

plt.show()


```
![uploads](/uploads/outpute.png)



And as you can see, the histogram is very, very similar to a normal distribution with
mean zero. And of course, we have a given standard deviation or variance.
So as we have seen the daily returns and by the way, this is the case when dealing
with monthly returns as well, have approximately normal distributions.

Normal distributions can be defined by two parameters,a $\mu $ mean and the $\sigma^{2} $ variance.

Therefore, the **daily return** can be defined with these parameters (mean and variance):
$$R(t)=\  mean\  +C\  \times \  standard\  deviation$$

Where we can see that we have a so-called deterministic part which is the mean and we have the stochastic part which is the constant times the deviation. 

Thus we can define return as a random variable drawn from a normal distribution

So, for example, if we want to get the stock price tomorrow, of course, we know the
stock price today $S(t)$, which means that we know capital assets.Then the asset price tomorrow $S(t+1)$ is a random variable drawn from a normal distribution.

So this is why we can use the same approach when dealing with stock prices, not just
daily returns, but if we want to model the fluctuations of a given stock price so stock price can be
described with the help of a so-called random walk $W(t)$ or the so-called Wiener process where $W(t)$ has continous sample path and it has independently distributued increments


# Wiener Process and Random Walk
The Wiener process and random walk are both mathematical concepts used to model stochastic processes, which are random processes evolving over time. They are related but have some differences

<span style="color:red"> Wiener Process:</span>: A Wiener process, also known as Brownian motion, is a continuous-time stochastic process that has the following properties:

- It starts at zero: $W(0) = 0$.
- -It has independent increments: The change in the process over non-overlapping intervals is independent.
- It has normally distributed increments: The change in the process over a time interval follows a normal distribution with mean 0 and variance proportional to the length of the interval $\left[ W(t)-W(s)\approx N(0,t-s)\right]  $ this is the Gaussian increaments
- It has continuous paths: The process is continuous in time, meaning there are no jumps or discontinuities in the path.

The Wiener process is widely used in finance, particularly in the Black-Scholes option pricing model and geometric Brownian motion for simulating stock prices.

<span style="color:red"> Random Walk:</span>: A random walk is a discrete-time stochastic process, where the value of the process at each time step is determined by a random variable. In the simplest form, a random walk can be represented as:
$$X(t+1)=X(t)+\varepsilon (t)$$

- Here, $X(t)$ is the value of the process at time $t$, and $\varepsilon (t)$ is a random variable representing the change in the process from time $t$ to time $t+1$.

- There are various types of random walks, such as symmetric random walks, where the probability of moving up or down is equal, and random walks with drift, where there's a tendency to move in a particular direction.


A random walk can be seen as a discrete version of the Wiener process when the increments $\varepsilon (t)$ are independent and normally distributed. In this case, the random walk can be approximated by a Wiener process as the time steps become smaller.

And this is why we can come to the conclusion that stock prices follow a so-called
normal distribution.
In probability theory, a normal distribution is a continuous probability distribution of a
random variable
whose logarithm is normally distributed, which means that if we take the natural
logarithm of stock
prices, that these values are normally distributed.
So if the random variable $X$ acts is normally distributed, then $Y=ln(X)$


### The Stochastic Differential Equation
The stochastic differential equation (SDE) is a type of differential equation that involves one or more random variables, which represent the effects of uncertainty on the system's evolution. In finance, SDEs are commonly used to model the behavior of stock prices, interest rates, and other financial variables.

For instance, one of the most well-known SDEs used in finance is the Geometric Brownian Motion (GBM) model, which is used to describe the evolution of stock prices. The GBM is given by the following stochastic differential equation:

$$dS=\mu S\text{dt} +\sigma S\text{dW} $$

$\text{dW}$ is a random  variable drawn from a normal distribution with mean $0$ and variance $dt$
- $dS$ is the $S(t+dt)-S(t)$ change the stock price 
- $\mu S\text{dt} $ deterministic part- the drift
- $\sigma S\text{dW} $ stochastic part with Wiener-process

This is contious model of asset price ad the fundamental assumption for most of the modern financial model


### Wiener-process implementation


```python 
import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt


def wiener_process(dt=0.5, x0=0, n=1000):

    # W(t=0)=0
    # initialize W(t) with zeros
    W = np.zeros(n+1)

    # we create N+1 timesteps: t=0,1,2,3...N
    t = np.linspace(x0, n, n+1)

    # we have to use cumulative sum: on every step the additional value is
    # drawn from a normal distribution with mean 0 and variance dt ... N(0,dt)
    # by the way: N(0,dt) = sqrt(dt)*N(0,1) usually this formula is used !!!
    W[1:n+1] = np.cumsum(np.random.normal(0, np.sqrt(dt), n))

    return t, W


def plot_process(t, W):
    plt.plot(t, W)
    plt.xlabel('Time(t)')
    plt.ylabel('Wiener-process W(t)')
    plt.title('Wiener-process')
    plt.show()


if __name__ == '__main__':
    time, data = wiener_process()
    plot_process(time, data)

```



![uploads](/uploads/outputf.png)


<span style="color:red"> What is Itô’s Lemma ? </span>

Itô’s Lemma is a theorem in stochastic calculus. It tells you that if you have a random walk, in y, say, and a function of that randomly walking variable, call it $f(y, t)$, then you can
easily write an expression for the random walk in $f$. A function of a random variable is itself random in general.

**Example** The obvious example concerns the random walk
$$dS=\mu S\text{dt} +\sigma S\text{dW} $$

commonly used to model an equity price or exchange rate, $S$. What is the stochastic differential equation for the logarithm of $S, lnS$?

$$d(lnS)=(\mu -\frac{1}{2} \sigma^{2} )dt+\sigma dX$$



<span style="color:green"> Stochastic Calculus: The solution of geometric random vork stochastic differential equation  </span>
$$S(t)=S(0)e^{\left( \mu -\frac{1}{2} \sigma^{2} \right)  t+\sigma W_{t}}$$
is used to model stock prices using geometric Brownian motion (GBM), which is a continuous-time stochastic process. In the equation:
- $S(t)$ represents the stock price at time t.
- $S(0)$ represents the initial stock price at time 0.
- $\mu$ represents the drift, which is the expected return of the stock.
- $\sigma$ (sigma) represents the volatility, which is the standard deviation of the stock's returns.
- $W_t$ represents the Wiener process (Brownian motion) at time t.
- $exp(x)$ is the exponential function, e^x.

The GBM model is based on the assumption that the stock prices follow a **log-normal distribution**, and it incorporates both the drift (trend) and the random fluctuations (volatility) of the stock price. The drift term $\left( \mu -\frac{1}{2} \sigma^{2} \right)  t$ captures the average growth of the stock price over time, while the stochastic term $\sigma W_{t}$ captures the random fluctuations in the price.


### Geometric Brownian Motion implementation



```python 
import matplotlib.pyplot as plt
import numpy as np


def simulate_geometric_random_walk(S0, T=2, N=1000, mu=0.1, sigma=0.05):

    dt = T/N
    t = np.linspace(0, T, N)
    # standard normal distribution N(0,1)
    W = np.random.standard_normal(size=N)
    # N(0,dt) = sqrt(dt) * N(0,1)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    S = S0 * np.exp(X)

    return t, S


def plot_simulation(t, S):
    plt.plot(t, S)
    plt.xlabel('Time (t)')
    plt.ylabel('Stock Price S(t)')
    plt.title('Geometric Brownian Motion')
    plt.show()


if __name__ == '__main__'

    time, data = simulate_geometric_random_walk(1)
    plot_simulation(time, data)

```

![uploads](/uploads/outputg.png)


# Black-Scholes Model


The Black–Scholes equation is a differential equation for the value of an option as a function of the underlying asset and time.

The model is based on the assumption that the underlying asset's price follows a geometric Brownian motion, which is characterized by a constant drift and volatility.

The model aims to calculate the fair value of an option, considering factors like the current stock price, the option's strike price, the time until expiration, the risk-free interest rate, and the underlying stock's volatility.

The key equation in the Black-Scholes model is the Black-Scholes partial differential equation, which can be solved to obtain the option price. The Black-Scholes formula for a European call option (the right to buy an asset at a specified price) is:

$$\frac{\partial V}{\partial t} +\frac{1}{2} \sigma^{2} S^{2}\frac{\partial^{2} V}{\partial S^{2}} +rS\frac{\partial V}{\partial S} -rV=0$$
where $V(S,t)$ is the option value as a function of asset price
$S$ and time $t$.

<span style="color:red"> Facts about the Black–Scholes equation:  </span>

- The equation follows from certain assumptions and from a mathematical and financial argument that involves hedging.
- The equation is linear and homogeneous (we say ‘there is no right-hand side,’ i.e. no non-V terms) so that you can value a portfolio of derivatives by summing the values of the individual contracts.
- It is a partial differential equation because it has more than one independent variable, here S and t.
- It is of parabolic type, meaning that one of the variables, $t$, only has a first-derivative term, and the other $S$ has a second-derivative term.
- It is of backward type, meaning that you specify a final condition representing the option payoff at expiry and then solve backwards in time to get the option value now. You can tell it’s backward by looking at the sign of the $t-$derivative term and the second $S-$derivative term, when on the same side of the equals sign they are both the same sign. If they were of opposite signs then it would be a forward equation.
The equation is an example of a diffusion equation or heat equation. Such equations have been around for nearly two hundred years and have been used to model all sorts of physical phenomena.
- The equation requires specification of two parameters, the risk-free interest rate and the asset volatility. The interest rate is easy enough to measure, and the option value isn’t so sensitive to it anyway. But the volatility is another matter, rather harder to forecast accurately.
- Because the main uncertainty in the equation is the volatility one sometimes thinks of the equation less as a valuation tool and more as a way of understanding the relationship between options and volatility.
- The equation is easy to solve numerically, by finite-difference or Monte Carlo methods, for example.
- The equation can be generalized to allow for dividends, other payoffs, stochastic volatility, jumping stock prices, etc.


<span style="color:red"> The Black–Scholes formulæ   </span>: which are solutions of the equation in special cases, such as for calls and puts. $\frac{\partial V}{\partial t} +\frac{1}{2} \sigma^{2} S^{2}\frac{\partial^{2} V}{\partial S^{2}} +rS\frac{\partial V}{\partial S} -rV=0$

The equation contains four terms:
- $\frac{\partial V}{\partial t} $ time decay, how much the option value changes by if the stock price doesnt change
- $\frac{1}{2} \sigma^{2} S^{2}\frac{\partial^{2} V}{\partial S^{2}} $ : convexity term, how much a hedged position makes on average from stock moves
-  $ S\frac{\partial V}{\partial S} $ : drift term allowing for the growth in the stock at the
risk-free rate
- $rV$ the discounting term, since the payoff is received
at expiration but you are valuing the option now.


<span style="color:red"> Solution to The Black–Scholes equation   </span>: 

Black-Scholes equation
$$\left[ \frac{\partial V}{\partial t} +\frac{1}{2} \sigma^{2} S^{2}\frac{\partial^{2} V}{\partial S^{2}} \right]  dt=r\left( V-S\frac{\partial V}{\partial S} \right)  dt$$

It is a parabolic partial differential equation

Linear: so the sum of the solutions is also a solution

Financial equations are usually parabolic: They are related to heat and diffusion equations of Physics

<span style="color:green"> Solution to The Black–Scholes equation   </span>: no dividend yields onn the underlying

$$N(x)=\frac{1}{\sqrt{2\pi } } \int^{x}_{-\infty } e^{-\frac{z^{2}}{2} }\  dz $$


Standard normal distribution


**Call option**
$$S(0)N(d_{1})-Ee^{-r(T-t)}N(d_{2})$$

**Put Option**
$$-S(0)N(-d_{1})+Ee^{-r(T-t)}N(-d_{2})$$


Where, $$d_{1}=\frac{log\left[ \frac{S\left( 0\right)  }{E} \right]  +(r+\frac{1}{2} \sigma^{2} )(T-t)}{\sigma \sqrt{T-t} } $$


$$d_{2}=d_{1}-\sigma \sqrt{T-t} $$

```python 
from scipy import stats
from numpy import log, exp, sqrt


def call_option_price(S, E, T, rf, sigma):
    # first we have to calculate d1 and d2 parameters
    d1 = (log(S / E) + (rf + sigma * sigma / 2.0) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    print("The d1 and d2 parameters: %s, %s" % (d1, d2))
    # use the N(x) to calculate the price of the option
    return S*stats.norm.cdf(d1)-E*exp(-rf*T)*stats.norm.cdf(d2)


def put_option_price(S, E, T, rf, sigma):
    # first we have to calculate d1 and d2 parameters
    d1 = (log(S / E) + (rf + sigma * sigma / 2.0) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    print("The d1 and d2 parameters: %s, %s" % (d1, d2))
    # use the N(x) to calculate the price of the option
    return -S*stats.norm.cdf(-d1)+E*exp(-rf*T)*stats.norm.cdf(-d2)


if __name__ == '__main__':
    # underlying stock price at t=0
    S0 = 100
    # strike price
    E = 100
    # expiry 1year=365days
    T = 1
    # risk-free rate
    rf = 0.05
    # volatility of the underlying stock
    sigma = 0.2

    print("Call option price according to Black-Scholes model: ",
          call_option_price(S0, E, T, rf, sigma))
    print("Put option price according to Black-Scholes model: ",
          put_option_price(S0, E, T, rf, sigma))
```


- The d1 and d2 parameters: 0.35000000000000003, 0.15000000000000002
- Call option price according to Black-Scholes model:  10.450583572185565
- The d1 and d2 parameters: 0.35000000000000003, 0.15000000000000002
- Put option price according to Black-Scholes model:  5.573526022256971

# Monte Carlo Simulation

### <span style="color:red"> Definition   </span>  
Monte Carlo simulations are a way of solving probabilistic problems by numerically ‘imagining’ many possible scenarios or games so as to calculate statistical properties such as expectations, variances or probabilities of certain outcomes. In finance we use such simulations to represent the future behaviour of equities, exchange rates, interest rates, etc., so as to either study the possible future performance of a port- folio or to price derivatives.

### <span style="color:red"> Example   </span>  
We hold a complex portfolio of investments, we would like
to know the probability of losing money over the next year since our bonus depends on our making a profit. We can estimate this probability by simulating how the individual components in our portfolio might evolve over the next year. This requires us to have a model for the random behaviour of each of the assets, including the relationship or correlation between them, if any.
Some problems which are completely deterministic can also be solved numerically by running simulations, most famously finding a value for $\pi$.

It is clear enough that probabilistic problems can be solved by simulations. What is the probability of tossing heads with a coin, just toss the coin often enough and you will find the answer. More on this and its relevance to finance shortly. But many deterministic problems can also be solved this way, provided you can find a probabilistic equivalent of the deterministic problem. A famous example of this is Buffon’s needle, a problem and solution dating back to 1777. Draw parallel lines on a table one inch apart. Drop a needle, also one inch long, onto this table. Simple trigonometry will show you that the probability of the needle touching one of the lines is $\frac{2}{\pi } $. So conduct many such experiments to get an approximation to $\pi$. Unfortunately because of the probabilistic nature of this method you will have to drop the needle many billions of times to find π accurate to half a dozen decimal places.

There can also be a relationship between certain types of differential equation and probabilistic methods. Stanislaw Ulam, inspired by a card game, invented this technique while working on the Manhattan Project towards the development of nuclear weapons. The name **Monte Carlo** was given to this idea by his colleague Nicholas Metropolis.


**Monte Carlo simulations** are used in financial problems for solving two types of problems:
- Exploring the statistical properties of a portfolio of investments or cashflows to determine quantities such as expected returns, risk, possible downsides, probabilities of making certain profits or losses, etc.
- Finding the value of derivatives by exploiting the theoretical relationship between option values and expected payoff under a risk-neutral **random walk**.

**Exploring portfolio statistics**: The most successful quantitative models represent investments as random walks. There is a whole mathematical theory behind these models, but to appreciate the role they play in portfolio analysis you just need to understand three simple concepts.

- First, you need an algorithm for how the most basic investments evolve randomly. In equities this is often the lognormal random walk. (If you know about the real/risk-neutral distinction then you should know that you will be using the real random walk here.) This can be represented on a spreadsheet or in code as how a stock price changes from one period to the next by adding on a random return. In the fixed-income world you may be using the BGM model ( *The BGM model, also known as the Brace-Gatarek-Musiela (BGM) model or the Libor Market Model (LMM), is a financial model used to describe the evolution of interest rates*) to model how interest rates of various maturities evolve. In credit you may have a model that models the random bankruptcy of a company. If you have more than one such investment that you must model then you will also need to represent any interrelationships between them. This is often achieved by using correlations.

- Once you can perform such simulations of the basic investments then you need to have models for more complicated contracts that depend on them, these are the options/derivatives/contingent claims. For this you need some theory, derivatives theory. This the second concept you must understand.

- Finally, you will be able to simulate many thousands, or more, future scenarios for your portfolio and use the results to examine the statistics of this portfolio. This is, for example, how classical Value at Risk can be estimated, among other things.

**Pricing derivatives** We know from the results of risk-neutral pricing that in the popular derivatives theories the value of an option can be calculated as the present value of the expected payoff under a risk-neutral random walk. And calculating expectations for a single contract is just a simple example of the above-mentioned portfolio analysis, but just for a single option and using the risk-neutral instead of the real random walk. Even though the pricing models can often be written as deterministic partial differential equations they can be solved in a probabilistic way, just as Stanislaw Ulam noted for other, non-financial, problems. This pricing methodology for derivatives was first proposed by the actuarially trained Phelim Boyle in 1977.
Whether you use Monte Carlo for probabilistic or deterministic problems the method is usually quite simple to implement in basic form and so is extremely popular in practice.


### <span style="color:red"> Application to stoke price  </span> 

We know that $S(t)$ assets (such as stocks) follow **lognormal random walk**
$$dS=\mu S\text{dt} +\sigma S\text{dW} $$
 
- $\mu S\text{dt} $ deterministic part- the drift that can be characterized by the mean 
- $\sigma S\text{dW} $ stochastic part with Wiener-process that can be characterized by the standard deviation or the so called volatility 

So what do we have to do if we know the starting point? So as zero, which means that the stock price at T goes to zero and we know the given parameters the mean and the standard deviation we can calculated based on historical data then we can make multiple simulations and it is quite cheap to create a simulation like these since we know how to simulate **lognormal random walks** 

We have to create tens of thousands of simulations in the sense that we are going to
create the first simulation.Of course, the stock price are going to fluctuate.
They will increase, they will decrease and so on. So the first simulation is going to yield a different path than the second simulation will yield another path. The simulation will yield another path of the underlying stock and so on. If we make tens of thousands of simulations, then we end up with this implementation


```python 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM_OF_SIMULATIONS = 10000


def stock_monte_carlo(S0, mu, sigma, N=252):

    result = []

    # number of simulations - possible S(t) realizations (of the process)
    for _ in range(NUM_OF_SIMULATIONS):
        prices = [S0]
        for _ in range(N):
            # we simulate the change day by day (t=1)
            stock_price = prices[-1] * np.exp((mu - 0.5 * sigma ** 2) +
                                              sigma * np.random.normal())
            prices.append(stock_price)

        result.append(prices)

    simulation_data = pd.DataFrame(result)
    # the given columns will contain the time series for a given simulation
    simulation_data = simulation_data.T

    # plt.plot(simulation_data)
    # plt.show()

    # print('Prediction for future stock price: $%.2f' % simulation_data['mean'].tail(1))

    return simulation_data


if __name__ == '__main__':
    simulation_data = stock_monte_carlo(50, 0.0002, 0.01)
    plt.plot(simulation_data)
    plt.show()

    print('Prediction for future stock price: $%.2f' % simulation_data.mean(axis=1).tail(1).values[0])


```

We generate a huge amount of possible $S(t)$ geometric random walk process
The mean of these simulations yields the $S_{B}\left( t\right)$ path with the highest probability in the future. This is a typical **Monte-Carlo simulation**


## Which Numerical Method should I Use and When?

 <span style="color:red"> Definition   </span>:  The three main numerical methods in common use are **Monte Carlo, finite difference and numerical quadrature**. (I’m including the binomial method as just a simplistic version of finite differences.)  <span style="color:green"> Monte Carlo  </span>is great for complex path dependency and high dimensionality, and for problems which can not easily be written in differential equation form. Monte Carlo methods simulate the random behaviour underlying the financial models. So, in a sense they get right to the heart of the problem. Always remember, though, that when pricing you must simulate the risk-neutral random walk(s), the value of a contract is then the expected present value of all cashflows. <span style="color:green"> Finite difference  </span> is best for low dimensions and contracts with decision features such as early exercise, ones which have a differential equation formulation Since we work with a mesh, not unlike the binomial method, we will find the contract value at all points is stock price-time space. In quantitative finance that differential equation is almost always of diffusion or parabolic type.  <span style="color:green"> Numerical quadrature  </span> is for when you can write the option value as a multiple integral. To be more detail ,occasionally one can write down the solution of an **option-pricing** problem in the form of a multiple integral. This is because you can interpret the option value as an expectation of a payoff, and an expectation of the payoff is mathematically just the integral of the product of that payoff function and a probability density function. This is only possible in special cases. The option has to be European, the underlying stochastic differential equation must be explicitly integrable (so the lognormal random walk is perfect for this) and the payoff shouldn’t usually be path dependent. So if this is possible then pricing is easy... you have a formula. The only difficulty comes in turning this formula into a number. And that’s the subject of numerical integration or quadrature.

 <span style="color:red"> Example  </span>  
You want to price a fixed-income contract using the BGM model. Which numerical method should you use? BGM is geared up for solution by simulation, so you would use a Monte Carlo simulation.

You want to price an option which is paid for in instalments, and you can stop paying and lose the option at any time if you think it’s not worth keeping up the payments. This may be one for finite-difference methods since it has a decision feature.

You want to price a European, non-path-dependent contract on a basket of equities. This may be recast as a multiple inte- gral and so you would use a quadrature method.

 <span style="color:red"> Summary  </span> 

 Pros and cons of different methods: Finite Dimension(FD), Monte-Carlos(MC), Numerical quadrature (Quand)

 | Subject            | Low dimensions | High dimensions | Path dependent Greeks | Portfolio Decisions | Non-linear |
|--------------------|----------------|-----------------|-----------------------|---------------------|------------|
| FD                 | Good           | Slow            | Depends               | Excellent           | Inefficient | Excellent | Excellent |
| MC                 | Inefficient    | Excellent       | Excellent             | Not good            | Very good   | Poor      | Poor      |
| Quad.              | Good           | Good            | Not good              | Excellent           | Very good   | Very poor | Very poor |


### **About the author**



<div align="center">
  <img src="/imagez/huybinh.png" alt="Author's Photo" width="150" style="border-radius: 50%; border: 2px solid #1E90FF;">
  <br>
  <strong>Huy Binh TRAN</strong>
  <br>
  <em>Master 2 Quantum Devices at Institute Paris Polytechnic, France</em>
  <br>
  <a href="https://www.linkedin.com/in/huybinhtran/" style="color:#1E90FF;">LinkedIn</a>
</div>




{{< math >}}
{{< /math >}} 