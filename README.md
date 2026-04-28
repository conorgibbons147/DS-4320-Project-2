# DS-4320-Project-2: Can Data Predict the Stock Market?
  
**Name:** Conor Gibbons

**NetID:** hjd3db

## Table of Contents
* __[Executive Summary](#executive-summary)__
* __[Problem Definition](#problem-definition)__
* __[Domain Exposition](#domain-exposition)__
* __[Data Creation](#data-creation)__
* __[Metadata](#metadata)__
* __[License](#license)__

## Pipeline
* __[pipeline.ipynb](pipeline/pipeline.ipynb)__
* __[pipeline.md](pipeline/pipeline.md)__

### DOI


### License
MIT License — see [LICENSE](LICENSE)

### Executive Summary
This repository contains the full pipeline for DS 4320 Project 2, a machine learning project that attempts to predict whether a stock's price will go up or down the following trading day. The project uses daily historical price and volume data for 13 large cap tech stocks including Apple, Microsoft, and NVIDIA spanning 2015 through 2024. Data was collected using the yfinance library and stored in MongoDB Atlas. An XGBoost classifier was trained on ten technical indicators derived from the raw price data and evaluated on held out data from 2023 to 2024, achieving 51.3% accuracy. The results confirm that simple technical indicators contain little to no exploitable signal in daily stock price movement, consistent with the Efficient Market Hypothesis. All code, data documentation, and results are organized in this repository.

## Problem Definition

**Initial Problem:** Forecasting stock prices

**Refined Problem:** Given historical daily price and volume data for a
selection of S&P 500 stocks, can we predict whether a stock's closing price
will be higher or lower the following trading day?

### Refinement Rationale
Stock price forecasting is a broad problem that could mean anything from
predicting trends to modeling exact pricing. Narrowing the problem to next day movement for S&P 500 stocks makes the project realistic and honest. The S&P 500 provides a well documented, easily accessible collection of stocks with clean historical data. Framing the prediction as a binary classification rather than an exact price will make the predictions simpler and give us a clear and interpretable evaluation metric. Daily price change is the sweet spot between too much noise at the intraday level and too little signal at the monthly level.

### Project Motivation
Stock price movement affects nearly everyone, through personal accounts, index funds, and the broader economy. Being able to predict whether a stock will go up or down the next day has obvious financial value, but it is also a fascinating challenge because markets are very unpredictable by nature. This project explores whether patterns in historical price and volume data contain any signal at all, and if so, how much. Even a model that performs modestly better than chance has real implications for how we think about markets and the limits of data driven decision making.

### Press Release
[Can Data Data Beat the Stock Market?](press-release.md)

## Domain Exposition

### Terminology:
| Term | Definition |
|------|------------|
| OHLCV | Open, High, Low, Close, Volume — the standard fields in daily stock data |
| Closing Price | The final price a stock traded at during a regular trading session |
| Moving Average (MA) | The average closing price over a rolling window of days, used to smooth out noise |
| RSI | Relative Strength Index — a momentum indicator measuring whether a stock is overbought or oversold on a scale of 0 to 100 |
| Trading Volume | The number of shares traded in a given day, used as a proxy for market interest |
| Binary Classification | A model that predicts one of two outcomes — in this case, price up or price down |
| Accuracy | The percentage of predictions the model got correct |
| Precision | Of all the times the model predicted "up", how often it was right |
| Recall | Of all the actual "up" days, how many the model correctly identified |
| Baseline | The accuracy of a naive model that always predicts the most common class, used as a benchmark |
| S&P 500 | An index of 500 large US publicly traded companies, widely used as a benchmark for the overall market |
| Data Leakage | When information from the future accidentally gets used to train the model, artificially inflating accuracy |

### Domain
This project lives in the domain of quantitative finance, specifically the use of historical price and volume data to forecast future stock movements. It is a well studied but genuinely difficult problem since markets are complex, noisy, and influenced by countless factors that no dataset can fully capture. That said, it is also a domain where even small predictive advantages have
real financial consequences, which is why quantitative traders continue to invest heavily in data driven approaches. Machine learning has opened new possibilities here by offering models that can detect subtle patterns in price and volume data that traditional statistical methods might miss.

### Background Reading
| Title | Description | Link |
|-------|-------------|------|
| Advancing Financial Forecasts: Stock Price Prediction Based on Time Series and Machine Learning | Compares LSTM with Random Forest, SVM, and other classifiers for next day stock direction prediction | [Link](background_reading/Advancing%20Financial%20Forecasts%20%20Stock%20Price%20Prediction%20Based%20on%20Time%20Series%20and%20Machine%20Learning%20Techniques.pdf) |
| A Multi-Model Machine Learning Framework for Daily Stock Price Prediction | Evaluates nine ML models including XGBoost and Random Forest on Apple, Tesla, and NVIDIA using technical indicators | [Link](background_reading/A%20Multi-Model%20Machine%20Learning%20Framework%20for%20Daily%20Stock%20Price%20Prediction.pdf) |
| Stock Market Prediction Using Machine Learning and Deep Learning Techniques: A Review | Broad review of ML and deep learning approaches to stock prediction, covering LSTM, CNN, and SVM | [Link](background_reading/Stock%20Market%20Prediction%20Using%20Machine%20Learning%20and%20Deep%20Learning%20Techniques%20A%20Review.pdf) |
| Stock Market Trend Prediction Using Deep Neural Network via Chart Analysis: A Practical Method or a Myth? | Critical look at why many LSTM based stock prediction studies produce misleading results in practice | [Link](background_reading/Stock%20market%20trend%20prediction%20using%20deep%20neural%20network%20via%20chart%20analysis%20a%20practical%20method%20or%20a%20myth.pdf) |
| Short-term Stock Market Price Trend Prediction Using a Comprehensive Deep Learning System | Proposes a deep learning pipeline with feature engineering for binary trend classification | [Link](background_reading/Short-term%20stock%20market%20price%20trend%20prediction%20using%20a%20comprehensive%20deep%20learning%20system.pdf) |

## Data Creation

### Provenance
Daily OHLCV (open, high, low, close, volume) data was collected for 13 technology stocks was collected using yfinance which is Yahoo Finance's public market data API. Records span from the beginning of 2015 to the end of 2024, with 32,682 documents across all companies. Prices are split and dividend adjusted to prevent artificial discontinuities from corporate events. The 13 tickers were hand-selected to ensure continuous listing throughout the full ten year window. The binary label, aimed at whether the closing price increased the following trading day, was derived by shifting the close price series forward by one row. A label of 1 indicates the next day's close strictly exceeded the current close and 0 indicates flat or down.

### Code
| File | Description | Link |
|---|---|---|
| `data_loader.ipynb` | Downloads daily OHLCV data for 13 tech stocks via `yfinance` and inserts 32,682 documents into MongoDB Atlas | [data_loader.ipynb](pipeline/data_loader.ipynb) |

### Bias Identification
The 13 tickers were selected from companies that are currently large-cap and continuously listed, meaning any firm that declined, was acquired, or was delisted between 2015 and 2024 is absent from the dataset. This skews the data toward historically successful companies. Restricting to the technology sector also limits the generalizability of the analysis, as tech stocks tend to move together in response to shared conditions/events. There is a 52.8/47.2 Up/Down label split that reflects a predominantly strong market decade, so any model trained here may underperform during sustained downs. Finally, the binary label is derived from the next day's closing price, which would be unavailable in a real prediction setting, requiring careful feature engineering to ensure no future information leaks into the model.

### Bias Mitigation
Restricting scope to large tech over a specific decade should be clearly stated as a limitation rather than something to correct, as the dataset was never intended to generalize beyond this context. Because the label split is close to 50/50, no resampling is needed, but reporting a baseline accuracy of 52.8% alongside model results ensures any performance gains are interpreted honestly. The look-ahead bias is controlled by using a time-based train/test split rather than a random one, so the model is always evaluated on data that falls after its training window.

### Rationale for Decisions
The decision to frame the problem as binary classification rather than price regression was made to produce a clear, interpretable evaluation metric. Predicting exact prices introduces high error with large data scope and is difficult to evaluate meaningfully. The ten year window from  was chosen to capture a variety of market conditions and hopefully reducing the risk that the model learns patterns specific to a single era. Restricting to 13 tech stocks was a tradeoff between dataset size and depth. A broader selection would have introduced sector noise that is hard to control for. Using auto_adjust=True in yfinance was a judgement call that simplifies preprocessing but means the raw price values are synthetic rather than what investors actually saw at the time. Finally, labeling flat days as 0 rather than a separate class keeps the problem binary but slightly penalizes the positive class. This edge case is rare though so it is unlikely to impact results.

## Metadata

### Implicit Schema
The following guidelines define the expected structure for all
documents in the `tech_stocks` collection. Every document must contain the following fields:

| Field | Type | Required |
|---|---|---|
| `ticker` | string | yes |
| `date` | datetime | yes |
| `open` | float | yes |
| `high` | float | yes |
| `low` | float | yes |
| `close` | float | yes |
| `volume` | integer | yes |
| `next_close` | float | yes |
| `label` | integer (0 or 1) | yes |

All numeric fields are rounded to 4 decimal places. `label` must be strictly 0 or 1. `date` must be stored as a Python datetime object No additional fields should be added.

### Data Summary
| Property | Value |
|---|---|
| Collection | `tech_stocks` |
| Total Documents | 32,682 |
| Tickers | 13 |
| Documents per Ticker | 2,514 |
| Date Range | January 1, 2015 — December 31, 2024 |
| Label Distribution | 52.8% Up (1), 47.2% Down/Flat (0) |
| Fields per Document | 9 |

### Data Dictionary
| Field | Data Type | Description | Example |
|---|---|---|---|
| `ticker` | string | Stock ticker symbol | `"AAPL"` |
| `date` | datetime | Trading day date | `2024-03-15 00:00:00` |
| `open` | float | Adjusted opening price | `172.3400` |
| `high` | float | Adjusted intraday high price | `174.9100` |
| `low` | float | Adjusted intraday low price | `171.8000` |
| `close` | float | Adjusted closing price | `173.5000` |
| `volume` | integer | Number of shares traded | `58432100` |
| `next_close` | float | Adjusted closing price of the following trading day | `175.2000` |
| `label` | integer | 1 if next_close > close, 0 otherwise | `1` |

### Quantification of Uncertainty
| Field | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| `open` | \$118.50 | \$116.76 | \$0.46 | \$696.28 |
| `high` | \$120.00 | \$118.20 | \$0.47 | \$699.54 |
| `low` | \$116.95 | \$115.23 | \$0.45 | \$678.91 |
| `close` | \$118.52 | \$116.76 | \$0.46 | \$688.37 |
| `volume` | 75,657,608 | 142,140,914 | 0 | 3,692,928,000 |