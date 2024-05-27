# CryptoSentiment Analysis: Unveiling Market Trends through Sentiment Exploration (FCSS - Final Project)

## Summary

The primary aim of this project is to evaluate the relationship between people's opinions on Reddit and the actual changes in the crypto market, specifically regarding Bitcoin's traded volume and market price. The regression analysis conducted in this study demonstrates a significant relationship between sentiments and changes in the crypto market.

## Reproducibility

This repository contains all the necessary code and data to reproduce the analysis:

### Folder Structure

- **Python_Functions_Codes/**: Contains all the .py files needed for the analysis.
- **Data/**: Includes all the required CSV files.
- **Visualization/**: Contains all the result graphs generated during the analysis.

## Note

1. Ensure that all the required Python packages are installed for the code to function correctly.
2. Update the file paths based on your use case to allow Python to import files as intended.

## Packages and Tools

The following packages and tools are used in this project:

- **Python**
  - **CCXT**: A package used to fetch cryptocurrency data from major trading platforms.
  - **Pandas**: For data manipulation and analysis.
  - **Praw**: For accessing Reddit API.
  - **BERT**: For natural language processing.
  - **VADER**: For sentiment analysis.
  - **transformers**: For advanced NLP tasks.
  - **matplotlib**: For data visualization.

## Installation

To install the necessary packages, you can use the following pip command:

```bash
pip install ccxt pandas praw transformers vaderSentiment matplotlib

