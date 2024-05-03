# Transformers

## Overview

This project applies transformer models to optimize trading strategies by predicting market buy/sell signals. Using deep learning techniques like the Transformer architecture, and technical analysis indicators, we aim to outperform standard trading approaches in various market conditions.


## Objectives

- To Understand Transformer Architecture: Gain a deep understanding of the transformer architecture, particularly the self-attention mechanism, and how it can be adapted to the domain of financial time series analysis.
- To Implement Transformer Models in Trading: Develop and implement transformer-based models to analyze and predict market behaviors using high-frequency trading data, focusing on 5-minute intervals for stocks like AAPL (Apple Inc.).
- To Enhance Predictive Accuracy: Leverage the inherent capabilities of transformers to capture complex temporal dependencies in financial data, aiming to improve the accuracy of predictive trading signals.
- To Compare with Traditional Models: Conduct comparative analysis between transformer models and traditional trading algorithms to evaluate performance improvements in terms of accuracy and efficiency.
- To Provide Real-Time Trading Insights: Implement the models in a way that they can be used to provide real-time insights into market dynamics, supporting automated or semi-automated trading decisions.
- To Foster Financial Strategy Development: Use insights derived from transformer models to inform and refine trading strategies, focusing on maximizing profitability and managing risk in trading portfolios.
- To Contribute to the Field of Quantitative Finance: Through rigorous testing and backtesting, contribute valuable findings to the community and literature of quantitative finance, showcasing the applicability of advanced machine learning techniques in the financial sector.
- To Educate and Inspire: Provide clear, accessible explanations and demonstrations of the technology for educational purposes, thereby inspiring further research and development in the application of deep learning in finance.


## Methodology

Our methodology consisted of a systematic approach to develop and evaluate trading models using transformer-based architectures. The process involved the following steps:

1. Data Preparation: We collected historical trading data for the AAPL stock at 5-minute intervals. The data was preprocessed to handle missing values and standardized to ensure uniformity across features.
2. Technical Indicator Application: We applied a variety of technical indicators, including Relative Strength Index (RSI), Weighted Moving Average (WMA), Moving Average Convergence Divergence (MACD), and Bollinger Bands, to derive additional features from the raw price data.
3. Model Configuration: We configured transformer models to accept input features derived from the technical indicators.
4. Training and Validation: The models were trained on the preprocessed data using a portion of the dataset, running 100 epochs of the transformer, while another portion was reserved for validation to assess model performance.
5. Backtesting: We implemented backtesting using the test data to validate that the signals received were accurate and reliable, preparing the model for potential real-world application.


## Key Findings

- **Optimal Strategy**: Described the composition of the best-performing strategy, including the indicators used, trade signal logic, structure of the transformers and parameters used.
- **Performance Comparison**: Demonstrated the superiority of the optimized strategy over a passive investment approach through detailed metrics and visualizations.


## Project Structure
    Transformers/                                  ->      root project dir
    ├── data/                                      ->      additional files
    ├── functions.py                               ->      containing the functions that are used in main
    ├── main.py                                    ->      main code
    ├── transformer_classifier.keras               ->      trained transformer model
    ├── Transformers - Backtesting.ipynb           ->      procedure      
    ├── report.ipynb                               ->      visualizations, tables & conclusions
    ├── venv/                                      ->      virtual environment
    ├── .gitignore                                 ->      python's gitignore file from GitHub
    ├── README.md                                  ->      project description, setup instructions
    └── requirements.txt                           ->      libraries & versions required


## Requirements

To run this project, you will need to install the Python libraries and their specific versions as listed in the `requirements.txt` file. This ensures that you have the right environment setup for replicating the analysis and results on the `venv` of this repository. 


## How to Navigate This Repository

This guide provides detailed instructions on how to execute different components of our codebase. Follow these steps to ensure proper functionality.

### Initial Setup
1. **Run `main.py` First:**
   - Ensure that all dependencies from requirements.txt are installed. This will set up the Python environment with the necessary libraries to run the project.
   - Compile and prepare the initial environment settings by running main.py. This script sets up the necessary data structures and preconditions for further analysis. It includes the application of various technical indicators such as RSI, WMA, MACD, and Bollinger Bands, which are crucial for the transformer model's input features.
   - **Data Standardization and Preparation:**
      - The script handles the initial standardization and preparation of the data, crucial for consistent model input and accurate predictions. This includes normalization of features and preparation of training and test datasets.
      - Important Note: Sections of this script dealing with data preparation and initial model training are enclosed within triple quotes (""" """), indicating that they have been pre-run and do not need to be executed again under routine use. This code block has already generated the standardized data and performed the initial model training.
      - If you need to regenerate the standardized data or retrain the model from scratch, you must remove these triple quotes to activate these sections of the code.
   - **Model Loading and Evaluation:**
      - After the initial setup, the pre-trained transformer model stored in transformer_classifier.keras can be loaded for immediate evaluation. This setup ensures that users can directly proceed to applying the model to new data for testing and validation.

### Generating Reports

2. **Independently Run `report.ipynb`:**
   The `report.ipynb` can be executed at any time, independent of whether `main.py` has been run. This is possible because the process exports results to CSV and XLSX files continuously as they are generated, allowing `report.ipynb` to create reports based on these files.

### Testing Different Temporalities

3. **Re-run for Different Temporalities:**
   To test different temporalities, restart the process. Begin again with running `main.py`.

These steps should enable you to successfully run and test various aspects of our code. If you encounter any issues, please refer to the troubleshooting section or contact our support team.


## Authors

- Claudia Valeria Chimal Parra
- Paulo Cesar Ayala Gutiérrez
- Juan Carlos Gutiérrez Valdivia
- Oscar Leonardo Vaca González
- Arturo Espinosa Carabez

