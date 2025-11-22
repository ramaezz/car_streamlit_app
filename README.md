# Used Car Price Prediction in Saudi Arabia

Predicting used car prices in Saudi Arabia using machine learning techniques.  
This project aims to provide fair, data-driven car valuations to help buyers and sellers make informed decisions in an inconsistent market.

---

## Project Overview

The used car market in Saudi Arabia is growing rapidly, as more people prefer affordable second-hand vehicles over new cars. However, pricing is highly inconsistent: the same car can be listed at very different prices depending on the seller, city, or platform. This inconsistency leads to confusion, unfair deals, and uncertainty for both buyers and sellers.

The goal of this project is to **understand how different car features influence price** and to **build a machine learning model that can predict fair car prices** based on real market data.

---

## Dataset

- Source: [Kaggle – Saudi Arabia Used Cars Dataset](https://www.kaggle.com/datasets)  
- Total entries: 8,000+ vehicles  
- Key features include:  
  - Brand, model, year, mileage  
  - Engine size, transmission, and more  
- Original source: Data scraped from the Syarah website, reflecting the real local market  

---

## Methodology

The project follows a structured machine learning workflow:

1. **Data Cleaning & Preprocessing**  
   - Handling missing values  
   - Encoding categorical features  
   - Preparing numerical features for modeling  

2. **Exploratory Data Analysis (EDA)**  
   - Analyzing relationships between car features and price  
   - Visualizing trends, e.g., higher mileage → lower price  

3. **Feature Engineering**  
   - Creating new features based on existing data to improve model performance  

4. **Model Training**  
   - Tested multiple regression models to predict car prices  
   - Evaluated models using metrics like RMSE and R²  

5. **Model Evaluation**  
   - Compared model accuracy and selected the best-performing one  

---

## Key Insights

- Cars with **higher mileage** or **older production years** tend to be cheaper  
- **Brand reputation**, **engine size**, and **newer models** increase price  
- The model helps buyers and sellers understand how the market values different car features  
- Accurate price prediction promotes transparency and fair trading  

---

## Impact

- **Fair pricing:** Helps sellers set reasonable prices and protects buyers from overpaying  
- **Sustainable consumption:** Encourages buying used cars, extending product lifecycles, and reducing environmental impact  
- **Future integration:** The model can be incorporated into car-selling platforms to standardize pricing across Saudi Arabia  

---

## Usage

1. Open the notebook in Google Colab.  
2. Run cells step-by-step to reproduce data analysis and model predictions.  
3. Optional: Replace the dataset with updated car listings for fresh predictions.
4. To access our deployed application, please access the following URL:
    https://ramaezz-car-streamlit-app-app-uuebnk.streamlit.app/

---

## Technologies & Tools

- Python 3  
- Pandas, NumPy  
- Matplotlib, Seaborn for visualization  
- Scikit-learn for regression modeling  

- Google Colab for interactive coding  
