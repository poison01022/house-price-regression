# 🏠 House Price Prediction - Linear Regression

This project implements **Simple & Multiple Linear Regression** using the **Boston Housing Dataset** to predict house prices. The implementation is done using Python libraries like `scikit-learn`, `pandas`, and `matplotlib`.

---

## 📁 Project Structure

```arduino
House_Price_Regression/
├── app.py # Main Python script for model training & evaluation
├── README.md # Project documentation

```

---

## 📊 Objective

- Understand and implement linear regression
- Train and evaluate predictive models
- Visualize actual vs predicted results
- Interpret model coefficients

---

## 🧰 Tools & Libraries

- Python 3.x
- scikit-learn
- pandas
- matplotlib
- seaborn (for heatmaps, optional)

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## 📥 Dataset

We used the **Boston Housing Dataset**, a well-known regression dataset.

```python
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
```

## 🚀 How to Run

1. Clone the repository or download the files.  
2. Make sure `app.py` is in your working directory.  
3. Run the script:

```bash
python app.py
```

## 📈 Key Features

- Split dataset into **training and testing** sets.
- Fit a **Linear Regression** model using:

  ```python
  sklearn.linear_model.LinearRegression
  ```

## ✅ Model Evaluation

- Evaluate model performance using:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **R² Score**

- Plot **Actual vs Predicted** house prices.
- Output **Feature Importance** via coefficients.

---

## 🧪 Sample Output

You can optionally save plots and results in an `outputs/` folder using:

```python
plt.savefig("outputs/actual_vs_predicted.png")
```
![Image](https://github.com/user-attachments/assets/96a0b007-a5ec-4183-85ee-25cb89263f33)
![Image](https://github.com/user-attachments/assets/cd9f67c9-2b41-42c7-85d0-c8d0599e14c7)



## 🧠 Example Insights

- Features like `RM` (average number of rooms) have a **strong positive** effect on price.
- `LSTAT` (lower status of population) has a **strong negative** influence.
- A **high R² score** indicates a good model fit.
