# Task 4: Machine Learning Model Comparison

## 📋 Project Overview

This project builds, trains, and compares two popular machine learning models: **Logistic Regression** and **Random Forest Classifier**. The comparison is performed on the Titanic dataset to predict passenger survival, evaluating models using multiple performance metrics including accuracy, precision, recall, F1-score, and AUC-ROC.

**Intern:** AI/ML Research Intern - OWLAI  
**Task:** Task 4 - Machine Learning Model Comparison  
**Dataset:** Titanic: Machine Learning from Disaster  
**Problem Type:** Binary Classification

---

## 🎯 Objectives

1. **Data Preprocessing**: Handle missing values, encode categorical variables, and engineer features
2. **Model Building**: Implement Logistic Regression and Random Forest Classifier
3. **Model Training**: Train both models on the same dataset with proper train-test split
4. **Model Evaluation**: Evaluate models using comprehensive metrics
5. **Performance Comparison**: Compare models using accuracy, precision, recall, F1-score, and AUC
6. **Visualization**: Create visual comparisons including ROC curves and confusion matrices
7. **Final Recommendation**: Determine the best model based on evaluation results

---

## 📊 Dataset Information

**Source:** [Kaggle - Titanic Dataset](https://www.kaggle.com/c/titanic/data)

**Problem Statement:** Predict whether a passenger survived the Titanic disaster based on features like age, gender, class, etc.

**Target Variable:** `Survived` (0 = Did Not Survive, 1 = Survived)

**Features Used:**
- `Pclass`: Passenger class (1, 2, 3)
- `Sex`: Gender (male, female)
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Passenger fare
- `Embarked`: Port of embarkation (C, Q, S)

**Engineered Features:**
- `FamilySize`: Total family members (SibSp + Parch + 1)
- `IsAlone`: Binary indicator if traveling alone

---

## 🤖 Models Compared

### Model 1: Logistic Regression

**Type:** Linear Classification Model

**Characteristics:**
- Simple and interpretable
- Fast training and prediction
- Works well for linearly separable data
- Provides probability estimates
- Lower computational cost

**Use Cases:**
- When interpretability is important
- When you have limited computational resources
- When the relationship between features and target is approximately linear

### Model 2: Random Forest Classifier

**Type:** Ensemble Learning Model (Decision Trees)

**Characteristics:**
- Handles non-linear relationships
- Robust to outliers and noise
- Provides feature importance
- Generally higher accuracy
- Reduces overfitting through averaging

**Use Cases:**
- When you need high accuracy
- When data has complex non-linear patterns
- When you want to understand feature importance
- When you have sufficient computational resources

---

## 🛠️ Tools & Technologies

### Programming Language
- **Python 3.8+**

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

### Machine Learning Libraries
- **scikit-learn**: Machine learning algorithms and tools
  - `LogisticRegression`: Linear classification model
  - `RandomForestClassifier`: Ensemble tree-based model
  - `train_test_split`: Data splitting
  - `StandardScaler`: Feature scaling
  - `LabelEncoder`: Categorical encoding
  - Performance metrics: accuracy, precision, recall, f1-score, ROC-AUC
  - `cross_val_score`: Cross-validation
  - `GridSearchCV`: Hyperparameter tuning (optional)

### Installation

```bash
# Install all required libraries
pip install pandas numpy matplotlib seaborn scikit-learn

# Or use requirements.txt
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
Task4_ML_Model_Comparison/
│
├── ml_model_comparison.ipynb  # Main Jupyter notebook
├── README.md                   # Project documentation (this file)
├── requirements.txt            # Python dependencies
├── train.csv                   # Titanic dataset (download from Kaggle)
└── outputs/                    # Saved plots and results (optional)
    ├── confusion_matrix_lr.png
    ├── confusion_matrix_rf.png
    ├── roc_curve_comparison.png
    ├── model_comparison.png
    └── feature_importance.png
```

---

## 📥 How to Download the Dataset

### Method 1: Manual Download
1. Visit [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)
2. Download `train.csv`
3. Place it in the same directory as the notebook

### Method 2: Kaggle API
```bash
# Install Kaggle
pip install kaggle

# Download dataset
kaggle competitions download -c titanic

# Unzip
unzip titanic.zip
```

---

## 🚀 How to Run

### Step 1: Setup Environment
```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the Notebook
```bash
# Start Jupyter Notebook
jupyter notebook

# Open ml_model_comparison.ipynb
# Run all cells: Cell > Run All
```

### Step 3: Alternative - Google Colab
1. Upload notebook to Google Colab
2. Upload `train.csv` dataset
3. Run all cells

---

## 📊 Workflow Overview

### 1. Data Preprocessing (Cells 1-11)
- Load dataset
- Handle missing values
- Encode categorical variables
- Engineer new features
- Visualize data distributions

### 2. Feature Selection & Scaling (Cells 12-15)
- Correlation analysis
- Select relevant features
- Split into train/test sets (80/20)
- Apply StandardScaler for Logistic Regression

### 3. Model 1: Logistic Regression (Cells 16-20)
- Train model
- Make predictions
- Evaluate on train and test sets
- Generate confusion matrix
- Classification report

### 4. Model 2: Random Forest (Cells 21-26)
- Train model with 100 trees
- Make predictions
- Evaluate on train and test sets
- Generate confusion matrix
- Analyze feature importance

### 5. Model Comparison (Cells 27-33)
- Side-by-side metrics comparison
- Visualize performance differences
- ROC curve comparison
- Precision-Recall curves
- Cross-validation analysis
- Overfitting detection

### 6. Final Recommendation (Cell 34)
- Determine best model
- Provide reasoning
- Summary statistics

---

## 📈 Evaluation Metrics Explained

### 1. **Accuracy**
- Formula: (TP + TN) / (TP + TN + FP + FN)
- Percentage of correct predictions
- Good for balanced datasets

### 2. **Precision**
- Formula: TP / (TP + FP)
- Of all predicted survivors, how many actually survived?
- Important when false positives are costly

### 3. **Recall (Sensitivity)**
- Formula: TP / (TP + FN)
- Of all actual survivors, how many did we correctly identify?
- Important when false negatives are costly

### 4. **F1-Score**
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean of precision and recall
- Good overall metric for imbalanced datasets

### 5. **AUC-ROC**
- Area Under the Receiver Operating Characteristic Curve
- Measures model's ability to distinguish between classes
- Range: 0 to 1 (higher is better)

### 6. **Confusion Matrix**
```
                Predicted
              0         1
Actual  0    TN        FP
        1    FN        TP
```

---

## 🔍 Expected Results

### Typical Performance (may vary)

**Logistic Regression:**
- Test Accuracy: ~78-81%
- Test Precision: ~76-80%
- Test Recall: ~68-72%
- Test F1-Score: ~72-76%
- AUC: ~0.82-0.86

**Random Forest:**
- Test Accuracy: ~80-84%
- Test Precision: ~78-83%
- Test Recall: ~72-76%
- Test F1-Score: ~75-79%
- AUC: ~0.85-0.89

**Note:** Random Forest typically outperforms Logistic Regression on this dataset due to non-linear patterns in the data.

---

## 💡 Key Insights

### Model Performance
1. **Random Forest** generally achieves higher accuracy and F1-score
2. **Logistic Regression** is faster and more interpretable
3. Both models show good generalization (low overfitting)

### Feature Importance (from Random Forest)
Top predictive features typically include:
1. Sex (Gender) - Most important
2. Pclass (Passenger Class)
3. Fare
4. Age
5. FamilySize

### Business Insights
- Gender was the strongest predictor of survival
- Higher-class passengers had better survival rates
- Family size played a role in survival chances
- Age groups showed different survival patterns

---

## 🎓 Skills Demonstrated

### Technical Skills
- ✅ Data preprocessing and cleaning
- ✅ Feature engineering
- ✅ Categorical encoding
- ✅ Train-test splitting
- ✅ Feature scaling
- ✅ Model implementation
- ✅ Hyperparameter setting
- ✅ Model evaluation
- ✅ Cross-validation
- ✅ Overfitting analysis

### Machine Learning Concepts
- ✅ Supervised learning
- ✅ Binary classification
- ✅ Linear vs ensemble models
- ✅ Bias-variance tradeoff
- ✅ Model comparison
- ✅ Performance metrics
- ✅ ROC analysis
- ✅ Feature importance

### Data Science Skills
- ✅ Exploratory data analysis
- ✅ Statistical analysis
- ✅ Data visualization
- ✅ Result interpretation
- ✅ Professional documentation

---

## 🔧 Customization Options

### Adjust Model Parameters

**Logistic Regression:**
```python
log_reg = LogisticRegression(
    random_state=42,
    max_iter=1000,
    C=1.0,              # Regularization strength
    penalty='l2',       # L1 or L2 regularization
    solver='lbfgs'      # Optimization algorithm
)
```

**Random Forest:**
```python
rf_clf = RandomForestClassifier(
    n_estimators=200,      # Number of trees
    max_depth=15,          # Maximum tree depth
    min_samples_split=10,  # Minimum samples to split
    min_samples_leaf=4,    # Minimum samples in leaf
    random_state=42
)
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10, 15]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1'
)
grid_search.fit(X_train, y_train)
```

---

## 📚 Learning Resources

### Logistic Regression
- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Understanding Logistic Regression](https://machinelearningmastery.com/logistic-regression-for-machine-learning/)

### Random Forest
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Random Forest Algorithm Explained](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/)

### Model Evaluation
- [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [ROC and AUC Explained](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

---

## 🚀 Future Enhancements

1. **Try More Models:**
   - Support Vector Machines (SVM)
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks

2. **Advanced Feature Engineering:**
   - Extract titles from names (Mr., Mrs., Master, etc.)
   - Create deck information from cabin numbers
   - Binning continuous variables

3. **Hyperparameter Optimization:**
   - Grid Search CV
   - Random Search CV
   - Bayesian Optimization

4. **Ensemble Methods:**
   - Voting Classifier
   - Stacking
   - Blending

5. **Advanced Evaluation:**
   - Learning curves
   - Validation curves
   - Calibration plots

---

## ⚠️ Common Issues & Solutions

### Issue 1: Import Errors
```bash
# Solution: Install missing libraries
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Issue 2: Dataset Not Found
```bash
# Solution: Ensure train.csv is in the same directory
# Or specify full path:
df = pd.read_csv('/path/to/train.csv')
```

### Issue 3: Memory Error
```python
# Solution: Reduce n_estimators for Random Forest
rf_clf = RandomForestClassifier(n_estimators=50)
```

### Issue 4: Poor Performance
- Check for data leakage
- Ensure proper train-test split
- Verify feature scaling
- Try different random_state values

---

## 📊 Performance Comparison Summary

| Metric | Logistic Regression | Random Forest | Winner |
|--------|---------------------|---------------|--------|
| Accuracy | ~79% | ~82% | RF |
| Precision | ~78% | ~81% | RF |
| Recall | ~70% | ~74% | RF |
| F1-Score | ~74% | ~77% | RF |
| AUC-ROC | ~0.84 | ~0.87 | RF |
| Training Time | Fast | Slower | LR |
| Interpretability | High | Medium | LR |
| Overfitting Risk | Low | Medium | LR |

**Recommendation:** Random Forest for better accuracy, Logistic Regression for interpretability and speed.

---

## 👤 Author

**AI/ML Research Intern - OWLAI**  
**Task:** Task 4 - Machine Learning Model Comparison  
**Date:** January 2026

---

## 📄 License

This project is for educational purposes as part of the OWLAI internship program.

---

## 🙏 Acknowledgments

- Kaggle for the Titanic dataset
- Scikit-learn developers
- OWLAI for the internship opportunity
- The machine learning community

---

## 📧 Submission Instructions

1. **GitHub Upload:**
   - Create a new repository: `OWLAI-Task4-ML-Model-Comparison`
   - Upload notebook, README, and requirements.txt
   - Share the repository link

2. **LinkedIn Post:**
   - Share a screenshot of model comparison results
   - Tag @OWLAI official page
   - Use hashtags: #MachineLearning #OWLAI #Internship

3. **Documentation:**
   - Ensure all cells run without errors
   - Include comments and explanations
   - Professional formatting

---

**Note:** This project demonstrates a complete machine learning workflow from data preprocessing to model comparison. The notebook contains 34 cells with comprehensive analysis, visualizations, and detailed comments suitable for portfolio presentation.