# Task 3: Exploratory Data Analysis (EDA) on Titanic Dataset

## 📋 Project Overview

This project performs comprehensive Exploratory Data Analysis (EDA) on the famous Titanic dataset from Kaggle. The analysis explores patterns, relationships, and insights about passenger survival rates during the Titanic disaster.

**Intern:** AI/ML Research Intern - OWLAI  
**Task:** Task 3 - Exploratory Data Analysis  
**Dataset:** Titanic: Machine Learning from Disaster

---

## 📊 Dataset Information

**Source:** [Kaggle - Titanic Dataset](https://www.kaggle.com/c/titanic/data)

**Description:** The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered "unsinkable" RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren't enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

**Dataset Size:** 891 passengers (training set)

**Features:**
- `PassengerId`: Unique ID for each passenger
- `Survived`: Survival status (0 = No, 1 = Yes) - **Target Variable**
- `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Name`: Passenger name
- `Sex`: Gender
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Passenger fare
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## 🎯 Objectives

1. **Data Understanding**: Explore the structure and characteristics of the dataset
2. **Missing Value Analysis**: Identify and visualize missing data patterns
3. **Statistical Analysis**: Generate descriptive statistics for all features
4. **Survival Analysis**: Investigate factors affecting passenger survival
5. **Feature Relationships**: Discover correlations and patterns between variables
6. **Data Visualization**: Create comprehensive visualizations for insights
7. **Key Insights**: Summarize findings and actionable insights

---

## 🛠️ Tools & Technologies

### Programming Language
- **Python 3.8+**

### Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **warnings**: Suppress warning messages

### Installation

```bash
# Install required libraries
pip install pandas numpy matplotlib seaborn

# Or use requirements.txt
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
Task3_EDA/
│
├── eda_titanic.ipynb          # Main Jupyter notebook with EDA
├── README.md                   # Project documentation (this file)
├── requirements.txt            # Python dependencies
├── train.csv                   # Titanic dataset (download from Kaggle)
└── outputs/                    # Folder for saving plots (optional)
    ├── missing_values.png
    ├── survival_rate.png
    ├── gender_analysis.png
    └── ...
```

---

## 📥 How to Download the Dataset

1. Go to [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)
2. Click on "Download All" or download `train.csv` separately
3. Place the `train.csv` file in the same directory as the notebook
4. If you don't have a Kaggle account, create one (it's free!)

**Alternative:** Use Kaggle API
```bash
# Install Kaggle API
pip install kaggle

# Download dataset
kaggle competitions download -c titanic

# Unzip the file
unzip titanic.zip
```

---

## 🚀 How to Run

### Option 1: Jupyter Notebook
```bash
# Start Jupyter Notebook
jupyter notebook

# Open eda_titanic.ipynb
# Run cells sequentially (Cell > Run All)
```

### Option 2: JupyterLab
```bash
# Start JupyterLab
jupyter lab

# Open eda_titanic.ipynb
# Execute all cells
```

### Option 3: Google Colab
1. Upload `eda_titanic.ipynb` to Google Colab
2. Upload `train.csv` dataset
3. Run all cells

---

## 📊 Analysis Breakdown

### 1. **Data Loading & Initial Exploration** (Cells 1-7)
- Import libraries
- Load dataset
- Display first/last rows
- Check data types and structure
- View random samples

### 2. **Statistical Summary** (Cells 8-9)
- Descriptive statistics for numerical features
- Summary of all columns including categorical

### 3. **Missing Value Analysis** (Cells 10-12)
- Identify missing values
- Calculate missing percentages
- Visualize missing data patterns
- Check for duplicate records

### 4. **Target Variable Analysis** (Cells 13-14)
- Survival rate distribution
- Count and proportion of survivors vs non-survivors
- Visualization: bar charts and pie charts

### 5. **Categorical Feature Analysis** (Cells 15-23)
- **Gender Analysis**: Distribution and survival rate by gender
- **Passenger Class**: Distribution and survival rate by class
- **Embarkation Port**: Distribution and survival rate by port

### 6. **Numerical Feature Analysis** (Cells 19-21)
- **Age Distribution**: Histograms, box plots, and survival comparison
- **Fare Distribution**: Statistical summary and visualizations

### 7. **Feature Engineering** (Cells 24-26)
- Create FamilySize feature (SibSp + Parch + 1)
- Analyze SibSp and Parch separately
- Survival analysis by family composition

### 8. **Correlation Analysis** (Cells 27-28)
- Calculate correlation matrix
- Create correlation heatmap
- Identify strong relationships

### 9. **Multi-variable Analysis** (Cells 29-31)
- Class + Gender + Survival
- Age groups and survival
- Fare vs Class comparison

### 10. **Outlier Detection** (Cell 32)
- Identify outliers using IQR method
- Analyze Fare and Age outliers

### 11. **Final Summary** (Cells 33-34)
- Comprehensive survival summary table
- Key insights and findings

---

## 🔍 Key Findings

### Overall Statistics
- **Total Passengers:** 891
- **Survival Rate:** ~38.4%
- **Survivors:** 342
- **Non-Survivors:** 549

### Gender Insights
- **Female Survival Rate:** ~74.2%
- **Male Survival Rate:** ~18.9%
- Females were approximately **3.9x more likely** to survive than males
- Clear evidence of "women and children first" policy

### Class Analysis
- **1st Class Survival Rate:** ~63%
- **2nd Class Survival Rate:** ~47%
- **3rd Class Survival Rate:** ~24%
- Higher class passengers had significantly better survival chances

### Age Patterns
- Children (Age < 12) had higher survival rates
- Average age of survivors: ~28 years
- Average age of non-survivors: ~30 years

### Family Size
- Passengers with small families (2-4 members) had better survival rates
- Solo travelers and very large families had lower survival rates

### Missing Data
- **Age:** 177 missing (19.87%)
- **Cabin:** 687 missing (77.10%)
- **Embarked:** 2 missing (0.22%)

---

## 📈 Visualizations Created

1. **Missing Values Heatmap**: Shows missing data patterns
2. **Survival Count & Pie Chart**: Overall survival distribution
3. **Gender vs Survival**: Bar charts and percentage plots
4. **Class vs Survival**: Count and rate comparisons
5. **Age Distribution**: Histograms and box plots
6. **Age vs Survival**: Overlapped histograms
7. **Fare Distribution**: Histogram and box plot
8. **Embarkation Analysis**: Count and survival rates
9. **Family Size Analysis**: Survival by family composition
10. **Correlation Heatmap**: Numerical feature relationships
11. **Multi-variable Plots**: Class + Gender combinations
12. **Age Groups**: Categorized age analysis
13. **Fare vs Class**: Box plots and violin plots

---

## 💡 Actionable Insights

1. **Passenger Class was a Strong Predictor**: Higher-class passengers had much better access to lifeboats
2. **Gender Bias in Rescue**: The "women and children first" protocol was clearly followed
3. **Age Mattered**: Children were prioritized during the rescue operations
4. **Optimal Family Size**: Being with 1-3 family members improved survival chances
5. **Embarkation Port Correlation**: Port C (Cherbourg) passengers had higher survival, possibly due to higher proportion of 1st class passengers

---

## 🎓 Skills Demonstrated

- ✅ Data loading and preprocessing
- ✅ Handling missing values
- ✅ Descriptive statistics
- ✅ Data visualization techniques
- ✅ Feature engineering
- ✅ Correlation analysis
- ✅ Outlier detection
- ✅ Multi-variate analysis
- ✅ Insight generation
- ✅ Professional documentation

---

## 📝 Future Work

- Impute missing Age values using advanced techniques
- Extract titles from Name feature (Mr., Mrs., Miss, etc.)
- Analyze Cabin information more deeply
- Create derived features for better prediction
- Perform hypothesis testing for statistical significance

---

## 📚 References

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

---

## 👤 Author

**AI/ML Research Intern - OWLAI**  
**Task:** Task 3 - Exploratory Data Analysis  
**Date:** January 2026

---

## 📄 License

This project is for educational purposes as part of the OWLAI internship program.

---

## 🙏 Acknowledgments

- Kaggle for providing the dataset
- OWLAI for the internship opportunity
- The data science community for inspiration

---

## 📧 Contact

For questions or feedback about this analysis:
- Post on LinkedIn with tag: @OWLAI
- Share your GitHub repository link

---

**Note:** This EDA is comprehensive and covers all major aspects of exploratory data analysis. The notebook contains 34 cells with detailed comments, visualizations, and insights. Each cell focuses on a specific aspect of the analysis, making it easy to understand and follow.