# Telco Churn Analytics - Complete Beginner's Guide

## 🎯 What This Project Does (In Simple Terms)

This project helps phone companies predict which customers might cancel their service. Think of it like a weather forecast, but instead of predicting rain, it predicts customer cancellations (called "churn").

The project looks at customer information like:
- How long they've been with the company
- How much they pay each month
- What services they use
- Whether they've called customer service

Then it uses this information to guess who might leave soon, so the company can try to keep them happy.

---

## 📋 What You'll Need Before Starting

**Required:**
- A computer (Mac, Windows, or Linux)
- Internet connection
- About 30 minutes of time

**Helpful but not required:**
- Basic computer skills (opening files, typing commands)
- Understanding of spreadsheets (like Excel)

---

## 🔧 Step 1: Install Python

### What is Python?
Python is a programming language that runs this project. Think of it like the engine in a car - the project won't work without it.

### How to Install:

#### For Mac Users:
1. Open the **Terminal** app (press `Command + Space`, type "Terminal", press Enter)
2. Type this command and press Enter:
   ```
   python3 --version
   ```
3. If you see a version number (like "Python 3.8.0"), you already have Python! Skip to Step 2.
4. If not, install Python by visiting: https://www.python.org/downloads/
5. Download the latest version and run the installer

#### For Windows Users:
1. Open **Command Prompt** (press `Windows + R`, type "cmd", press Enter)
2. Type this command and press Enter:
   ```
   python --version
   ```
3. If you see a version number, you have Python! Skip to Step 2.
4. If not, install Python by visiting: https://www.python.org/downloads/
5. Download the latest version and run the installer
6. **IMPORTANT:** During installation, check the box that says "Add Python to PATH"

---

## 📥 Step 2: Get the Project Files

### Option A: If you have the project files already
1. Make sure you have the `telco-churn-analytics` folder on your computer
2. Open your terminal/command prompt
3. Navigate to the project folder:
   ```
   cd path/to/telco-churn-analytics
   ```
   (Replace `path/to/` with the actual folder location)

### Option B: Download the project
1. Open your web browser
2. Go to the project's GitHub page
3. Click the green "Code" button, then "Download ZIP"
4. Extract the ZIP file to your computer
5. Open terminal/command prompt and navigate to the folder

---

## 🏗️ Step 3: Set Up the Project Environment

### What is a Virtual Environment?
Think of this like a clean workspace for your project. It keeps all the project's tools separate from other programs on your computer.

### How to Set Up:

1. **Open your terminal/command prompt**
2. **Navigate to the project folder** (if you haven't already):
   ```
   cd path/to/telco-churn-analytics
   ```

3. **Create the virtual environment:**
   ```
   python3 -m venv .venv
   ```
   (On Windows, use: `python -m venv .venv`)

4. **Activate the virtual environment:**
   
   **Mac/Linux:**
   ```
   source .venv/bin/activate
   ```
   
   **Windows:**
   ```
   .venv\Scripts\activate
   ```

5. **You'll know it worked when you see `(.venv)` at the beginning of your command line**

---

## 📦 Step 4: Install Project Dependencies

### What are Dependencies?
These are the tools and libraries that the project needs to work. Think of them like ingredients for a recipe.

### How to Install:

1. **Make sure your virtual environment is active** (you should see `(.venv)`)

2. **Install all dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Wait for it to finish** - this might take a few minutes
4. **If you see any error messages**, don't worry - most warnings are fine

---

## 🚀 Step 5: Run Your First Analysis

### The Easy Way (Recommended):
This runs everything automatically - data processing, model training, and predictions.

1. **Make sure you're in the project folder with virtual environment active**

2. **Run the complete pipeline:**
   ```
   python src/run_pipeline.py
   ```

3. **Watch it work!** You'll see messages like:
   - "STARTING END-TO-END PIPELINE"
   - "DATA PREPROCESSING"
   - "MODEL TRAINING"
   - "GENERATING PREDICTIONS"
   - "PIPELINE COMPLETED SUCCESSFULLY"

4. **When it's done, you'll see the results:**
   - Accuracy: ~79%
   - ROC-AUC: ~83%

---

## 📊 Step 6: Look at Your Results

### Where to Find Results:
- **Test Predictions**: `results/test_predictions.csv`
- **Model Comparison**: `models/model_comparison_results.csv`

### How to View Results:
1. **Open the results folder**
2. **Double-click `test_predictions.csv`** - it should open in Excel or a spreadsheet program
3. **You'll see columns like:**
   - CustomerID: Customer identification
   - Actual_Churn: Did they actually leave? (Yes/No)
   - Predicted_Churn: Did we predict they would leave? (0/1)
   - Churn_Probability: How likely they are to leave (0-1)

---

## 🎮 Step 7: Try Making Your Own Predictions

### Predict on New Customer Data:

1. **Use the existing data to test:**
   ```
   python src/predict.py --input data/raw/telco_customer_churn.csv --output my_predictions.csv
   ```

2. **Check your results:**
   - Open `my_predictions.csv` to see predictions for all customers

---

## 🧪 Step 8: Explore with Jupyter Notebooks (Optional)

### What are Jupyter Notebooks?
These are interactive documents that let you explore the data step by step. Great for learning!

### How to Use:

1. **Start Jupyter:**
   ```
   jupyter notebook notebooks/
   ```

2. **This will open a web browser** with the notebooks

3. **Run notebooks in order:**
   - `01_eda.ipynb` - Explore the data
   - `02_preprocessing.ipynb` - Clean and prepare data
   - `03_modeling.ipynb` - Train models
   - `04_results.ipynb` - Visualize results

4. **How to run a notebook:**
   - Click on a notebook file
   - Click "Run" button or press `Shift + Enter` to run each cell
   - Run cells in order from top to bottom

---

## 📁 Understanding the Project Structure

```
telco-churn-analytics/
├── data/
│   ├── raw/                    # Original customer data
│   │   └── telco_customer_churn.csv
│   └── processed/              # Cleaned and prepared data
├── src/                        # Main program files
│   ├── config.py              # Project settings
│   ├── utils.py               # Helper functions
│   ├── preprocess.py          # Data cleaning
│   ├── train.py               # Model training
│   ├── predict.py             # Making predictions
│   └── run_pipeline.py        # Complete workflow
├── notebooks/                  # Interactive exploration notebooks
├── models/                     # Trained models
└── results/                    # Prediction results
```

---

## 🛠️ Common Problems & Solutions

### Problem: "Command not found: python"
**Solution:** 
- Make sure Python is installed (Step 1)
- Try `python3` instead of `python` (on Mac/Linux)

### Problem: "pip is not recognized"
**Solution:**
- Make sure your virtual environment is active (Step 3)
- Look for `(.venv)` at the beginning of your command line

### Problem: "No module named 'pandas'"
**Solution:**
- Install dependencies again: `pip install -r requirements.txt`
- Make sure you're in the correct folder

### Problem: "FileNotFoundError"
**Solution:**
- Make sure you're in the correct project folder
- Check that the data file exists in `data/raw/`

### Problem: Jupyter won't open
**Solution:**
- Make sure you installed Jupyter: `pip install jupyter`
- Try opening manually: `jupyter notebook` then navigate to the notebooks folder

---

## 🎓 What You're Learning

By completing this guide, you've learned:
- How to set up a Python project
- How to use virtual environments
- How to install packages
- How machine learning predictions work
- How to interpret results

---

## 📞 Need Help?

### If Something Goes Wrong:
1. **Read the error message carefully** - it often tells you what's wrong
2. **Try the steps again** - make sure you didn't miss anything
3. **Check the troubleshooting section** above

### Common Questions:
**Q: Do I need to understand the code to use this?**
A: No! You can run the project without understanding the code. The guide shows you how.

**Q: Can I use my own data?**
A: Yes! The predict.py script can work with new customer data that has the same columns.

**Q: What if I want to change the predictions?**
A: You can adjust the prediction threshold in the predict.py file (look for `threshold=0.5`).

---

## 🎉 Congratulations!

You've successfully:
- ✅ Set up a Python machine learning project
- ✅ Trained a model to predict customer churn
- ✅ Generated predictions
- ✅ Analyzed results

You're now ready to explore more advanced features or try this with your own data!

---

## 📚 Next Steps (Optional)

If you want to learn more:
1. **Explore the Jupyter notebooks** to understand the data better
2. **Try different prediction thresholds** in the predict.py file
3. **Look at the model comparison results** to see which algorithm worked best
4. **Read about machine learning concepts** like accuracy, precision, and recall

Remember: Every expert was once a beginner. You've taken the first step into the exciting world of machine learning! 🚀
