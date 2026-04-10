# 🩸 Diabetes Risk Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Deployment](https://img.shields.io/badge/Deployed%20on-Hugging%20Face-yellow)

## 📌 Project Overview
The **Diabetes Risk Predictor** is an end-to-end Machine Learning web application designed to predict a user's likelihood of developing diabetes based on key medical diagnostic variables. 

By taking user inputs such as Glucose levels, Body Mass Index (BMI), Age, and Blood Pressure, the pre-trained machine learning model instantly evaluates the data and returns a diagnostic probability. This project serves as a practical demonstration of how predictive analytics can be utilized for early healthcare screening.

🚀 **[Try the Live Application on Hugging Face Here!](https://huggingface.co/spaces/Pathik-Kundu/Diabetes-predictor)**

---

## 🛠️ Technologies Used
* **Programming Language:** Python
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Model Serialization:** Pickle
* **Web Framework / UI:** Gradio / Streamlit *(Used for Hugging Face deployment)*
* **Deployment Platform:** Hugging Face Spaces

---

## 📂 Repository Structure
Here is a breakdown of the files included in this repository and what they do:

* `app.py`: The main application script that powers the web interface and handles the user input to model prediction pipeline.
* `rc_model.py`: The Python script used for data preprocessing, training the machine learning algorithm, and evaluating its accuracy.
* `diabetes_model.pkl`: The saved (serialized) pre-trained machine learning model. This allows the web app to make predictions without needing to retrain the model every time.
* `diabetes.csv`: The dataset used to train and test the predictive model (contains historical patient diagnostic metrics).
* `requirements.txt`: A list of all the Python libraries and dependencies required to run the project.
* `venv/`: The virtual environment directory for managing local dependencies safely.

---

## 💻 How to Run the Project Locally

If you want to download this project and run it on your own machine, follow these steps:

### 1. Clone the Repository
Open your terminal and run:
```bash
git clone [https://github.com/Pathik-Kundu/Diabets_checking.git](https://github.com/Pathik-Kundu/Diabets_checking.git)
cd Diabets_checking
