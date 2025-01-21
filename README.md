# MACHINE-LEARNING-MODEL-IMPLEMENTATION

**COMPANY** : CODETECH IT SOLUTIONS

**NAME** : ANIKET KUMAR PRASAD

**INTERN ID** : CT12WLSO

**BATCH DURATION** : January 10th, 2025 to April 10th, 2025

**MENTOR NAME** : Neela Santhosh

# DESCRIPTION OF PROJECT

Project Title: **Car Buyer Predictor Using Machine Learning**

**Objective**:
The goal of this project is to predict whether a person will purchase a car based on their social network activity. The model uses features like the user's age, gender, and estimated salary to predict whether they will buy a car or not.

**Dataset**:
The dataset contains information about individuals who were targeted with social network ads for a car. The dataset includes the following columns:
- **User ID**: A unique identifier for each user.
- **Gender**: The gender of the user (Male/Female).
- **Age**: The age of the user.
- **EstimatedSalary**: The estimated annual salary of the user.
- **Purchased**: The target variable that indicates whether the user purchased the car (1 for purchased, 0 for not purchased).

**Steps Involved**:
1. **Data Preprocessing**: 
   - The dataset is cleaned and prepared for model training. Only relevant features are selected for input, and the target column (Purchased) is used as the output.
   
2. **Feature Selection**: 
   - The features used to train the model are Age and EstimatedSalary, while the target column is Purchased.
   
3. **Data Splitting**:
   - The dataset is split into two parts: one for training the model and the other for testing it. This is done using a 75%-25% split using train_test_split().

4. **Feature Scaling**:
   - To make sure all the input features are on the same scale, the features are standardized using StandardScaler. This ensures that all the values of the features lie within the same range, which helps improve the model's performance.

5. **Model Training**:
   - A **Logistic Regression** model is used to predict the binary outcome (whether a person will buy a car or not). The model is trained using the training dataset (x_train and y_train).

6. **Model Prediction**:
   - After the model is trained, it is tested on the test data (x_test) to predict whether the users in the test set will buy the car or not. The predicted results are compared to the actual values to evaluate the model's performance.

7. **Visualization**:
   - The results are visualized using scatter plots to compare the actual and predicted outcomes. Three scatter plots are created:
     - **Actual vs Age**: Shows actual purchase outcomes against age.
     - **Predicted vs Age**: Shows predicted purchase outcomes against age.
     - **Color-coded Actual vs Age**: Highlights the correct and incorrect predictions using color-coding.

8. **Model Evaluation**:
   - The performance of the model is evaluated using **accuracy** and **confusion matrix**. Accuracy indicates how many predictions were correct, while the confusion matrix helps understand the distribution of correct and incorrect predictions.

**Tools & Libraries Used**:
- **Python**: The primary programming language.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For implementing machine learning algorithms like Logistic Regression, data splitting, and scaling.

**Outcome**:
The project predicts whether a person is likely to purchase a car based on their age and estimated salary. The Logistic Regression model, after training, can accurately predict the purchasing decision, as verified by the accuracy score and confusion matrix.

**Tools & Libraries Used**:
Python: The primary programming language.
Pandas: For data manipulation and analysis.
NumPy: For numerical operations.
Matplotlib: For data visualization.
Scikit-learn: For implementing machine learning algorithms like Logistic Regression, data splitting, and scaling.

**Resources Used**:
LearnVern: For understanding machine learning concepts and data preprocessing techniques.
W3Schools: For learning basic Python, Pandas, and Scikit-learn functionalities.
YouTube: For tutorials and practical implementation guides on machine learning models.
Google Bard: For additional explanations and resources related to machine learning concepts.
ChatGPT: For quick problem-solving, code explanations, and project guidance.

# OUTPUT OF THE TASK

![Image](https://github.com/user-attachments/assets/a7c9374f-42d1-4cb8-a8a5-a752a910e190)
![Image](https://github.com/user-attachments/assets/c9363fb5-8247-4032-9f06-08b1e00e9726)
![Image](https://github.com/user-attachments/assets/a9ac58b6-3dd5-4ed8-999d-db08015c628c)
![Image](https://github.com/user-attachments/assets/c26de94e-9eb9-412c-8d76-1d0d011141ca)
![Image](https://github.com/user-attachments/assets/0e57f92f-30d5-4b97-a6af-c2c126371367)

