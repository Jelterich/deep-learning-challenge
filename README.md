# deep-learning-challenge
Deep Learning Charity Predictor
Overview
This project involves building a binary classification model using a neural network to predict the success of organizations receiving funding from a nonprofit foundation, Alphabet Soup. The data includes various organizational features, such as application type, income classification, and funding amount requested.

The primary objectives of the project are:

Data Preprocessing: Clean and prepare the data for machine learning.
Model Development: Build and train a neural network using TensorFlow/Keras.
Optimization: Optimize the model to improve its accuracy beyond 75%.
Troubleshooting: Address and document issues faced during the development process.
Technologies Used
Python (3.8+)
Pandas
Scikit-Learn
TensorFlow/Keras
Jupyter Notebook / Google Colab
Steps to Run the Project
1. Prerequisites
Before running the code, ensure you have the following dependencies installed:

bash
Copy code
pip install pandas scikit-learn tensorflow
2. Data Preprocessing
Load Data: The dataset charity_data.csv is loaded into a Pandas DataFrame.
Drop Unnecessary Columns: Columns like EIN and NAME are removed as they do not contribute to the model.
Combine Rare Categories: Categorical columns with rare values are grouped into a single category named "Other".
Encode Categorical Data: Use pd.get_dummies() to convert categorical variables into numerical format.
Feature Scaling: Use StandardScaler to scale numerical features.
3. Model Creation and Training
Neural Network: A sequential model is created with:
Two hidden layers (80 and 30 nodes).
ReLU activation for hidden layers.
Sigmoid activation for the output layer.
Compilation: The model is compiled using the Adam optimizer and binary cross-entropy loss function.
Training: The model is trained for 50 epochs with a batch size of 32.
4. Optimization
To improve accuracy, the following optimizations are attempted:

Adjusting the number of hidden layers and neurons.
Experimenting with activation functions.
Changing the number of epochs.
Handling rare categories more effectively.
Troubles Faced During Development
During the development of the neural network, I encountered several challenges that impacted my progress:

1. TensorFlow Import Errors
Problem: Importing TensorFlow threw errors such as:
AttributeError: module 'tensorflow' has no attribute 'keras'
ModuleNotFoundError: No module named 'tensorflow.keras'
Cause: The system was using an outdated version of TensorFlow (1.x) where tf.keras does not exist.
Solution:
Upgraded TensorFlow to version 2.x:
bash
Copy code
pip uninstall tensorflow
pip install tensorflow --upgrade
Restarted the kernel to ensure changes took effect.
2. Naming Conflicts
Problem: Errors like module 'tensorflow' has no attribute '__version__' appeared.
Cause: A file named tensorflow.py in the working directory conflicted with the TensorFlow library.
Solution:
Identified the conflict by listing directory files:
python
Copy code
import os
print(os.listdir())
Renamed or removed the conflicting file.
3. Model Training Issues
Problem: Model accuracy was initially below 70%, even after preprocessing.
Solution:
Optimized the model by:
Adding more neurons to the hidden layers.
Increasing epochs for better training.
Fine-tuning activation functions.
Results
Final Test Accuracy: Achieved ~75% accuracy after optimizations.
Model File: The final model is saved as AlphabetSoupCharity.h5.
How to Use the Model
Run the preprocessing and model training code.
Evaluate the model using test data.
Load the model to make predictions on new data:
python
Copy code
from tensorflow.keras.models import load_model

model = load_model("AlphabetSoupCharity.h5")
predictions = model.predict(new_data)
Conclusion
This project demonstrates the ability to preprocess data, design a neural network, and optimize its performance. Despite challenges with TensorFlow and model accuracy, persistence and troubleshooting led to a functioning model.