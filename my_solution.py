import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class MySolutions:
    """ a kNN classifier with L2 distance """

    def __init__(self, X_train, y_train, X_test):
        self.X_train_original = X_train
        self.y_train_original = y_train
        self.X_test_original = X_test
        pass

    def get_simple_logistic_reg(self):
        """
        Train the simple Logistic Regression classifier (from sklearn) with default hyperparameters.
        Return predicted class probabilities on the training and testing data.

        Returns:
            Predicted class probabilities and labels for train and test data in format:
            train_predicted_labels, train_predicted_probas, test_predicted_labels, test_predicted_probas
        """
        # some imports if needed
        ### YOUR CODE HERE
        model = LogisticRegression()
        model.fit(self.X_train_original, self.y_train_original)
        
        # Predict class probabilities
        probas_train = model.predict_proba(self.X_train_original)
        probas_test = model.predict_proba(self.X_test_original)
        
        # Predict labels
        labels_train = model.predict(self.X_train_original)
        labels_test = model.predict(self.X_test_original)
        
        return labels_train, probas_train, labels_test, probas_test
    
    def get_simple_naive_bayes(self):
        """
        Train the Naive Bayes classifier with Normal distribution as a prior.
        Use sklearn version (correct one!) and default hyperparameters.
        
        Returns:
            Predicted class probabilities for train and test data.
        """
        # some imports if needed
        ### YOUR CODE HERE

        # Initialize and train Gaussian Naive Bayes model
        model = GaussianNB()
        model.fit(self.X_train_original, self.y_train_original)
        
        # Predict class probabilities
        probas_train = model.predict_proba(self.X_train_original)
        probas_test = model.predict_proba(self.X_test_original)
        
        # Predict labels
        labels_train = model.predict(self.X_train_original)
        labels_test = model.predict(self.X_test_original)
        
        return labels_train, probas_train, labels_test, probas_test


    def get_best_solution(self):
        """
        Train your best model. You can run some preprocessing (analysing the dataset might be useful),
        normalize the data, use nonlinear model etc. Get highscore!
        Please, do not use any external libraries but sklearn and numpy.

        Returns:
            Predicted class probabilities for train and test data.
        """
        # some imports if needed
        ### YOUR CODE HERE

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train_original)
        X_test_scaled = scaler.transform(self.X_test_original)
        
        # Initialize and train SVM classifier
        model = SVC(probability=True)
        model.fit(X_train_scaled, self.y_train_original)
        
        # Predict class probabilities
        probas_train = model.predict_proba(X_train_scaled)
        probas_test = model.predict_proba(X_test_scaled)
        
        # Predict labels
        labels_train = model.predict(X_train_scaled)
        labels_test = model.predict(X_test_scaled)
        
        return labels_train, probas_train, labels_test, probas_test
