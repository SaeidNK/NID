from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import pandas 
def Train(df, preprocessor, classifier):
    target = 'class' 
    # Create a pipeline that combines the preprocessor with a classifier
    print(classifier )
    clf1 = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', classifier)])

    # Split the data into features (X) and target variable (y)
    X = df.drop(columns=[target])
    y = df[target]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the pipeline (including preprocessing) to the training data
    clf1.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf1.predict(X_test)

    # Evaluate the model
    #result=classification_report(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_d = pandas.DataFrame(report).transpose()
    column_name= 'Title'
    column_data=['anomaly','normal','accuracy','macro avg', 'weighted avg']
    report_d.insert(0, column_name, column_data)
    html_table = report_d.to_html(classes='data', header="true", index=False)
    #print(html_table)

    #report_dict = classification_report(y_test, y_pred, output_dict=True)

    #-------------------------------------------------------------saving model
    # Dictionary mapping classifiers to model names
    classifier_names = {
        LogisticRegression: 'LogisticRegression',
        DecisionTreeClassifier: 'DecisionTree',
        RandomForestClassifier: 'RandomForest',
        SVC: 'SVM',
        GradientBoostingClassifier: 'GradientBoosting',
        KNeighborsClassifier: 'KNeighbors'
    }

    # Get the model name based on the classifier
    model_name = classifier_names.get(type(classifier), 'Unknown')

    # Save the model to a file
    file_name = f'model_{model_name}.pkl'
    joblib.dump(clf1, file_name)
    return  html_table