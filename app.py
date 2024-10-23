import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns
from warnings import filterwarnings
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from flask import Flask, request, render_template, send_file
import os
import pdfkit
import matplotlib
matplotlib.use('Agg')



filterwarnings(action='ignore')

wine = pd.read_csv("https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-white.csv", sep=';')
wine.sample(25)
wine.info()
wine.describe()
wine.isnull().sum()
wine.groupby('quality').mean()

# Data Analysis

# sns.countplot(wine['quality'])
# plt.show()

wine.plot(kind ='box',subplots = True, layout =(4,4),sharex = False)

wine['fixed acidity'].plot(kind ='box')

wine.hist(figsize=(10,10),bins=50)
plt.show()

# Feature Selection
wine.sample(5)

wine['quality'].unique()

# If wine quality is 7 or above then will consider as good quality wine
wine['goodquality'] = [1 if x >= 7 else 0 for x in wine['quality']]
wine.sample(5)

# See total number of good vs bad wines samples
wine['goodquality'].value_counts()

# Separate depedent and indepedent variables
X = wine.drop(['quality','goodquality'], axis = 1)
Y = wine['goodquality']

X
# print("Quality", Y)

# Feature Importance
classifiern = ExtraTreesClassifier()
classifiern.fit(X,Y)
score = classifiern.feature_importances_
print("Score Name",score)

# Splitting Dataset

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)

model_res=pd.DataFrame(columns=['Model', 'Score'])

# LogisticRegression

model = LogisticRegression()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)


# accuracy_score(Y_test,Y_pred)
model_res.loc[len(model_res)] = ['LogisticRegression', accuracy_score(Y_test,y_pred)]
model_res

# Using KNN

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)


model_res.loc[len(model_res)] = ['KNeighborsClassifier', accuracy_score(Y_test,y_pred)]
model_res

# Using SVM

model = SVC()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))
model_res.loc[len(model_res)] = ['SVC', accuracy_score(Y_test,y_pred)]
model_res

# Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=7)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))
model_res.loc[len(model_res)] = ['DecisionTreeClassifier', accuracy_score(Y_test,y_pred)]
model_res

# Using GaussianNB
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train,Y_train)
y_pred = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))
model_res.loc[len(model_res)] = ['GaussianNB', accuracy_score(Y_test,y_pred)]
model_res

# Using Random Forest
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, Y_train)
y_pred = model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))
model_res.loc[len(model_res)] = ['RandomForestClassifier', accuracy_score(Y_test,y_pred)]
model_res


# Using Xgboost
import xgboost as xgb
model5 = xgb.XGBClassifier(random_state=1)
model5.fit(X_train, Y_train)
y_pred = model5.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))
model_res.loc[len(model_res)] = ['XGBClassifier', accuracy_score(Y_test,y_pred)]
model_res

model_res = model_res.sort_values(by='Score', ascending=False)
model_res


# ### Web Interface for Dataset Upload and Display ####

app = Flask(__name__)

# Specify the path to wkhtmltopdf executable
path_to_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'  # Update this path as per your installation
config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        df = pd.read_csv(file)

        # Add an ID column if it doesn't exist
        if 'ID' not in df.columns:
            df['ID'] = range(1, len(df) + 1)

        # Perform your data analysis here
        sns.countplot(df['quality'])
        # Ensure the static directory exists
        if not os.path.exists('static'):
            os.makedirs('static')
        plt.savefig('static/plot.png')
        plt.close()  # Close the plot to avoid memory issues
      
      # Feature Selection
        df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
        X = df.drop(['quality', 'goodquality'], axis=1)
        Y = df['goodquality']

        # Determine if the wine is good or not
        good_wine_count = df['goodquality'].sum()
        total_wine_count = df.shape[0]
        good_wine_percentage = (good_wine_count / total_wine_count) * 100
        wine_quality_message = f"{good_wine_percentage:.2f}% of the wines are of good quality."

        # Get IDs of good quality wines
        good_wine_ids = df[df['goodquality'] == 1]['ID'].tolist()
      
        # Generate PDF
        html = render_template('result.html', tables=[df.to_html(classes='data')], titles=df.columns.values,
                               model_res=model_res.to_html(classes='data'),
                               wine_quality_message=wine_quality_message,
                               good_wine_ids=good_wine_ids)
        options = {
            'disable-smart-shrinking': '',
            'no-stop-slow-scripts': '',
            'enable-local-file-access': ''
        }
        pdfkit.from_string(html, 'static/output.pdf', configuration=config, options=options)

        return render_template('result.html', tables=[df.to_html(classes='data')], titles=df.columns.values,
                               model_res=model_res.to_html(classes='data'),
                               wine_quality_message=wine_quality_message,
                               good_wine_ids=good_wine_ids)

@app.route('/download')
def download_file():
    return send_file('static/output.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)