# import libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# heading
st.write("""
         # Explore Different ML Models and Dataset
        let see which  is best model?""")

# make sidebar and add boxes then enter dataset name
dataset_name = st.sidebar.selectbox('select dataset', ('iris', 'brest cancer', 'wine'))

#add classifier
classifier_name = st.sidebar.selectbox('select classifier', ('KNN', 'SVM', 'Random Forest'))

# define fuction load to the three datasets
def get_dataset(dataset_name):
    data = None
    if dataset_name == 'iris':
        data = datasets.load_iris()
    elif dataset_name == 'brest cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    # split data
    x = data.data
    y = data.target
    return x, y
# call this function or equal to the x  y variables
X, y = get_dataset(dataset_name)

# print dataset shape in app
st.write('shape of dataset', X.shape)
# target value ki data type class hai tu uniqe number lay ghay or un ki value len hai
st.write('number of classes', len(np.unique(y)))

# define parameters of these three classifiers
def add_parameters_ui(classifier_name):
    params = dict() #create empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)#silder name c, ya params ki **key** hai 
        params['C'] = C #C classifier kay parameter ka name, its name degree of correct classification     
    elif  classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K # its the number of nearest neighbour
    else:
         
        max_depth = st.sidebar.slider('max_depth', 2, 15) 
        params['max_depth'] = max_depth #depth of every tree  that grow in random forest
        n_estimator = st.sidebar.slider('n_estimator', 1, 100) 
        params['n_estimator'] = n_estimator # number of tree
    return params         
# call function nad equal to the params variable
params = add_parameters_ui(classifier_name)        

#create classifier base on classifier_name and params
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C']) #SVC classifier create kay or uns main param C add kar dey or clf kay equal kar dey
    elif  classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        
        clf = RandomForestClassifier(n_estimators=params['n_estimator'],
                                     max_depth=params['max_depth'], random_state=1234) # random_state mean result bar bar reproduce ho thay rhay ghay                  
    return clf
clf = get_classifier(classifier_name, params)    

#split dataset by 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234 )
#train of classifier 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# check model accurancy,this acc print on app
from sklearn.metrics import accuracy_score
# pass both true labels and predicted labels
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred) 


#acc = accuracy_score((y_test, y_pred))# y_test or y_pred ki accuracy ko check kar ghay
st.write(f'classifier = {classifier_name}')
st.write(f'accuracy{score}')

#plot dataset
#plot all features on 2 dimensions use of PCA(feature ko reduce kar kay 2 dimension main plot kary ghay)
pca = PCA(2)
X_projection = pca.fit_transform(X)
#split data of the 0 or 1 dimension
x1 = X_projection[:, 0]
x2 = X_projection[:, 1]
fig = plt.figure()
plt.scatter(x1, x2,
            c=y, alpha=0.8, 
            cmap='viridis')#c=y color sakmi ko labels kya equall kar dey hai, alpha mean scatter plot main jo dots hai an ki transpernse kay hai 
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.colorbar()
#show plot
st.pyplot(fig)

# **assigments:
#add more models and parameters