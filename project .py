
# data preprocessing

# importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import ttk

#importing the datasets  using pandas library
dataset = pd.read_csv('data1.csv')    
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , 10].values

# Taking care of missing data
"""
# Fitting Decision tree Classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
"""

root = tk.Tk()
top_margin = tk.Frame(root, height=15, background='grey')
left_margin = tk.Frame(root, width=15, background='grey')
sheet_area = tk.Frame(root, background='grey')
right_margin = tk.Frame(root, width=15, background='grey')
bottom_margin = tk.Frame(root, height=15, background='grey')
top_margin.pack(side=tk.TOP, expand=tk.YES, fill=tk.X, anchor=tk.N)
bottom_margin.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.X, anchor=tk.S)
left_margin.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)
right_margin.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)
f1 = tk.Frame(root)
f2 = tk.Frame(root)
f3 = tk.Frame(root)
f4 = tk.Frame(root)
f5 = tk.Frame(root)
f11 = tk.Frame(root)
f1.pack(padx=30, pady=10)
f2.pack(padx=30, pady=10)
f3.pack(padx=30, pady=10)
f4.pack(padx=30, pady=10)
f5.pack(padx=30, pady=10)
f11.pack(padx=30, pady=10)
l1 = ttk.Label(f1, text="        Clump thickness")
tb1 = tk.Entry(f1, text="name1")
l2 = ttk.Label(f1, text="uniformity of cell size")
tb2 = tk.Entry(f1, text="name2")
l3 = ttk.Label(f2, text="                   cell Shape")
tb3 = tk.Entry(f2, text="name3")
l4 = ttk.Label(f2, text="      Marginal adhesion")
tb4 = tk.Entry(f2, text="name4")
l5 = ttk.Label(f3, text="   Epithelial cell size")
tb5 = tk.Entry(f3, text="name5")
l6 = ttk.Label(f3, text="             Bare Nuclei")
tb6 = tk.Entry(f3, text="name6")
l7 = ttk.Label(f4, text="        Bland Chromatin")
tb7 = tk.Entry(f4, text="name7")
l8 = ttk.Label(f4, text="        Normal Nucleoli")
tb8 = tk.Entry(f4, text="name8")
l9 = ttk.Label(f5, text="             Mitoses")
tb9 = tk.Entry(f5, text="name9")
l10 = ttk.Label(f5, text="           Petient ID")
tb10 = tk.Entry(f5, text="name10")
l11 = ttk.Label(f11, text="Result")
tb11 = tk.Entry(f11, text="name11")
l1.pack(padx=00, pady=10, side="left")
tb1.pack(padx=60, pady=10,side="left") 
l2.pack(padx=00, pady=10,side="left")
tb2.pack(padx=60, pady=10,side="left")
l3.pack(padx=00, pady=10,side="left")
tb3.pack(padx=60, pady=10,side="left")
l4.pack(padx=00, pady=10,side="left")
tb4.pack(padx=60, pady=10,side="left")
l5.pack(padx=00, pady=10,side="left")
tb5.pack(padx=60, pady=10,side="left")
l6.pack(padx=0, pady=10,side="left")
tb6.pack(padx=60, pady=10,side="left")
l7.pack(padx=0, pady=10,side="left")
tb7.pack(padx=60, pady=10,side="left")
l8.pack(padx=0, pady=10,side="left")
tb8.pack(padx=60, pady=10,side="left")
l9.pack(padx=0, pady=10,side="left")
tb9.pack(padx=60, pady=10,side="left")
l10.pack(padx=0, pady=10,side="left")
tb10.pack(padx=60, pady=20,side="left")
l11.pack(padx=10, pady=10,side="right")
tb11.pack(padx=10, pady=10,side="right")            
def get_result():
    clump_thickness = int(tb1.get())
    uniformity_of_cell_size = int(tb2.get())
    cell_shape = int(tb3.get())
    marginal_adhesion = int(tb4.get())
    epithelial_cell_size = int(tb5.get())
    bare_nuclei = int(tb6.get())
    bland_chromatin = int(tb7.get())
    normal_nucleoli = int(tb8.get())
    mitoses = int(tb9.get())
    _id = int(tb10.get())
    new_data = [_id,clump_thickness,uniformity_of_cell_size,cell_shape,marginal_adhesion,epithelial_cell_size,bare_nuclei,bland_chromatin,normal_nucleoli,mitoses]
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X[:, 6:7])
    X[:, 6:7] = imputer.transform(X[:, 6:7])
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    X_test = np.append(X_test,[new_data],axis=0)
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Fitting Random Forest Classifier to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    if y_pred[len(y_pred)-1] == 2:
        tb11.delete(0,'end')
        tb11.insert(0,"Negative")
    else:
        tb11.delete(0,'end')
        tb11.insert(0,"Positive")
    
b1 = tk.Button(f11, text="QUIT", fg="red",command=root.destroy)
b2 = tk.Button(f11, text="Get Result", fg="blue", command=lambda:get_result())
b1.pack(padx=0, pady=10,side="left")
b2.pack(padx=30, pady=10,side="right")
root.mainloop()
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)