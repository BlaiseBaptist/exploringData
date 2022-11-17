import warnings

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
def main():
    plt.close("all")
    penguin_data = pd.read_csv('./penguins.csv').dropna()
    filter = (penguin_data["species"] == "Gentoo")
    penguin_data = penguin_data[filter]
    y = "sex"
    X = "body_mass_g"
    encoder = OrdinalEncoder().fit_transform(
        penguin_data[y].to_numpy().reshape(-1,1))
    i=65
    X_train, X_test , y_train, y_test= train_test_split(penguin_data[X].to_numpy().reshape(-1,1),encoder,random_state=i)
    reg = linear_model.LogisticRegression()
    reg.fit(X_train, y_train)
    xs = np.linspace(penguin_data[X].min(),penguin_data[X].max()).reshape(-1,1)
    ys = reg.predict_proba(xs)[:,1]
    fig, ax = plt.subplots()
    ax.plot(xs,ys,color="orange",label="line 65")
    ax.scatter(X_train,y_train,label="points")
    ax.set_xlabel(X)
    ax.set_ylabel(y)
    ax.legend()
    plt.savefig("line_65.svg")
    plt.show()



if __name__ == '__main__':
    main()