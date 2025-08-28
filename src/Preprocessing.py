from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def preprocessor(num_col, cat_col):
    #use pipeline inside coloumntranfomer becouse coloumn tranfomer doesn't allow to combine operation it will combine columns 
    num_pipe = Pipeline([
        ("missing_value_num", SimpleImputer(strategy="median")),
        ("scaling",StandardScaler())
    ])

    cat_pipe =Pipeline([
        ("missing_value_cat",SimpleImputer(strategy="most_frequent")),
        ("encoding",OneHotEncoder(handle_unknown="ignore")) #if we are making models like logistic regression , linear regression we need to drop first so that to avoid multicollinearity
    ])

    preprocessor = ColumnTransformer([
        ("numerical", num_pipe, num_col),
        ("categorical", cat_pipe, cat_col)
    ])

    All_func= Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_classif, k=10))
    ])

    return All_func

