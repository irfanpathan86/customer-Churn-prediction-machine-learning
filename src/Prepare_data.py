import pandas as pd
from Preprocessing import preprocessor
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Data\custemort_data.csv')


X = data.drop(columns=["customerID", "Churn"], axis=1)
y = data["Churn"]

le = LabelEncoder()
y= le.fit_transform(y)
y = pd.Series(y, name="Churn")   #convert back to series for concat

num_col = X.select_dtypes(include=["int64","float64"]).columns
cat_col = X.select_dtypes(include=["object"]).columns


pipe = preprocessor(num_col, cat_col)


X_transformed = pipe.fit_transform(X, y)

#use hasattr becaouse after ohe X_transformed may be an sparse matrix so that has .toarray but normal numpy matrix don't have toarray 
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()#if we do sparse=false in ohe then you can directly write that X_df line don't need to write if and that 2 line 
X_df = pd.DataFrame(X_transformed)
final_df = pd.concat([X_df, y.reset_index(drop=True)], axis=1)


final_df.to_csv("Data/processed.csv", index=False)
print("Preprocessed Data saved to data/processed.csv")
