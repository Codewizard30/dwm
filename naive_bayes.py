import pandas as pd

# Load the dataset
df = pd.read_excel(r'C:\Users\shrad\OneDrive\Desktop\dwm final codes\Electronics.xlsx')

# Assume the last column is the target/class label
target_col = df.columns[-1]

# Drop R_Id column if present
if 'R_Id' in df.columns:
    df = df.drop(columns=['R_Id'])

# Get feature columns (exclude the target)
feature_cols = [col for col in df.columns if col != target_col]

# Step 1: Compute prior probabilities for each class
total_samples = len(df)
classes = df[target_col].unique()
priors = {
    c: len(df[df[target_col] == c]) / total_samples
    for c in classes
}

print("\nPrior Probabilities:")
for c in priors:
    print(f"P({target_col} = '{c}') = {round(priors[c], 3)}")

# Step 2: Define a function to compute likelihood
def calc_likelihood(df, feature, feature_value, target_class):
    subset = df[df[target_col] == target_class]
    feature_count = len(subset[subset[feature] == feature_value])
    total_count = len(subset)
    return feature_count / total_count if total_count > 0 else 0

# Step 3: Define the input sample for prediction (edit values as needed)

sample = {}
print("\nEnter values for the following features:")
for feature in feature_cols:
    options = df[feature].unique()
    print(f"Options for {feature}: {list(options)}")
    value = input(f"Enter value for {feature}: ")
    sample[feature] = value


# Step 4: Calculate likelihoods and posterior probabilities
likelihoods = {}
posteriors = {}

print("\nLikelihoods:")

for c in classes:
    prob = 1
    print(f"\nFor class = '{c}':")
    for feature in feature_cols:
        feature_val = sample.get(feature)
        likelihood = calc_likelihood(df, feature, feature_val, c)
        prob *= likelihood
        print(f"P({feature} = '{feature_val}' | {target_col} = '{c}') = {round(likelihood, 3)}")
    posteriors[c] = prob * priors[c]

# Step 5: Print posterior probabilities
print("\nPosterior Probabilities (after applying Bayes' Theorem):")
for c in posteriors:
    print(f"P(X | {target_col} = '{c}') * P({target_col} = '{c}') = {round(posteriors[c], 3)}")

# Step 6: Predict the class with the highest posterior
prediction = max(posteriors, key=posteriors.get)
print(f"\nPredicted class: {target_col} = '{prediction}'")

#sample input
# Youth , Medium , Yes , Fair
