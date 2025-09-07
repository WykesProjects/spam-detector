# 📥 Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# 📂 Load dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

# Create column names: 57 features + 1 target label ('is_spam')
column_names = [f'feature_{i}' for i in range(57)] + ['is_spam']
df = pd.read_csv(url, header=None, names=column_names)

# Show first few rows of the dataset
print(df.head())

# 🎯 Split data into features (X) and labels (y)
X = df.drop('is_spam', axis=1)  # All columns except 'is_spam'
y = df['is_spam']               # Target column

# 🧪 Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# 🧠 Train the model using Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 🤖 Predict labels for test data
y_pred = model.predict(X_test)

# 📊 Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 🖨️ Print the evaluation metrics
print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1 Score:", round(f1, 3))

# 🔲 Create and display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Spam', 'Spam'],
            yticklabels=['Not Spam', 'Spam'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# 💾 Optional: Save the plot to a file for your GitHub repo
# plt.savefig("metrics.png")