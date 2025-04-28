import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os
from joblib import dump

csv_path = os.path.join(os.path.dirname(__file__), 'stock_prices.csv')

# Read the file as raw lines
with open(csv_path, 'r') as f:
    lines = f.readlines()

# The second line contains the stock symbols
header_line = lines[1].strip().split(',')
# The third line is the actual header row for data (should start with 'Date')
data_lines = lines[3:]

# Build DataFrame with correct columns
df = pd.DataFrame([l.strip().split(',') for l in data_lines], columns=header_line)

print("First 5 rows of loaded DataFrame:")
print(df.head())
print(f"Columns: {list(df.columns)}")
symbols = [col for col in df.columns if col != 'Date' and str(col).strip() != '']
print(f"Symbols found: {symbols}")

# Convert all columns except 'Date' to numeric
for symbol in symbols:
    df[symbol] = pd.to_numeric(df[symbol], errors='coerce')

# Create models directory if it doesn't exist
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

results = {}

for symbol in symbols:
    print(f"Training and saving model for: {symbol}")
    data = df[['Date', symbol]].copy()
    data = data.rename(columns={symbol: 'Close'})
    data = data.dropna()
    print(f"Valid data rows for {symbol}: {len(data)}")
    if data.empty or data['Close'].isnull().all():
        print(f"No valid data for symbol: {symbol}, skipping.")
        continue
    data['Return'] = data['Close'].pct_change()
    data['Target'] = np.where(data['Return'].shift(-1) > 0.002, 1, np.where(data['Return'].shift(-1) < -0.002, -1, 0))
    data['MA5'] = data['Close'].rolling(5).mean()
    data['MA10'] = data['Close'].rolling(10).mean()
    np.random.seed(42)
    data['Sentiment'] = np.random.uniform(-1, 1, size=len(data))
    data = data.dropna()
    X = data[['Return', 'MA5', 'MA10', 'Sentiment']]
    y = data['Target']
    if X.empty or y.empty:
        print(f"No features/targets for symbol: {symbol}, skipping.")
        continue
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Not enough data to split for symbol: {symbol}, skipping.")
        continue
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    model_path = os.path.join(models_dir, f'{symbol}.pkl')
    print(f"Saving model to: {model_path}")
    dump(clf, model_path)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    results[symbol] = {'accuracy': acc, 'report': report, 'confusion_matrix': cm.tolist()}

# Show results for top 5 symbols
for symbol in list(results.keys())[:5]:
    print(f"\n=== {symbol} ===")
    print(f"Accuracy: {results[symbol]['accuracy']:.2f}")
    print("Classification Report:\n", results[symbol]['report'])
    print("Confusion Matrix:\n", results[symbol]['confusion_matrix'])

# Save all results to file
try:
    with open('ml_results.txt', 'w') as f:
        for symbol in results:
            f.write(f"\n=== {symbol} ===\n")
            f.write(f"Accuracy: {results[symbol]['accuracy']:.2f}\n")
            f.write("Classification Report:\n" + str(results[symbol]['report']) + "\n")
            f.write("Confusion Matrix:\n" + str(results[symbol]['confusion_matrix']) + "\n")
    print("Results written to ml_results.txt")
except Exception as e:
    print(f"Error writing results to ml_results.txt: {e}")
    print("Results summary:")
    for symbol in results:
        print(f"\n=== {symbol} ===")
        print(f"Accuracy: {results[symbol]['accuracy']:.2f}")
        print("Classification Report:\n", results[symbol]['report'])
        print("Confusion Matrix:\n", results[symbol]['confusion_matrix'])
