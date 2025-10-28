# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings

# --- Suppress Warnings for Cleaner Output ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# PART 0: CONFIGURATION (Please check this section)
# ============================================================================

# --- 1. Define your data file ---
DATA_FILENAME = '/kaggle/input/gem5-results/gem5_results.csv'  # <--- !! RENAME THIS TO YOUR CSV FILE !!

# --- 2. Define your Input Features (xi) ---
# (Based on your sample, 'exp_id' is excluded as it's an index)
INPUT_FEATURES = [
    'cpu_clock_GHz', 'l1i_kb', 'l1d_kb', 'l1_assoc', 'l2_kb',
    'l2_assoc', 'fetchWidth', 'decodeWidth', 'issueWidth', 'commitWidth',
    'numROBEntries', 'numIQEntries', 'LQEntries', 'SQEntries', 'branch_predictor'
]

# --- 3. Define your Output Metrics (y1) ---
OUTPUT_METRICS = [
    'Area', 'Peak Power', 'Total Leakage', 'Peak Dynamic', 
    'Subthreshold Leakage', 'Gate Leakage', 'Runtime Dynamic', 'ipc',
    'branch_misprediction_rate', 'icache_miss_rate', 
    'dcache_read_miss_rate', 'dcache_write_miss_rate'
]

# --- 4. Define your Categorical Input Features ---
CATEGORICAL_FEATURES = ['branch_predictor']


# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================
print("--- [Part 1] Loading and Preprocessing Data ---")

# Check if file exists
if not os.path.exists(DATA_FILENAME):
    print(f"Error: Could not find data file '{DATA_FILENAME}'.")
    print("Please make sure the file is in the same directory and the name is correct.")
    exit()

# Load the single CSV file
try:
    df = pd.read_csv(DATA_FILENAME)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Separate into Input (X) and Output (Y) DataFrames
try:
    X_df = df[INPUT_FEATURES]
    y_df = df[OUTPUT_METRICS]
except KeyError as e:
    print(f"\nError: A column name in your lists is not in the CSV file.")
    print(f"Column not found: {e}")
    print("Please check your 'INPUT_FEATURES' and 'OUTPUT_METRICS' lists.")
    exit()

print(f"Successfully loaded {len(df)} configurations from '{DATA_FILENAME}'.")

# Automatically determine numerical features
numerical_features = [col for col in INPUT_FEATURES if col not in CATEGORICAL_FEATURES]

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        # Scale all numerical features
        ('num', StandardScaler(), numerical_features),
        # One-hot encode all categorical features
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
    ],
    remainder='passthrough'
)


X_df.head()

y_df.head()



# ============================================================================
# PART 2: MODEL TRAINING AND TESTING (SURROGATE MODEL)
# ============================================================================
print("\n--- [Part 2] Model Training and Testing ---")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_df, test_size=0.2, random_state=42
)
print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples.")

# Fit the preprocessor on the TRAINING data and transform both sets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get processed feature and output counts
n_features = X_train_processed.shape[1]
n_outputs = y_df.shape[1]

print(f"Original input features: {len(INPUT_FEATURES)}")
print(f"Processed input features (after encoding): {n_features}")
print(f"Output metrics: {n_outputs}")

# Build the Neural Network
surrogate_model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    # Output layer: n_outputs neurons, 'linear' activation for regression
    Dense(n_outputs, activation='linear') 
])

surrogate_model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_percentage_error']
)

# Train the model
print("\n--- Training Surrogate Model ---")
history = surrogate_model.fit(
    X_train_processed,
    y_train,
    epochs=100, # You can increase this if loss is still improving
    batch_size=32,
    validation_split=0.2,
    verbose=0 # Set to 1 to see epoch-by-epoch progress
)
print("Model training complete. ✅")


# Test the model
print("\n--- Evaluating Model on Test Set ---")
loss, mape = surrogate_model.evaluate(X_test_processed, y_test, verbose=0)
print(f"Test Set Loss (MSE): {loss:.4f}")
print(f"Test Set Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Calculate R-squared score for a more interpretable metric
y_pred_test = surrogate_model.predict(X_test_processed, verbose=0)
r2 = r2_score(y_test, y_pred_test)
print(f"Test Set R-squared (R²): {r2:.4f} (Closer to 1.0 is better)")


print("\n--- ML-Powered Random Search (Find New Configs) ---")

# First, get the "search space" boundaries from the original data
search_space = {}
for col in numerical_features:
    search_space[col] = ('int' if pd.api.types.is_integer_dtype(X_df[col]) else 'float', 
                         X_df[col].min(), 
                         X_df[col].max())
for col in CATEGORICAL_FEATURES:
    search_space[col] = ('categorical', X_df[col].unique())

def generate_random_configs(n, space):
    """Generates n random configurations as a DataFrame."""
    configs = {}
    for col, (dtype, *bounds) in space.items():
        if dtype == 'int':
            configs[col] = np.random.randint(bounds[0], bounds[1] + 1, size=n)
        elif dtype == 'float':
            configs[col] = np.random.uniform(bounds[0], bounds[1], size=n)
        elif dtype == 'categorical':
            configs[col] = np.random.choice(bounds[0], size=n)
    return pd.DataFrame(configs)

def find_best_from_model(model, preproc, space, constraints, objective, n_iter=50000):
    print(f"Running ML-powered random search for {n_iter} iterations...")
    
    # Generate all random configs at once
    random_X_df = generate_random_configs(n_iter, space)
    
    # Preprocess all of them
    random_X_processed = preproc.transform(random_X_df)
    
    # Predict all of them
    y_pred_all = model.predict(random_X_processed, batch_size=1024, verbose=0)
    y_pred_df = pd.DataFrame(y_pred_all, columns=OUTPUT_METRICS)
    
    # Find feasible ones (similar to Method A)
    feasible_mask = pd.Series([True] * n_iter)
    for key, (op, value) in constraints.items():
        if op == '<=':
            feasible_mask &= (y_pred_df[key] <= value)
        elif op == '>=':
            feasible_mask &= (y_pred_df[key] >= value)
            
    feasible_X = random_X_df[feasible_mask]
    feasible_y_pred = y_pred_df[feasible_mask]
    
    print(f"Found {len(feasible_y_pred)} *predicted* feasible configurations.")
    
    if feasible_y_pred.empty:
        return None, None
        
    # Find the best one from the feasible set
    obj_col, obj_mode = objective
    if obj_mode == 'maximize':
        best_index = feasible_y_pred[obj_col].idxmax()
    else: # 'minimize'
        best_index = feasible_y_pred[obj_col].idxmin()
        
    return feasible_X.loc[best_index], feasible_y_pred.loc[best_index]



# ============================================================================
# PART 3: INFERENCE (SOLVING THE CONSTRAINED OPTIMIZATION)
# ============================================================================

CONSTRAINTS = {
    'Area': ('<=', 150),
    'Peak Power': ('<=', 80),
    'ipc': ('>=', 0.8),
    'branch_misprediction_rate': ('<=', 0.05)
}
# This is the objective to optimize (e.g., maximize 'ipc')
OBJECTIVE = ('ipc', 'maximize')

print("\n" + "="*50)
print(" [Part 3] Inference / Optimization")
print("="*50)
print(f"Goal: Find config that satisfies constraints and maximizes '{OBJECTIVE[0]}'.")
print("Constraints:")
for k, (op, v) in CONSTRAINTS.items():
    print(f"  - {k} {op} {v}")


# Run Method B
best_X_model, best_y_model = find_best_from_model(
    surrogate_model, preprocessor, search_space, CONSTRAINTS, OBJECTIVE
)

if best_X_model is not None:
    print("\n✨ Best *Predicted* Feasible Configuration (from ML model):")
    print("Configuration (xi):")
    print(best_X_model.to_string())
    print("\nPredicted Metrics (y1):")
    print(best_y_model.to_string())
    print("\n---")
    print("⚠️  NOTE: This is a *prediction*. You must run this configuration")
    print("in your simulator to get the true ground-truth metrics.")
else:
    print("\nML model could not find a feasible configuration in its search.")

print("\n--- End of Script ---")
