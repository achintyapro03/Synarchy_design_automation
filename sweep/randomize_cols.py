import pandas as pd
import numpy as np

# ================= CONFIGURATION =================
NUM_SAMPLES = int(input("Enter the number of samples to generate: "))

# ================= PARAMETER SPACE =================
param_space = {
    "num_cores": [1, 2, 4, 8, 16, 32],
    "cpu_clock_GHz": np.round(np.linspace(1.0, 4.0, 16), 2),  # 1.0 GHz to 4.0 GHz
    "l1i_kb": [16, 32, 64, 128],
    "l1d_kb": [16, 32, 64, 128],
    "l1_assoc": [1, 2, 4, 8],
    "l2_kb": [128, 256, 512, 1024, 2048],
    "l2_assoc": [2, 4, 8, 16],
    "fetchWidth": [2, 4, 8, 12],
    "decodeWidth": [2, 4, 8, 12],
    "issueWidth": [2, 4, 8, 12],
    "commitWidth": [2, 4, 8, 12],
    "numROBEntries": [32, 64, 128, 192, 256],
    "numIQEntries": [16, 32, 64, 96, 128],
    "LQEntries": [8, 16, 32, 64],
    "SQEntries": [8, 16, 32, 64],
    "branch_predictor": [
        "BiModeBP",
        "LocalBP",
        "TAGE",
        "TAGE_SC_L_64KB",
        "MultiperspectivePerceptron64KB",
        "TournamentBP"
    ]
}

# param_space = {
#     # Keep all other parameters constant
#     "num_cores": [4],
#     "cpu_clock_GHz": [3.0],
#     "l1i_kb": [32],
#     "l1d_kb": [32],
#     "l1_assoc": [4],
#     "l2_kb": [512],
#     "l2_assoc": [8],
#     "fetchWidth": [4],
#     "decodeWidth": [4],
#     "issueWidth": [4],
#     "commitWidth": [4],
#     "numROBEntries": [128],
#     "numIQEntries": [64],
#     "LQEntries": [16],
#     "SQEntries": [16],

#     # Only vary branch predictor
#     "branch_predictor": [
#         "BiModeBP",
#         "LocalBP",
#         "TAGE",
#         "TAGE_SC_L_64KB",
#         "MultiperspectivePerceptron64KB",
#         "TournamentBP"
#     ]
# }

# ================= DATA GENERATION =================
data = []
for i in range(NUM_SAMPLES):
    sample = {"exp_id": i}
    for param, choices in param_space.items():
        if param == "cpu_clock_GHz":
            sample[param] = float(np.random.choice(choices))
        else:
            sample[param] = np.random.choice(choices)
    data.append(sample)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_file = "arch_sweep_dataset2.csv"
df.to_csv(output_file, index=False)

print(f"✅ Generated {NUM_SAMPLES} samples and saved to {output_file}")
print(df.head())
