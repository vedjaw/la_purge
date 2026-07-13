import sys
import torch

# Check if the user forgot to provide the file path
if len(sys.argv) < 2:
    print("Error: Please provide the path to the .pth file.")
    print("Usage: python script_name.py /path/to/file.pth")
    sys.exit(1)

# Get the path from the terminal argument
file_path = sys.argv[1]

print(f"Loading file: {file_path}\n" + "="*50)

# Load the file safely onto the CPU
data = torch.load(file_path, map_location=torch.device('cpu'))

print(f"Data Type: {type(data)}\n")

# Print keys and shapes safely (works for dicts and OrderedDicts)
if isinstance(data, dict):
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"Layer: {key} | Shape: {value.shape}")
        else:
            print(f"Key: {key} | Value Type: {type(value)}")
else:
    print("The file did not return a dictionary structure. Top-level content:")
    print(data)
