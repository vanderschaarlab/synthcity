import os

# 1) Put all the file paths here.
FILE_PATHS = [
    "src/synthcity/plugins/core/plugin.py",
    "src/synthcity/plugins/core/schema.py",
    "src/synthcity/plugins/core/serializable.py",
    "src/synthcity/plugins/core/constraints.py",
    "src/synthcity/plugins/core/distribution.py",
    "src/synthcity/plugins/core/dataset.py",
    "src/synthcity/plugins/core/models/syn_seq/syn_seq_encoder.py",
    "src/synthcity/plugins/core/models/syn_seq/syn_seq.py",
    "src/synthcity/plugins/core/models/syn_seq/syn_seq_constraints.py",
    "src/synthcity/plugins/core/dataloader.py",
    "src/synthcity/plugins/generic/plugin_syn_seq.py",
    "src/synthcity/utils/reproducibility.py",
    "src/synthcity/utils/samplers.py",
    "src/synthcity/utils/serialization.py",
    "src/synthcity/utils/dataframe.py",
    "src/synthcity/utils/compression.py"
]

# 2) Specify the output file name here.
OUTPUT_FILE = "prompt2.txt"

def write_file_contents(filepath, out):
    """
    Helper function that writes a file's path and contents to the output file.
    """
    out.write(filepath + "\n")  # Write the path
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            out.write(f.read())
    except Exception as e:
        out.write(f"** Could not read file: {e} **")
    out.write("\n\n")  # Blank line after each file

def collect_files_into_one(filepaths, output_file):
    """
    Given a list of file paths, writes their contents into 'output_file'.
    """
    with open(output_file, "w", encoding="utf-8") as out:
        for path in filepaths:
            # Make sure the path actually exists and is a file
            if os.path.isfile(path):
                write_file_contents(path, out)
            else:
                out.write(path + "\n")
                out.write("** Not a valid file path or file does not exist. **")
                out.write("\n\n")

def main():
    """
    This script collects code from the given hard-coded file paths and
    writes them into a single text file.
    """
    print(f"Collecting files into {OUTPUT_FILE}...")
    collect_files_into_one(FILE_PATHS, OUTPUT_FILE)
    print(f"Done! Created {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
