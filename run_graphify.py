import os
import subprocess
import sys

def run_command(command):
    print(f"Executing: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

def main():
    print("--- Graphifyy Automation Script ---")
    
    # 1. Install/Initialize Graphify internals if needed
    # (The tool requires an initial 'graphify install' step)
    print("Initializing Graphify...")
    run_command("graphify install")
    
    # 2. Run the graph generation
    # We ignore the faiss_index.bin and other large binary files
    # The tool will scan code and documentation (kb.pdf)
    print("Generating Knowledge Graph (this may take a few minutes)...")
    run_command("graphify run")
    
    print("\n✅ Knowledge Graph generation complete!")
    print("Check the 'graphify-out/' directory for your interactive results.")

if __name__ == "__main__":
    main()
