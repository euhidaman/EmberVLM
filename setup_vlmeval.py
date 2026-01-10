"""
Setup script for VLMEvalKit integration with EmberVLM.
Installs VLMEvalKit and downloads necessary benchmark data.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   EmberVLM + VLMEvalKit Setup                           ║
║   This will install VLMEvalKit for benchmarking         ║
╚══════════════════════════════════════════════════════════╝
""")

    # Check if VLMEvalKit directory exists
    vlmeval_path = Path("../VLMEvalKit")
    if not vlmeval_path.exists():
        print(f"⚠️  VLMEvalKit not found at {vlmeval_path.absolute()}")
        print("Please ensure VLMEvalKit is cloned at d:/BabyLM/VLMEvalKit")
        sys.exit(1)

    # Step 1: Install VLMEvalKit in development mode
    print("\n🔧 Step 1/3: Installing VLMEvalKit...")
    os.chdir(vlmeval_path)
    if not run_command("pip install -e .", "Installing VLMEvalKit"):
        print("❌ Failed to install VLMEvalKit")
        sys.exit(1)

    # Step 2: Install additional requirements
    print("\n🔧 Step 2/3: Installing additional dependencies...")
    additional_deps = [
        "openpyxl",  # For Excel file handling
        "apted",     # For tree edit distance
        "colormath", # For color processing
        "decord",    # For video processing
        "distance",  # For string distance metrics
    ]
    
    for dep in additional_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")

    # Step 3: Verify installation
    print("\n🔧 Step 3/3: Verifying installation...")
    try:
        import vlmeval
        print(f"✅ VLMEvalKit successfully installed!")
        print(f"   Version: {vlmeval.__version__ if hasattr(vlmeval, '__version__') else 'unknown'}")
    except ImportError:
        print("❌ VLMEvalKit import failed. Please check installation.")
        sys.exit(1)

    # Return to original directory
    os.chdir("../EmberVLM")

    print("""
╔══════════════════════════════════════════════════════════╗
║   ✅ Setup Complete!                                     ║
║                                                          ║
║   VLMEvalKit is now integrated with EmberVLM            ║
║                                                          ║
║   Benchmark data will be downloaded automatically       ║
║   when you run evaluation for the first time.           ║
║                                                          ║
║   To test benchmarking:                                 ║
║   python scripts/train_all.py --stage 2.5 \\             ║
║          --benchmark_subset quick                       ║
╚══════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()
