#!/usr/bin/env python3
"""
Demo script showing the autotuning system in action.
This script demonstrates the full autotuning capability with reasonable settings.
"""

import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("OFFLINE RL AUTOTUNING SYSTEM DEMONSTRATION")
    print("=" * 60)
    print()
    
    print("This demonstration will:")
    print("1. Run autotune with short episodes for quick demonstration")
    print("2. Show the complete autotuning process")
    print("3. Display all generated outputs")
    print()
    
    input("Press Enter to start the demonstration...")
    print()
    
    # Run autotuning with demonstration settings
    print("Starting autotuning with demonstration settings:")
    print("- Episodes per run: 25 (normally 250)")
    print("- Timeout: 5 minutes (normally 60)")
    print("- Will stop after a few successful runs or improvements")
    print()
    
    try:
        # Clean up any existing autotune directory
        if os.path.exists("autotune"):
            import shutil
            shutil.rmtree("autotune")
            
        # Run the autotuning
        result = subprocess.run([
            sys.executable, "autotune.py", 
            "--episodes", "25",
            "--timeout", "5"
        ], check=True, capture_output=False)
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE!")
        print("=" * 60)
        
        # Show generated files
        print("\nGenerated files:")
        if os.path.exists("autotune"):
            for file in sorted(os.listdir("autotune")):
                print(f"  autotune/{file}")
                
        print("\nTo run full 250-episode autotuning:")
        print("  python autotune.py")
        print("\nOr use the primary command interface:")
        print("  python train_offline.py --episodes 250")
        
    except subprocess.CalledProcessError as e:
        print(f"Demo failed with error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())