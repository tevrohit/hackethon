#!/usr/bin/env python3
"""
Test runner script for the hackathon backend tests.
"""

import subprocess
import sys
import os

def run_tests():
    """Run all tests with pytest."""
    print("🧪 Running hackathon backend tests...")
    print("=" * 50)
    
    # Ensure we're in the backend directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run pytest with coverage if available
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--disable-warnings"
        ], check=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with exit code {result.returncode}")
            
        return result.returncode
        
    except FileNotFoundError:
        print("❌ pytest not found. Please install test dependencies:")
        print("pip install -r requirements.txt")
        return 1

def run_specific_tests(test_pattern):
    """Run specific tests matching a pattern."""
    print(f"🧪 Running tests matching pattern: {test_pattern}")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/",
            "-k", test_pattern,
            "-v",
            "--tb=short",
            "--disable-warnings"
        ], check=False)
        
        return result.returncode
        
    except FileNotFoundError:
        print("❌ pytest not found. Please install test dependencies:")
        print("pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific tests
        pattern = sys.argv[1]
        exit_code = run_specific_tests(pattern)
    else:
        # Run all tests
        exit_code = run_tests()
    
    sys.exit(exit_code)
