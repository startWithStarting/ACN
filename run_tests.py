import sys
import os
import unittest

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")

# Add the project root and src directory to Python path
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

if __name__ == "__main__":
    # Create a test loader and discover tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(project_root, "tests")
    suite = loader.discover(start_dir)

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful())