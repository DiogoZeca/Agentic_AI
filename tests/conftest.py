import os
import sys

# Add repo root so spike.* imports resolve in all test files.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add tests/ so `import helpers` works from any test file without package ceremony.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
