# Workaround to test files from other directory
import os.path, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))