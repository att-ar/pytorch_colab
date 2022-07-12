import os
import sys

import_path = os.path.abspath(os.path.join(os.getcwd(), ".."))

if import_path not in sys.path:
    sys.path.append(import_path)
