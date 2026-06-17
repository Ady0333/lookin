"""Step 0 environment check for Lookin.

Imports the core dependencies, prints their versions, and confirms the
environment is wired up correctly. No face logic here yet.
"""

import cv2
import deepface


def main() -> None:
    print(f"deepface version: {deepface.__version__}")
    print(f"opencv version:   {cv2.__version__}")
    print("Environment OK")


if __name__ == "__main__":
    main()
