"""
AASIST
Copyright (c) 2021-present NAVER Corp.
Modified by Aneesh Grover in 2025
MIT license
"""

import os
import zipfile
import shutil

if __name__ == "__main__":
    # Download the zip file using curl
    cmd = "curl -o ./LA.zip -# https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
    os.system(cmd)
    
    # Unzip the file
    cmd = "unzip LA.zip"
    os.system(cmd)

    # Move files to the LA folder (if needed)
    if os.path.isdir("LA/LA"):
        # Create LA directory if it doesn't exist
        if not os.path.exists("LA"):
            os.makedirs("LA")
        
        # Move contents from LA/LA to LA
        for item in os.listdir("LA/LA"):
            s = os.path.join("LA/LA", item)
            d = os.path.join("LA", item)
            if os.path.isdir(s):
                shutil.move(s, d)
            else:
                shutil.move(s, d)
        
        # Remove the empty LA/LA folder after moving its contents
        shutil.rmtree("LA/LA")

    # Optionally, remove the zip file
    os.remove("LA.zip")
