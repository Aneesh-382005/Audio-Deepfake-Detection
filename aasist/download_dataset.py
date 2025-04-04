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
    cmd = "curl -o ./LA.zip -# https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
    os.system(cmd)
    
    cmd = "unzip LA.zip"
    os.system(cmd)

    # Move files to the LA folder
    if os.path.isdir("LA/LA"):
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
        
        shutil.rmtree("LA/LA")

    os.remove("LA.zip")
