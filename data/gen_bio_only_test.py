"""
Copyright 2025 Justin Kreikemeyer, Miłosz Jankowski

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:
 
  The above copyright notice and this permission notice shall be included in all copies or
  substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
  ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""

import os
import json
from copy import deepcopy

with open("V11.0-1000/test_wo_meta.json", "r") as file:
    data = json.load(file)

bio_only = deepcopy(data)
del bio_only["samples"]
bio_only["samples"] = []

for idx, entry in enumerate(data["samples"]):
    if entry[0]["domain"] == "bio":
        bio_only["samples"].append(entry)

num = len(bio_only["samples"])
print(num, "bio samples.")

if not os.path.exists(f"V11.0-{num}-bio"):
  os.mkdir(f"V11.0-{num}-bio")

with open(f"V11.0-{num}-bio/test_wo_meta.json", "w") as file:
    json.dump(bio_only, file, indent=2)
    
