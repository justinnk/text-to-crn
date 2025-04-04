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


import glob
import json

max_accuracy = float("-inf")
max_conf = ""
for file in glob.glob("results/lora_scan/**/*/summary.json"):
    with open(file) as f:
        accuracy = json.load(f)["mean"]
        print(file, accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_conf = file

print("best\n", "-"*10)
print(max_conf, "(", max_accuracy, ")")