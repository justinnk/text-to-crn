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

def print_max_accuracy(searchpaths: list[str], name: str):
  max_accuracy = float("-inf")
  stddev = 0
  for file in searchpaths:
    with open(file) as f:
      data = json.load(f)
      accuracy = data["mean"]
      if accuracy > max_accuracy:
        max_accuracy = accuracy
        stddev = data["stddev"]
  print(name + ":", max_accuracy, "+-", stddev)

if __name__ == "__main__":
    print_max_accuracy([
      *glob.glob("results/temp_scan/lora-v11.0-1000-few-shot/**/summary.json"),
      *glob.glob("results/temp_scan/lora-v11.0-1000-zero-shot/**/summary.json")
    ], "Mistral-7B-v0.3/LoRA")

    print_max_accuracy([
      *glob.glob("results/temp_scan/few-shot/**/summary.json"),
      *glob.glob("results/temp_scan/one-shot/**/summary.json")
    ], "Mistral-7B-v0.3/fs")

    print_max_accuracy(glob.glob("results/temp_scan/gpt/**/summary.json"), "Gpt-4o/fs")

    print_max_accuracy(glob.glob("results/temp_scan/llama/**/summary.json"), "Llama-8B-v3.1/fs")