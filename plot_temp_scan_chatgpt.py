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

import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "ieee"])

ALL_COLORS = ["#00843D","#78BE20","#772583","#981D97","#FF6600","#FFA300","#00566e","#009CA6"]

temp_scan_gpt = pd.read_csv("results/temp_scan/gpt/summary-wo_grammar.csv")
temp_scan_llama = pd.read_csv("results/temp_scan/llama/summary-wo_grammar.csv")
mistral_best_data = pd.read_csv("results/temp_scan/lora-v11.0-1000-zero-shot/summary-wo_grammar.csv")

plt.errorbar(temp_scan_gpt.temperature, temp_scan_gpt["mean"], yerr=temp_scan_gpt["stddev"], label="GPT$_{4o}$ (Few shot)", color=ALL_COLORS[4], ls=":")
plt.errorbar(temp_scan_llama.temperature, temp_scan_llama["mean"], yerr=temp_scan_llama["stddev"], label="Llama$^{8B}_{v3.1}$ (Few shot)", color=ALL_COLORS[2], ls="--")
plt.errorbar(mistral_best_data.temperature, mistral_best_data["mean"], yerr=mistral_best_data["stddev"], label="Mistral$^{7B}_{v0.3}$/LoRA (Zero shot)", color=ALL_COLORS[0], ls="-")
plt.ylim((-0.05, 1.05))
plt.xlim((-0.05, 1.05))
plt.xticks([],[],minor=True)
plt.xlabel("Decoding Temperature")
plt.ylabel("Accuracy")
#plt.title("Accuracy with Different Temperatures")
plt.legend(frameon=True)
plt.gcf().set_size_inches(2.5, 2.5)
plt.savefig("exp_temp_scan_gpt.pdf", dpi=300)