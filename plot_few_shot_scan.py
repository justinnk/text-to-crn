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


example_scan_mistral = pd.read_csv(f"results/mistral_few_shot_scan.csv").sort_values(by="few_shot_examples")
example_scan_llama = pd.read_csv(f"results/llama_few_shot_scan.csv").sort_values(by="few_shot_examples")
example_scan_gpt = pd.read_csv("results/gpt_few_shot_scan.csv").sort_values(by="few_shot_examples")

plt.plot(example_scan_mistral.few_shot_examples, example_scan_mistral["mean"], label="Mistral$^{7B}_{v0.3}$", color=ALL_COLORS[0], ls="-")
plt.plot(example_scan_llama.few_shot_examples, example_scan_llama["mean"], label="Llama$^{8B}_{v3.1}$", color=ALL_COLORS[2], ls="--")
plt.plot(example_scan_gpt.few_shot_examples, example_scan_gpt["mean"], label="GPT$_{4o}$", color=ALL_COLORS[4], ls=":")
plt.xlabel("Number of Few-Shot Examples\n(prepended to conversation)")
plt.ylabel("Accuracy")
plt.xlim((-1,71))
plt.ylim((-0.01,1.1))
plt.xticks(ticks=[0, 1, 5, 10, 20, 30, 40, 50, 60, 70], labels=[0,"",5,10,20,30,40,50,60,70])
plt.xticks([], [], minor=True)
#plt.title("Accuracy with Different Number of Examples at Temperature 0")
plt.legend(frameon=True)
plt.gcf().set_size_inches(2.5, 2.5)
plt.savefig(f"exp_few_shot_scan.pdf", dpi=300)


plt.cla()
plt.clf()


example_scan_mistral = pd.read_csv(f"results/mistral-embedded_few_shot_scan.csv").sort_values(by="few_shot_examples")
example_scan_llama = pd.read_csv(f"results/llama-embedded_few_shot_scan.csv").sort_values(by="few_shot_examples")

plt.plot(example_scan_mistral.few_shot_examples, example_scan_mistral["mean"], label="Mistral$^{7B}_{v0.3}$ (emb.)", color=ALL_COLORS[0], ls="-")
plt.plot(example_scan_llama.few_shot_examples, example_scan_llama["mean"], label="Llama$^{8B}_{v3.1}$ (emb.)", color=ALL_COLORS[2], ls="--")
plt.xlabel("Number of Few-Shot Examples\n(embedded in first prompt)")
plt.ylabel("Accuracy")
plt.xlim((-1,71))
plt.ylim((-0.01,1.1))
plt.xticks(ticks=[0, 1, 5, 10, 20, 30, 40, 50, 60, 70], labels=[0,"",5,10,20,30,40,50,60,70])
plt.xticks([], [], minor=True)
#plt.title("Accuracy with Different Number of (embedded) Examples at Temperature 0")
plt.legend(frameon=True)
plt.gcf().set_size_inches(2, 2)
plt.savefig(f"exp_few_shot_scan_embedded.pdf", dpi=300)
