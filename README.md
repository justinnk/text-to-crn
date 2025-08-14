[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15198397.svg)](https://doi.org/10.5281/zenodo.15198397)
# Using (Not-so) Large Language Models to Generate Simulation Models in a Formal DSL: A Study on Reaction Networks

Code artifacts for the paper "Using (Not-so) Large Language Models to Generate Simulation Models in a Formal DSL: A Study on Reaction Networks" just accepted to TOMACS special issue for the PADS'25.

## Authors and Contacts

- Justin N. Kreikemeyer, University of Rostock (corresponding author, [contact](https://mosi.informatik.uni-rostock.de/team/staff/justin-kreikemeyer/))
- Miłosz Jankowski, University of Rostock
- Pia Wilsdorf, University of Rostock ([contact](https://mosi.informatik.uni-rostock.de/team/staff/pia-wilsdorf/))
- Adelinde M. Uhrmacher, University of Rostock ([contact](https://mosi.informatik.uni-rostock.de/team/staff/prof-dr-rer-nat-adelinde-m-uhrmacher/))

## :rocket: Quickstart

1. Clone the repository and move to the repository folder:
```shell
git clone https://github.com/justinnk/text-to-crn.git
cd text-to-crn
```
2. Run the script `./check_and_install.sh` to check requirements and install the dependencies. After running this script, if there were no errors, you may directly jump ahead to the reproduction.
4. Consult the section "Reproduce Results" below (important!)
3. Insert your Huggingface (see the first bullet in the section "Other Requirements") and OpenAI Keys at the top of `reproduce-all.sh` and the project id and organization id in `ChatGptAPI.py`. Run the script (`./reproduce-all.sh`) to reproduce all results. If you chose to manually run experiments and the results are now all present in the results folder, you may run `./reproduce-all.sh manual`. The figures and tables will then be placed in the folder `reproduction` and labelled like the figures and tables in the paper.

## :cd: Setup

### Required Software

- Linux operating system; tested on Ubuntu 22.04.4 LTS.
  - Should generally also work on Windows but adaptations may be required
- Python; tested on version 3.10.12 (Ubuntu)
- The dependencies listed in `requirements.txt`
  - can be installed automatically, see the next section
  - although all versions are fixed for best reproducibility, other versions are likely to work as well

### Required Hardware

- A CUDA-capable GPU with at least 24GB of VRAM and bf16 floating point compatibility
- Tested on RTX A5000 with 24GB VRAM with driver version `560.25.05` and CUDA version `12.6` and RTX3090Ti with driver version `570.133.07` and CUDA version `12.8`
  - Note: due to the use of float16, results for the tables 3-6 in the appendix may slightly differ when using a differnt GPU model. A patch (`precision_patch.txt`) is provided to mitigate this, but not guaranteed to work in every scenario (tested with the RTX 3090Ti).
- Around 600GB of free hard drive space to store the tested LLMs, trained LoRAs, and results

### Other Requirements

- A Huggingface Account
  - Access to the gated repositories of [Mistral 7B v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) and [Llama Instruct 8B v3.1](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
    - when logged in, this can be achieved by just clicking the buttons at the top of the respective page; *for the Llama model, you need to fill in the form and await approval (typically very quick).*
    - you then need to create an access token [here](https://huggingface.co/settings/tokens)
    - this needs to be provided to the reproduction script
- For reproduction only: Download our trained Mistral LoRA from the [related Zenodo archive](https://doi.org/10.5281/zenodo.15145040), unzip, and place it in the `loras` folder.
- API Key, project id, and organization id with funding for OpenAI API (provided by the authors)

### Installation Guide
(for Linux)

This is the manual installation guide. If you used the quickstart script without problems, skip this section.

1. Clone the repository and move to the repository folder:
```shell
git clone https://github.com/justinnk/text-to-crn.git
cd text-to-crn
```
2. Create a virtual environment and install the dependencies
```shell
# depending on your linux distribution, you may have
# to use either "python3" or "python" in this step.
# Afterwards, "python" suffices inside the venv.
python3 -m venv .venv                                    
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
You should now have a virtual environment stored in the folder `.venv` with all the dependencies installed.

Download our Mistral LoRA from [here](https://doi.org/10.5281/zenodo.15145040) and unzip into the `lora` folder, e.g.,
```shell
curl -LO https://zenodo.org/records/15145041/files/best_lora.zip
mv best_lora.zip loras/
cd loras/ && unzip best_lora.zip && rm best_lora.zip
```
 
## :file_folder: Overview of Contents

The following table provides an overview of the contents of the repository root.

| Folder/file                                               | Content/Purpose                                                                                                                                |
| ------:                                                   | :--------                                                                                                                                      |
| `data/`                                                   | Contains (automatically generated) training, testing and validation datasets.                                                                  | 
| `data/KinModGPT/`                                         | Json translation of the examples used in the KinModGPT paper[^1]                                                                               | 
| `data/V11.0-1000/`                                        | Json datasets used for training (`_train`), validation (`eval_`), and testing (`test_`); with (`w`) and without (`wo`) metadata (`meta`).      | 
| `data/V11.0-62-bio/`                                      | Testing dataset reduced to examples from the biology domain. Generated from the above using `gen_bio_only_test.py`.                            |
| `data/gen_bio_only_test.py`                               | Script to transform test data to a new version that only contains biological examples.                                                         |
| `data/print_human_readable.py`                            | Used to output to stdout a version of a dataset that is more redable by humans.                                                                |
| `data_generation/`                                        | Contains everything that is part of the synthetic data generation.                                                                             |
| `data_generation/ingredients/`                            | Text fragments used in the data generation.                                                                                                    |
| `data_generation/ingredients/connectors.txt`              | List of additive connectives.                                                                                                                  |
| `data_generation/ingredients/constructs.json`             | Sentence/CRN constructs supported by each domain.                                                                                              |
| `data_generation/ingredients/descriptions.csv`            | Template sentences that describe CRNs.                                                                                                         |
| `data_generation/ingredients/relational_sentences.csv`    | Relational sentence templates.                                                                                                                 |
| `data_generation/ingredients/species_names.csv`           | List of species names for each domain.                                                                                                         |
| `data_generation/ingredients/species_attributes.csv`      | List of species attributes for each domain.                                                                                                    |
| `data_generation/domain_specific_reaction_generator.py`   | Used to generate reactions and their metadata (to fill the templates) according to different constructs and domains.                           |
| `data_generation/manual_validation_samples.py`            | Some manually designed instruction-output pairs for validation.                                                                                |
| `data_generation/gen_descriptive_data.py`                 | Main data generation script.                                                                                                                   |
| `logs/`                                                   | Contains experiment stdout/stderr logs.                                                                                                        |
| `models/`                                                 | Contains LLMs automatically downloaded from [Huggingface](https://huggingface.co/) (not present by default, created by the scripts).           |
| `results/`                                                | Contains all the experimental results (once they are created).                                                                                 |
| `reactions.gbnf`                                          | GBNF grammar used for constrained decoding.                                                                                                    |
| `API.py`                                                  | Base class for LLM interfaces.                                                                                                                 |
| `TransformersAPI.py`                                      | Interface with LLMs using the [`transformers`](https://github.com/huggingface/transformers) API by Huggingface.                                |
| `ChatGptAPI.py`                                           | Interface with OpenAI's LLMs using their API.                                                                                                  |
| `KinModGptAPI.py`                                         | Implements the KinModGpt[^1] few-shot prompting method and our adaption.                                                                       |
| `IndraAPI.py`                                             | Interface with INDRA[^2] to translate models (not LLM-based).                                                                                  |
| `chat.py`                                                 | Use one of the API's in a chat style.                                                                                                          |
| `chat_user_study.py`                                      | Chat interface used by the participants of the user study.                                                                                     |
| `convergence_test.py`                                     | Implements stochastic convergence test according to[^3].                                                                                       |
| `tester.py`                                               | Used to evaluate approaches on their translation capability using the testing dataset and possibly a convergence test (for stochastic gen.).   |
| `logger.py`                                               | Helper function for better output logging.                                                                                                     |
| `exp_*.py`                                                | Experiment specifications for few-shot scan, lora hyperparameter scan, temperature scan, KinModGpt comparison, and INDRA comparison.           |
| `plot_*.py`                                               | Plot the results of the above experiments.                                                                                                     |
| `extract_table.py`                                        | Transform the output log of the last reproduction step into more usable LaTeX table column format.                                             |
| `precision_patch.txt`                                     | A patch that improves reproducibility when GPUs other than the RTX A5000 are used (due to different floating point implementations).           |
| `find_best_lora.py`                                       | Helper script to show the best LoRA configuration identified in the hyperparameter scan.                                                       |
| `.gitignore`                                              | Contains list of paths that should not be included in the Git version control.                                                                 |
| `requirements.txt`                                        | Contains Python dependencies required by the Python scripts (can be instlled with `pip install -r requirements.txt`).                          |
| `check-and-install.sh`                                    | Quick setup script to check requirements and install dependencies.                                                                             |
| `reproduce-all.sh`                                        | Quick script to run all experiments in sequence.                                                                                               |

[^1]: Maeda, Kazuhiro, and Hiroyuki Kurata. "Automatic generation of sbml kinetic models from natural language texts using gpt." International Journal of Molecular Sciences 24.8 (2023): 7296. https://doi.org/10.3390/ijms24087296
[^2]: https://github.com/sorgerlab/indra
[^3]: Kathryn Hoad, Stewart Robinson, and Ruth Davies. 2007. Automating des output analysis: How many replications to run. In 2007 Winter Simulation Conference. 505–512. https://doi.org/10.1109/WSC.2007.4419641


## :balance_scale: License

This project is licensed under the MIT License contained in `LICENSE`, unless indicated otherwise in a file.

## :bar_chart: Reproduce Results

These are the steps to reproduce the figures and claims in the paper.
If you are not using the automated reproduction script, the following steps assume that you are inside the virtual environment created before with the installation guide above (using `source .venv/bin/activate`, if not already done).
In any case, we highly recommend using `tmux` or something similar to start a persistent terminal session when working on a remote server.
This ensures that experiments are not interrupted.
However, almost all experiments are designed in a way that they are automatically continued (and not re-run) if reinvoked in case they were interrupted.
Unless you want to redo an experiment, you never have to delete a result from the `results` folder.
You can just run the respective command (or `./reproduce-all.sh`) again and it will continue where it was interrupted.

Running all experiments may take from 5 to 10 days of computation time, depending on how fast your GPU is or whether you use multiple ones to run different experiments in parallel.
When doing the latter, you can parallelize the two costliest steps 2 and 3 below (note: they typically depend on each other, but we also provide the LoRA trained in 2 to get you started with 3) and save up to 3 days in time.
Experiment 4 is also completely independent of the others and may be run in parallel.
Make sure you copy the results of all parallel sessions to the `results` folder of a single machine for plotting (e.g., using `rsync`).
Especially the hyperparameter scan (experiment 2) over LoRAs will take a lot of time (around 3 days on our RTX3090Ti) and also space (around 600GB).
The temperature scan (experiment 3) will take around 4 days.
The reproducibility of the experiments involving ChatGpt is only possible using the OpenAI API and purchasing Tokens there. The estimated cost for this is ~40€ due to 23541027 total tokens resulting from in 6262 API calls at the time of writing (if you are reproducing the results for the ACM reproducibility review, the authors will provide you with a funded API key).
Please be careful to not re-execute the experiment scripts of 1, 3, and 4 below, as running these results in costs.

Finally, there has been an issue with setting the seed during training (learning: the seed has to be set *before* loading a model for training, as an internal random source is already initialized at this time. It is not sufficient to set a training seed.). Thus, it is now impossible to reproduce the exact LoRA trained for the paper. For this reason, we provide our weights which are used for exact reproduction of the figures. They are either automatically downloaded and installed using the quickstart script or you can follow the instructions given in the `loras` folder.
The hyperparameter scan (experiment 2) still yields very similar results and a model of similar accuracy (which may then also be used to reproduce very similar results in all experiments based on this if desired). The seed for this is now also correctly set so that future reproductions yield the exact same table every time. Note that with the current seed, the highest model accuracy is a couple percent lower than the one we got in our experiments (81.5% with alpha 8 rank 16 dropout 0.5 vs. 84.5% with alpha 8 rank 8 dropout 0.3). This is however well within the range of values we observed and it should be easy (given sufficient time) to find a seed producing a more accurate model. It also supports our claim that the hyperparameters don't make a big difference, as the highest resulting LoRA with the current seed is trained with quite a different configuration than the one we identified.

To reproduce all results (with the above restrictions), execute the following steps. You can alternatively use the script `reproduce-all.sh` which will automatically run everything in sequence. Important: insert you HF_TOKEN and OPENAI_API_KEY at the top of the script and (even when not using the script) the project and organization id in `ChatGptAPI.py`! We also recommend checking in at least once a day to see if any errors occured. Start by creating the folders `logs`, `results`, and `loras` in the root.

0. [running time: ~1min] Generate the dataset with `python data_generation/gen_descriptive_data.py`. Generate the bio-only dataset with `cd data && python gen_bio_only_test.py` After this, if reproduction was successful, `git diff` should show no differences in the `data` folder, except for the timestamps in the datasets (this repository already comes with the data used in the paper, but this is a good way to check reproducibility of the data generation) and any further changes you made, such as adding your API keys. If you are using the automatic script, a file named `data/data_generation_diff.txt` (later also found in `reproduction/data_generation_diff.txt`) will be created to contain the diff.
1. [running time: ~12h] Execute the few-shot scan to reproduce Figure 2 and Figure 5 (top) with `(OPENAI_API_KEY=<key> HF_TOKEN=<token>; unbuffer time python exp_few_shot_scan.py 2>&1) | tee -a "logs/log-few-shot-scan-$(date -Iminutes)".txt`, `python plot_few_shot_scan.py`. The plot for Figure 2 is then stored as `exp_few_shot_scan.pdf` and for Figure 5 (top) as `exp_few_shot_scan_embedded.pdf`.
2. [running time: ~2d] Execute the LoRA hyperparameter scan `(HF_TOKEN=<token>; unbuffer time python exp_lora_scan.py 2>&1) | tee -a "logs/log-lora-scan-$(date -Iminutes)".txt`. Fint the best performing LoRA with `python find_best_lora.py`. The table can be found in `results/lora_scan/summary.csv`
3. [running time: ~4d] Execute the temperature scan `(OPENAI_API_KEY=<key> HF_TOKEN=<token>; unbuffer time python exp_temp_scan.py 2>&1) | tee -a "logs/log-temp-scan-$(date -Iminutes)".txt`. Plot results with `python plot_temp_scan.py`, `python plot_temp_scan_chatgpt.py`, and `python plot_comparison_table.py`.
4. To prepare the next step, if you are not using an RTX A5000 GPU, run `patch -p1 TransformersAPI.py < precision_patch.txt`. As we used the `torch.float16` format for efficiency when doing our experiments, the results on other GPU models may differ to such an extent that sometimes different tokens are selected in the LLM inference. The patch is an attempt to bring the values closer together, but may still fail sometimes.
5. [running time: ~1h] Conduct comparison to INDRA and KinModGpt using `(OPENAI_API_KEY=<key> HF_TOKEN=<token>; unbuffer time python exp_test_kinmodgpt.py 2>&1) | tee -a "logs/log-kinmodgpt-$(date -Iminutes)".txt` and `(unbuffer time python exp_compare_indra.py 2>&1) | tee -a "logs/log-indra-$(date -Iminutes)".txt`
6. Undo the patch with `patch -R -p1 TransformersAPI.py < precision_patch.txt`.
7. Run `./reproduce-all.sh manual` to generate the resulting figures and tables as in the paper. They will be placed in the `reproduction` folder.

The claims of the paper are mostly the empirical evaluation data visible in the plots. The plots from the paper should now be placed in the root/`reproduction` folder. They are (in order of appearance in the paper):

| Figure                | Path                                                                                                                                                                                                                            |
| ---:                  | :---                                                                                                                                                                                                                            |
| Figure 2              | `reproduction/figure2.pdf`                                                                                                                                                                                                      |
| Figure 3              | `reproduction/figure3.pdf`                                                                                                                                                                                                      |
| Figure 4              | `reproduction/figure4.pdf`                                                                                                                                                                                                      |
| Table 1               | `reproduction/table1.txt`                                                                                                                                                                                                       |
| Appendix C/Figure 5   | `reproduction/figure5_{top,bottom}.pdf`                                                                                                                                                                                         |
| Appendix A            | Fragments are taken from `data/V11.0-1000/train_wo_meta.json` (you can use text search to find them).                                                                                                                           |
| Appendix B/Table 2    | `reproduction/table2.csv`, `table2_best_lora.txt`                                                                                                                                                                               |
| Appendix F/Tables 3-6 | `reproduction/tables_3_4_5_6.txt` (for raw log) or `reproduction/tables_3_4_5_6.tex` (for LaTeX table column format). Also note that reactions may be reordered or renumbered wrt. the output you get (especially in Table 6).  |

Other claims from the paper text:

| Claim                                                                                                                                                                                           | Where to find the evidence                                                                                                                                                              |
| ---:                                                                                                                                                                                            | :---                                                                                                                                                                                    |
| "Embedding prompts resulted in similar, but slightly worse performance" (Section 5.5)"                                                                                                          | `reproduction/figure5_{top,bottom}.pdf` (specifically looking at temperature 0)                                                                                                         |
| "It is evident from the results that there is no systematic way in which the hyperparameters influenced the accuracy. All combinations result in an accuracy between 70% and 80%" (Section 5.6) | `reproduction/table2.csv`, `reproduction/table2_best_lora.txt`                                                                                                                          |
| KinModGpt accuracy on our data is 95.2% (Section 5.9)                                                                                                                                           | `results/kinmodgpt_scan/our_examples/summary.json`; Note that the actual value obtained might be slightly different as the OpenAI API cannot be used in a fully deterministic fashion. |
| Results regarding INDRA (Section 5.9)                                                                                                                                                           | `results/indra/bio-examples-test`                                                                                                                                                       |
| "In only around 42% percent of cases did the [INDRA] tool yield a non-empty model." (Section 5.9)                                                                                               | `results/indra/bio-examples-test/seed-1234-temperature_0/metadata.json` (62 - (value of CRITICAL_FAIL))/62                                                                              |

Structure of the `results` folder, when all experimental data was reproduced:

| File                                           | Contents                                                                                                             |
| ---:                                           | :---                                                                                                                 |
| `few_shot_scan/test-few-shot-x`                | Results of few shot scan of Mistral with x examples.                                                                 |
| `few_shot_scan/test-llama-few-shot-x`          | Results of few shot scan of Llama with x examples.                                                                   |
| `few_shot_scan/test-gpt-few-shot-x`            | Results of few shot scan of ChatGpt with x examples.                                                                 |
| `few_shot_scan_embedded/`                      | Same as above, but with examples embedded in the message rather than chat history. Used for figure in Appendix C     |
| `lora_scan/test-lora-x`                        | Results of LoRA hyperparameter scan with hyperparameters x                                                           |
| `temp_scan/few-shot`                           | Purple lines in Figure 3.                                                                                            |
| `temp_scan/one-shot`                           | Green lines in Figure 3.                                                                                             |
| `temp_scan/lora-v11.0-1000-few-shot`           | Blue lines in Figure 3.                                                                                              |
| `temp_scan/lora-v11.0-1000-zero-shot`          | Orange lines in Figure 3.                                                                                            |
| `temp_scan/lora-v11.0-1000-few-shot-embedded`  | Orange lines in Figure 5 (bottom).                                                                                   |
| `indra/`                                       | Results of the INDRA comparison experiment.                                                                          |
| `kinmodgpt_scan/`                              | Results of the KinModGpt comparison experiment.                                                                      |

Generally, the individual results in the above folders have the following form:

| File                                        | Contents                                                                                   |
| ---:                                        | :---                                                                                       |
| `test-V11.0-1000-<name>-temperature_x`      | Root folder for a single experiment at temperature x                                       |
| `log.txt`                                   | Log of the experiment.                                                                     |
| `summary.json`                              | This is the most important file containing the final summary statistics of the experiment. |
| `seed-<seed>-temperature-<x>`               | Contains results of a single experiment replication.                                       |
| `seed-<seed>-temperature-<x>/metadata.json` | Further information about the experiment.                                                  |
| `seed-<seed>-temperature-<x>/summary.json`  | Summary statistics of this replication.                                                    |
| `seed-<seed>-temperature-<x>/results.csv`   | Detailed results of this replication.                                                      |

## :page_facing_up: Cite

```
Justin Noah Kreikemeyer, Miłosz Jankowski, Pia Wilsdorf, and Adelinde M. Uhrmacher. 2025.
Using (Not-so) Large Language Models to Generate Simulation Models in a Formal DSL: A Study on Reaction Networks.
ACM Trans. Model. Comput. Simul. Just Accepted (May 2025). https://doi.org/10.1145/3733719
```
