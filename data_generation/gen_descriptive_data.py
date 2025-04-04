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


"""
Classes to generate the 'descriptive' dataset.
Authors: Justin Kreikemeyer, Miłosz Jankowski
"""

from domain_specific_reaction_generator import DomainSpecificReactionGenerator
from manual_validation_samples import manual_validation_set

import json
import random
import os
import time
from itertools import product
from collections import defaultdict
from copy import deepcopy

import pandas as pd
import numpy as np
from tqdm import tqdm


num_to_str = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


class Ingredients:
    """
    Class to manage template sentences and species names for different domains.

    The database is stored in the "ingredients" folder in the files
    - `descriptions.csv`
      - text snippets to describe a myriad of models from different domains in an (almost) natural way
    - `relational_sentences.csv`
      - sentences used to relate to previous descriptions, e.g., to clarify the a rate
    - `species_names.csv`
      - species names to work with in the data
    - `species_attributes.csv`
      - possible attributes species can have in each domain
    - `connectors.txt`:
      - one connector, like "Additionally" per line

    Examples for kinds of reactions:
    - Production: -> A
    - Degradation: A ->
    - Complexation: A + B -> AB
    - Dissociation: AB -> A + B
    - Conversion: A + B -> C + D
    - Catalysis: A + E -> B + E
    """

    INGREDIENTS_DIR = "data_generation/ingredients/"

    def __init__(self, random_state: np.random.RandomState, train_test_split=0.8) -> None:
        self.random_state = random_state
        self.train_test_split = train_test_split
        descriptions = pd.read_csv(os.path.join(self.INGREDIENTS_DIR, "descriptions.csv"))
        species_names = pd.read_csv(os.path.join(self.INGREDIENTS_DIR, "species_names.csv"))
        relational_sentences = pd.read_csv(os.path.join(self.INGREDIENTS_DIR, "relational_sentences.csv"))
        self.species_attributes = pd.read_csv(os.path.join(self.INGREDIENTS_DIR, "species_attributes.csv"))
        self.connectors = self._get_connectors_from_file()
        self.constructs = self._get_constructs_from_file()
        self.domains = descriptions["domain"].drop_duplicates()
        # species (biology, ecology, epidemiology, letters/numbers)
        self.species_train, self.species_test = self._split_train_test(species_names)
        # descriptions (use only bio with letter domain)
        self.descriptions_train, self.descriptions_test = self._split_train_test(descriptions, groupby=["domain","rate","category","nreactants","nproducts"])
        self.relational_sentences_train, self.relational_sentences_test = self._split_train_test(relational_sentences, groupby=["domain","category","nreactants","nproducts"])
        #print(self.descriptions_train.to_string())
        #print()
        #print(self.descriptions_test.to_string())
        #print(self.relational_sentences_train.to_string())
        #print()
        #print(self.relational_sentences_test.to_string())
        #print(self.species_train.to_string())
        #print()
        #print(self.species_test.to_string())
        #exit()
    
    #@staticmethod
    #def _get_letter_species_names() -> list[str]:
    #    """Returns all combinations of single capital letters and numbers from 0-9 as a list."""
    #    return list(map(
    #        lambda x: "".join(x),
    #        product(
    #            string.ascii_uppercase,
    #            map(str, range(3))
    #        )
    #    ))

    def _split_train_test(
        self,
        data: pd.DataFrame,
        groupby=["domain"],
    ) -> tuple[pd.DataFrame,pd.DataFrame]:
        """
        Randomly split the data into training and test datasets according to the given fraction.
        The data is grouped by the given column names first and the split is done only within each group.
        This ensures, e.g., the inclusion of every domain in the training and test data.
        """
        data_train = data.groupby(groupby).sample(frac=self.train_test_split, random_state=self.random_state)
        data_test = data[~data.index.isin(data_train.index)]
        data_train = data_train.reset_index(drop=True)
        data_test = data_test.reset_index(drop=True)
        return data_train, data_test

    def _get_connectors_from_file(self):
        """Read in the list of sentence connectors."""
        with open(os.path.join(self.INGREDIENTS_DIR, "connectors.txt"), "r") as f:
            return f.read().splitlines()

    def _get_constructs_from_file(self):
        """Read in the list of sentence connectors."""
        with open(os.path.join(self.INGREDIENTS_DIR, "constructs.json"), "r") as f:
            return json.load(f)

    def get_reaction_types(self, domain: str):
        """Return all the possible reaction types for a given domain."""
        if domain in self.descriptions_train.domain.values:
            return self.descriptions_train[
                (self.descriptions_train.domain == domain) &
                (self.descriptions_train.category != "general")
            ].category.drop_duplicates().values
        print("[get_reaction_type] Domain", domain, "doesn't exist.")
    

class DatasetGenerator:
    """Builds the 'descriptive' training and test datasets from ingredients."""

    def __init__(self, dataset_description: str, dataset_version: str, num_samples: int, num_few_shot_samples: int, seed: int = 2137, train_test_split: float = 0.8, use_manual_eval: bool = False) -> None:
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.random_state = np.random.RandomState(self.seed)
        self.ingredients = Ingredients(self.random_state, train_test_split=train_test_split)
        self.reaction_gen = DomainSpecificReactionGenerator(self.rng)
        self.dataset_description = dataset_description
        self.dataset_version = dataset_version
        self.use_manual_eval = use_manual_eval
        self.train_set = []
        self.test_set = []
        self.few_shot_set = []
        # reactions
        self.generate(num_samples, num_few_shot_samples, train_test_split)

    def generate(self, num_samples: int, num_few_shot_samples: int, train_test_split: float = 0.8) -> None:
        """Generate training and test datasets."""
        training_samples = int(num_samples * train_test_split + 0.5)
        self.generate_trainset(training_samples)
        self.generate_testset(num_samples - training_samples)
        self.generate_few_shot_set(num_few_shot_samples)
        self._basic_duplicate_check()

    def generate_testset(self, num_samples: int) -> None:
        self.test_set = self._generate(num_samples, "test")

    def generate_trainset(self, num_samples: int) -> None:
        self.train_set = self._generate(num_samples, "train")

    def generate_few_shot_set(self, num_samples: int) -> None:
        # few shot uses the same building blocks as train (separate from test)
        self.few_shot_set = self._generate(num_samples, "train")

    def _generate(self, num_samples: int, train_or_test: str):
        data = {}
        data["metadata"] = {
            "description": self.dataset_description,
            "version": self.dataset_version,
            "time": time.ctime(),
            "seed": self.seed,
            "num_samples": num_samples,
        }
        data["samples"] = []
        species = self.ingredients.species_train if train_or_test == "train" else self.ingredients.species_test
        for _ in tqdm(range(num_samples)):
            data["samples"].append([self._get_sample(species, train_or_test)])
        return data
    
    def _get_sample(self, species: pd.DataFrame, train_or_test: str):
        sample = {
            "instruction": "",
            "reactions": [], # [{"left":[], "right":[], "rate":0.0}]
            "entities": [],
            "num_reactions": 0,
            "domain": "",
        }
        instructions = []
        output = []
        entities = []

        # make decision on domain
        domains = ["bio", "epi", "eco"]
        # uniform distribution over domains to make them equally "important"
        domain = self.rng.choice(domains)
        # select a random set of species involved in the model
        num_species = self.rng.randint(3, 5)
        relevant_species = species[species.domain == domain]
        species_names = list(relevant_species.sample(
            n=min(num_species, len(relevant_species)), # if there are less relevant than num_species
            random_state=self.random_state
        ).name.values)
        species_attributes = self.ingredients.species_attributes
        species_attributes = species_attributes[species_attributes.domain == domain].attribute.values

        # in the ecology domain, add attributes to the species
        if domain == "eco":
            for idx in range(len(species_names)):
                if self.rng.uniform(0, 100) < 60:
                    species_names[idx] += "_" + self.rng.choice(species_attributes)

        # load descriptions fitting the chosen domain
        descriptions = self.ingredients.descriptions_train if train_or_test == "train" else self.ingredients.descriptions_test
        descriptions = descriptions[descriptions.domain == domain]

        # load relational sentences fitting the chosen domain
        relationals = self.ingredients.relational_sentences_train if train_or_test == "train" else self.ingredients.relational_sentences_test
        relationals = relationals[relationals.domain == domain]

        curr_var_idx = 0 # for variables in rates

        # select a random number of "sentence constructs"
        # this is roughly the number of reactions, but some sentences may describe multiple reactions
        # hence, the actual number of reactions can be up to around 8
        num_constructs = self.rng.randint(2, 4)
        num_reactions = 0
        prev_construct = ""
        for construct_idx in range(num_constructs):
            # choose construct, apply an approximate duplicate elimination
            construct = prev_construct
            while construct.split("_")[0] == prev_construct.split("_")[0]:
              if num_species < 4:
                # conversion needs at least 4 species
                no_conversion = deepcopy(self.ingredients.constructs[domain])
                if "conversion" in no_conversion:
                   no_conversion.remove("conversion")
                construct = self.rng.choice(no_conversion)
              else:
                construct = self.rng.choice(self.ingredients.constructs[domain])
            # generate reaction(s) according to chosen construct, species names, and domain
            reac_data = self.reaction_gen.get_reactions(domain, construct, species_names)
            # insert fitting variable indices for rate and count number of reactions
            for reaction in reac_data["reactions"]:
                if reaction["rate"] == "k":
                    reaction["rate"] = "k" + str(curr_var_idx)
                    curr_var_idx += 1
                num_reactions += 1
            #print(reac_data)
            # assemble natural language description of generated reaction(s)
            # TODO: this should be a separate function really and could use some abstraction...
            description = "Not implemented"
            if construct in ["general", "degradation", "production", "conversion", "complexation", "predation", "catalysis", "chain"]:
                # select desired category, rate inclusion, number of reactions, etc.
                applicable_descriptions = descriptions[descriptions.category == construct]
                applicable_descriptions = applicable_descriptions[
                    applicable_descriptions.rate != reac_data["reactions"][0]["rate"].startswith("k")
                ]
                if construct == "chain":
                  applicable_descriptions = applicable_descriptions[applicable_descriptions.nreactions == "n"]
                else:
                  applicable_descriptions = applicable_descriptions[applicable_descriptions.nreactions == "1"]
                all_reactants = [r for reac in reac_data["reactions"] for r in reac["reactants"]]
                all_products = [p for prod in reac_data["reactions"] for p in prod["products"]]
                if construct != "chain":
                  applicable_descriptions = self._get_applicable_sentence_number(applicable_descriptions, len(all_reactants), "reactants")
                  applicable_descriptions = self._get_applicable_sentence_number(applicable_descriptions, len(all_products), "products")
                # randomly select one of the results
                selected_description = applicable_descriptions.sample(n=1, random_state=self.random_state)
                selected_description = selected_description.text.values[0]
                # fill in templates
                if construct == "chain":
                  reactants = self._join_species_list(self._transform_attributes(domain, reac_data["reactions"][0]["reactants"]))
                  products = self._join_species_list(self._transform_attributes(domain, reac_data["reactions"][-1]["products"]))
                else:
                  reactants = self._join_species_list(self._transform_attributes(domain, reac_data["reactions"][0]["reactants"]))
                  products = self._join_species_list(self._transform_attributes(domain, reac_data["reactions"][0]["products"]))
                intermediates = ""
                if len(reac_data["intermediates"]) != 0:
                    intermediates = self._join_species_list(self._transform_attributes(domain, reac_data["intermediates"]))
                description = selected_description.format(
                    reactants=reactants,
                    products=products,
                    rate=reac_data["reactions"][0]["rate"],
                    reactants1=self._transform_attributes(domain, [reac_data["reactants1"]])[0],
                    reactants2=self._transform_attributes(domain, [reac_data["reactants2"]])[0],
                    products1=self._transform_attributes(domain, [reac_data["products1"]])[0],
                    products2=self._transform_attributes(domain, [reac_data["products2"]])[0],
                    intermediates=intermediates
                )
            elif ("degradation" in construct or "production" in construct) and construct.endswith("multi"):
                # select desired category, rate inclusion, number of reactions, etc.
                applicable_descriptions = descriptions[descriptions.category == construct[:construct.find("_")]]
                applicable_descriptions = applicable_descriptions[
                    applicable_descriptions.rate != reac_data["reactions"][0]["rate"].startswith("k")
                ]
                applicable_descriptions = applicable_descriptions[applicable_descriptions.nreactions == "1"]
                all_reactants = [r for reac in reac_data["reactions"] for r in reac["reactants"]]
                all_products = [p for prod in reac_data["reactions"] for p in prod["products"]]
                applicable_descriptions = self._get_applicable_sentence_number(applicable_descriptions, len(all_reactants), "reactants")
                applicable_descriptions = self._get_applicable_sentence_number(applicable_descriptions, len(all_products), "products")
                # randomly select one of the results
                selected_description = applicable_descriptions.sample(n=1, random_state=self.random_state)
                selected_description = selected_description.text.values[0]
                # fill in templates
                reactants = self._join_species_list(self._transform_attributes(domain, all_reactants))
                products = self._join_species_list(self._transform_attributes(domain, all_products))
                description = selected_description.format(
                    reactants=reactants,
                    products=products,
                    rate=reaction["rate"]
                )
            elif ("degradation" in construct or "production" in construct) and construct.endswith("relational"):
                # choose sentence with rate = False for first part
                applicable_descriptions = descriptions[descriptions.category == construct[:construct.find("_")]]
                applicable_descriptions = applicable_descriptions[applicable_descriptions.rate == False]
                applicable_descriptions = applicable_descriptions[applicable_descriptions.nreactions == "1"]
                all_reactants = [r for reac in reac_data["reactions"] for r in reac["reactants"]]
                all_products = [p for prod in reac_data["reactions"] for p in prod["products"]]
                applicable_descriptions = self._get_applicable_sentence_number(applicable_descriptions, len(all_reactants), "reactants")
                applicable_descriptions = self._get_applicable_sentence_number(applicable_descriptions, len(all_products), "products")
                applicable_descriptions = applicable_descriptions[applicable_descriptions.rate == False]
                selected_description = applicable_descriptions.sample(n=1, random_state=self.random_state)
                selected_description = selected_description.text.values[0]
                # choose relational sentence with rate = True for second part
                applicable_relational = relationals[
                    (relationals.category == construct[:construct.find("_")]) |
                    (relationals.category == "general")
                ]
                all_reactants = [r for reac in reac_data["reactions"] for r in reac["reactants"]]
                all_products = [p for prod in reac_data["reactions"] for p in prod["products"]]
                applicable_relational = self._get_applicable_sentence_number(applicable_relational, len(all_reactants), "reactants")
                applicable_relational = self._get_applicable_sentence_number(applicable_relational, len(all_products), "products")
                selected_relational = applicable_relational.sample(n=1, random_state=self.random_state)
                selected_relational = selected_relational.text.values[0]
                # fill in templates
                reactants = self._join_species_list(self._transform_attributes(domain, reac_data["reactions"][0]["reactants"]))
                products = self._join_species_list(self._transform_attributes(domain, reac_data["reactions"][0]["products"]))
                description = selected_description.format(
                    reactants=reactants,
                    products=products,
                    rate="ERROR"
                ) + " " + selected_relational.format(
                    reactants=reactants,
                    products=products,
                    rate=reac_data["reactions"][0]["rate"]
                )
            elif construct == "mating":
                applicable_descriptions = descriptions[descriptions.category == construct]
                applicable_descriptions = applicable_descriptions[
                    applicable_descriptions.rate != reac_data["reactions"][0]["rate"].startswith("k")
                ]
                applicable_descriptions = applicable_descriptions[applicable_descriptions.nreactions == "1"]
                all_reactants = [r for reac in reac_data["reactions"] for r in reac["reactants"]]
                all_products = [p for prod in reac_data["reactions"] for p in prod["products"]]
                applicable_descriptions = self._get_applicable_sentence_number(applicable_descriptions, len(all_reactants), "reactants")
                applicable_descriptions = self._get_applicable_sentence_number(applicable_descriptions, len(all_products), "products")
                # randomly select one of the results
                selected_description = applicable_descriptions.sample(n=1, random_state=self.random_state)
                selected_description = selected_description.text.values[0]
                # fill in templates
                reactants = self._join_species_list(self._transform_attributes(domain, reac_data["reactions"][0]["reactants"]))
                products = self._join_species_list(self._transform_attributes(domain, [reac_data["offspring"]]))
                description = selected_description.format(
                    reactants=reactants,
                    products=products,
                    rate=reac_data["reactions"][0]["rate"]
                )
            else:
                raise Exception("The construct {construct} is not implemented yet in the sentence construction!")

            # if there is a lower case letter at the start of the sentence, make it uppercase
            # (this can be the case, e.g., if the first word is a number word, like "three"
            # introduced in the previous step)
            if description[0].isalpha() and description[0].islower():
              description = description[0].upper() + description[1:]

            # add connectors between sentences with 50% probability
            first_is_species_name = description.split(" ")[0] in reac_data["species_names"] or description.split(",")[0] in reac_data["species_names"]
            if construct_idx > 0 and construct_idx < num_constructs - 1 and self.rng.randint(0, 1) == 1:
                description = self.rng.choice(self.ingredients.connectors) + ", " + (description[0].lower() if not first_is_species_name else description[0])  + description[1:]
            elif construct_idx == construct_idx - 1 and self.rng.randint(0, 1) == 1:
                description = "Finally, " + (description[0].lower() if not first_is_species_name else description[0]) + description[1:]

            instructions.append(description)
            output.append(self._reac_data_to_str(reac_data))
            for reaction in reac_data["reactions"]:
              sample["reactions"].append({
                  "left": self._aggregate_species_list(reaction["reactants"], use_number_strings=False),
                  "right": self._aggregate_species_list(reaction["products"], use_number_strings=False),
                  "rate": reaction["rate"]
              })
              entities.extend(reaction["reactants"] + reaction["products"])
            prev_construct = construct
        intro = "The following describes a reaction system. Please translate to a formal description. "
        sample["instruction"] = intro + " ".join(instructions)
        sample["output"] = "```\n" + ''.join(output) + "```"
        sample["num_reactions"] = num_reactions
        sample["entities"] = sorted(list(set([x[1:] if x[0].isnumeric() else x for x in entities])))
        sample["domain"] = domain
        return sample
    
    @staticmethod
    def _get_applicable_sentence_number(descriptions: pd.DataFrame, number_of_species: int, reactions_or_products: str):
        """Returns a new data frame that contains only descriptions applicable to the number of reactants/products."""
        # cases are:                                                    encoded as
        # =0 (no species)                                                0
        # >= 1 (general) always applicable, except if no species        -1
        # =1 (exactly one) singular                                      1
        # >=2 (2 or more) plural                                         2
        reactions_or_products = "n" + reactions_or_products
        if number_of_species == 0:
            return descriptions[(descriptions[reactions_or_products] == 0)]
        elif number_of_species == 1:
            return descriptions[(descriptions[reactions_or_products] == 1) | (descriptions[reactions_or_products] == -1)]
        elif number_of_species >= 2:
            return descriptions[(descriptions[reactions_or_products] == 2) | (descriptions[reactions_or_products] == -1)]

    @staticmethod
    def _aggregate_species_list(species_list: list[str], use_number_strings: bool = True) -> list[str]:
        """
        Aggregates species that appear multiple times into a single with a number, e.g., "Fox, Fox" -> "2 Fox".
        if use_number_strings is True, a word (e.g., "one") is used instead of a number (e.g., "1").
        """
        # If a species appears more than once in the list, aggregate using two, three, ...
        actual_species_list = []
        # sorting is very important to retain order of species, e.g., for chain reactions
        for unique_species in sorted(set(species_list), key=species_list.index):
            count = len([s for s in species_list if s == unique_species])
            if count > 1:
                actual_species_list.append((num_to_str[count] + " " if use_number_strings else str(count)) + unique_species)
            else:
                actual_species_list.append(unique_species)
        return actual_species_list

    @staticmethod
    def _join_species_list(species_list: list[str]) -> str:
        """Joins a list of species into a comma-separated sentence builing block, introducing "and" where appropriate."""
        actual_species_list = DatasetGenerator._aggregate_species_list(species_list)
        # join the species list into a string
        joint = ", " if len(actual_species_list) > 2 else " and "
        species_list_str = joint.join(actual_species_list)
        if len(actual_species_list) > 2:
          last_comma = species_list_str.rfind(",")
          species_list_str = species_list_str[:last_comma+1] + " and" + species_list_str[last_comma+1:]
        return species_list_str

    def _transform_attributes(self, domain: str, species_names: list[str]) -> list[str]:
        """
        Transform a species including attributes into a natural sounding string.
        E.g., "Fox_healthy_female" -> "healthy female Fox"
        """
        if domain != "eco": return species_names
        def transform(species_name):
          if "_" not in species_name: return species_name
          parts = species_name.split("_")
          species = parts.pop(0)
          self.rng.shuffle(parts)
          parts.append(species)
          return " ".join(parts)
        return list(map(transform, species_names))

    @staticmethod
    def _reac_data_to_str(reac_data: dict):
        """Transforms reaction data into a formal model string following our grammar."""
        out = ""
        for reaction in reac_data["reactions"]:
            out += " + ".join(DatasetGenerator._aggregate_species_list(reaction["reactants"], use_number_strings=False))
            out += " -> "
            out += " + ".join(DatasetGenerator._aggregate_species_list(reaction["products"], use_number_strings=False))
            out += " @ "
            out += reaction["rate"]
            out += ";\n"
        return out

    def to_json(self, train_or_test: str, include_metadata: bool = True) -> str:
        """Return the desired dataset as json with or without metadata."""
        to_dump = self.test_set
        if train_or_test == "train":
            to_dump = deepcopy(self.train_set)
            for key in ["reactions", "entities", "num_reactions", "domain"]:
                for sample, _ in enumerate(to_dump["samples"]):
                  for subsample, _ in enumerate(to_dump["samples"][sample]):
                    del to_dump["samples"][sample][subsample][key]
        if include_metadata:
            return json.dumps(to_dump, indent=4)
        return json.dumps(to_dump["samples"], indent=4)
    
    def dump_all(self, eval_len: int = 100) -> tuple[str,str,str]:
        """Return every needed variant as json. (train, test, eval)"""
        trainset = deepcopy(self.train_set)
        for key in ["reactions", "entities", "num_reactions", "domain"]:
            for sample, _ in enumerate(trainset["samples"]):
                for subsample, _ in enumerate(trainset["samples"][sample]):
                  del trainset["samples"][sample][subsample][key]
        trainset = trainset["samples"]

        _test = self.test_set
        if not self.use_manual_eval:
          _train = trainset[eval_len:]
          _eval = trainset[:eval_len]
        else:
          _train = trainset
          _eval = manual_validation_set

        _examples = deepcopy(self.few_shot_set)
        for key in ["reactions", "entities", "num_reactions", "domain"]:
            for sample, _ in enumerate(_examples["samples"]):
                for subsample, _ in enumerate(_examples["samples"][sample]):
                  del _examples["samples"][sample][subsample][key]
        _examples = _examples["samples"]

        dump = lambda x: json.dumps(x, indent=4)
        return dump(_train), dump(_test), dump(_eval), dump(_examples)

    def _basic_duplicate_check(self) -> None:
        instructions_train = [subsample["instruction"] for sample in self.train_set["samples"] for subsample in sample]
        instructions_test = [subsample["instruction"] for sample in self.test_set["samples"] for subsample in sample]
        if not (len(instructions_train) == len(set(instructions_train)) and len(instructions_test) == len(set(instructions_test))):
            print("Train, duplicates:", len(instructions_train) - len(set(instructions_train)))
            print("Test, duplicates:", len(instructions_test) - len(set(instructions_test)))


if __name__ == "__main__":
        name = "V11.0-1000"
        dg = DatasetGenerator(
            f"Small dataset with 1000 samples to test few shot generation. Uses manually procured eval set.",
            name,
            1000,
            num_few_shot_samples=100,
            use_manual_eval=True,
            seed=424242
        )
        # print(dg.to_json("train", include_metadata=False))
        # print(dg.to_json("test", include_metadata=False))
        # print(dg.to_json("train", include_metadata=True))
        # print(dg.to_json("test", include_metadata=True))
        # exit()

        if not os.path.exists(f"data/{name}"):
          os.mkdir(f"data/{name}")

        train, test, _eval, examples = dg.dump_all()

        with open(f"data/{name}/test_wo_meta.json", "w") as f:
            f.write(test)

        with open(f"data/{name}/train_wo_meta.json", "w") as f:
            f.write(train)

        with open(f"data/{name}/eval_wo_meta.json", "w") as f:
            f.write(_eval)

        with open(f"data/{name}/examples_wo_meta.json", "w") as f:
            f.write(examples)

        with open(f"data/{name}/train_w_meta.json", "w") as f:
            f.write(dg.to_json("train", include_metadata=True))