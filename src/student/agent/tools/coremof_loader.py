import os
from collections import defaultdict
from typing import Dict, List

from student.agent.tools.tools import RaspaTool
from student.agent.utils import file, quick_search


class CoreMofLoader(RaspaTool):

    def __init__(self, path=None):
        name = "framework_loader"
        description = """Load the framework (MOF) file using coremof."""
        super().__init__(name, description, path)
        self.has_file = False
        self.structures: Dict[str, List[str]] = None

    def run(self, mof_name: str, output_file: str = "mol.cif"):
        import CoRE_MOF

        name = self.search_names(mof_name)
        if name is None:
            return self.get_output(e="No entry found in coremof names.")
        path = self.get_path(full=True)
        out_path = os.path.join(path, output_file)
        datasets = self.get_coremof_datasets(name)
        if datasets is None:
            return self.get_output(e=f"No dataset found for {name}")
        errors = []
        for dataset in datasets:
            try:
                mof = CoRE_MOF.get_structure(dataset, name)
                mof.to_file(out_path)
                self.has_file = True
                return self.get_output(
                    content=f"Generated from Coremof: {file(output_file)}"
                )
            except Exception as e:
                errors.append(e)
        return self.get_output(content=None, e=errors)

    def get_coremof_structures(self):
        import CoRE_MOF

        structures = defaultdict(list)
        datasets = {
            "2014": "2014",
            "2019-ASR": "2019-ASR",
            "2019-FSR": "2019-FSR",
        }  # CoRE_MOF.load.__datasets
        for dataset in datasets:
            for name in CoRE_MOF.list_structures(dataset):
                structures[name].append(dataset)
        return dict(structures)

    def get_structures(self):
        if self.structures is None:
            self.structures = self.get_coremof_structures()
        return self.structures

    def get_coremof_datasets(self, framework):
        return self.get_structures().get(framework, None)

    def structures_names(self):
        return self.get_structures().keys()

    def search_names(self, query, score_cutoff=90):
        candidates = self.structures_names()
        limit = 5

        matches = quick_search(
            query, candidates, limit=limit, score_cutoff=score_cutoff
        )

        if len(matches) == 0:
            return None

        best_match = matches[0]
        return best_match[0]
