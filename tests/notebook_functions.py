"""Load the functions an EXAMPLE notebook defines, for its tests.

A notebook keeps the functions only it uses in its own code cells, next
to the markdown that explains them. Each such cell starts with the line
"# Functions of this notebook" and holds only imports, constants and
definitions, so executing it runs no calculation. notebook_functions
executes exactly those cells, in notebook order, and returns their names.
"""

import json
from pathlib import Path
import sys
import types


PROJECT = Path(__file__).resolve().parents[1]
MARKER = "# Functions of this notebook"


def notebook_functions(filename):
    """Execute a notebook's definition cells and return what they define.

    Arguments:
      filename = notebook path relative to the project, for example
        "EXAMPLE_FEEDBACK_SCENARIOS.ipynb".
    Returns:
      a module holding every name the definition cells define, registered
      in sys.modules (Cobaya looks up a theory class's module by name).
      PROJECT is set beforehand, as the notebook's setup cell sets it.
    Raises:
      ValueError if the notebook has no definition cell.
    """
    notebook = json.loads((PROJECT / filename).read_text(encoding="utf-8"))
    name = "notebook_" + Path(filename).stem
    module = types.ModuleType(name)
    module.PROJECT = PROJECT
    sys.modules[name] = module
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        if "code" != cell["cell_type"] or not source.startswith(MARKER):
            continue
        exec(compile(source, f"{filename}, cell {index}", "exec"), module.__dict__)
        count += 1
    if 0 == count:
        raise ValueError(f"{filename} has no cell starting with {MARKER!r}.")
    return module
