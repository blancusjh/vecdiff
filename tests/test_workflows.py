"""Examples are tested scientific programs, not unexecuted illustrative snippets."""
from importlib import import_module
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


@pytest.mark.workflow
@pytest.mark.parametrize("name", ["field_propagation", "plane_interface", "cavity_resonance",
                                  "frustrated_tir", "curved_interface", "vector_focus"])
def test_complete_scientific_workflow(name):
    before = plt.get_fignums()
    module = import_module("examples."+name)
    assert plt.get_fignums() == before
    figure, report = module.run()
    assert figure.axes
    assert "parameters" in report and "assumptions" in report
    json.dumps(report, allow_nan=False)
    plt.close(figure)
