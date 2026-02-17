import ast
from pathlib import Path


def test_legacy_rectifier_module_is_pure_reexport_wrapper():
    path = Path("src/cbfkit/controllers/cbf_clf/utils/rectify_relative_degree.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    import_from_nodes = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert len(import_from_nodes) == 1

    import_from = import_from_nodes[0]
    assert import_from.module == "cbfkit.certificates.rectifiers"
    imported_names = {alias.name for alias in import_from.names}
    assert imported_names == {
        "rectify_relative_degree",
        "compute_function_list",
        "polynomial_coefficients_from_roots",
    }
