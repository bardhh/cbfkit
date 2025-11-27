try:
    from cbfkit.controllers.cbf_clf.risk_aware_path_integral_cbf_clf_qp_control_laws import (
        risk_aware_path_integral_cbf_clf_qp_controller as cbf_controller,
    )
    print("Import successful!")
    print(cbf_controller)
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
