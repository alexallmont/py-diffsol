def test_module_name():
    from diffsol import nalgebra_dense_lu_f64, faer_sparse_lu_f64, faer_sparse_klu_f64

    assert nalgebra_dense_lu_f64.name() == "nalgebra_dense_lu_f64"
    assert faer_sparse_lu_f64.name() == "faer_sparse_lu_f64"
    assert faer_sparse_klu_f64.name() == "faer_sparse_klu_f64"
