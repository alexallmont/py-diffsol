def test_module_docstrings():
    import diffsol
    assert diffsol.__doc__ == "Top-level typed diffsol bindings"

    from diffsol import nalgebra_dense_lu_f64 as ds_nd
    assert ds_nd.__doc__ == "Wrapper for nalgebra_dense_lu_f64 diffsol type"

    from diffsol import faer_sparse_lu_f64 as ds_fs
    assert ds_fs.__doc__ == "Wrapper for faer_sparse_lu_f64 diffsol type"
