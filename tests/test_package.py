def test_package_exposes_version() -> None:
    import bionic_head

    assert bionic_head.__version__ == "0.1.0"
