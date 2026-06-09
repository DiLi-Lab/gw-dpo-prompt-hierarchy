"""Smoke-test that the external-benchmark package skeleton imports cleanly."""


def test_external_package_imports() -> None:
    import src.evaluation.external  # noqa: F401
    import src.evaluation.external.xstest  # noqa: F401
    import src.evaluation.external.iheval  # noqa: F401
    import src.evaluation.external.iheval.adapters  # noqa: F401
