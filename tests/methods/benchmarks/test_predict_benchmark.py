import pytest


@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_bench_predict(benchmark, trained_tilings, test_states, batch_size):
    X = test_states[:batch_size]
    benchmark(trained_tilings.predict, X)


@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_bench_p_predict(benchmark, trained_tilings, test_states, batch_size):
    X = test_states[:batch_size]
    benchmark(trained_tilings._p_predict, X)
