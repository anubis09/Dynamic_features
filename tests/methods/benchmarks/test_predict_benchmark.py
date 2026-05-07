import pytest


@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_bench_encode(benchmark, trained_tilings, test_states, batch_size):
    X = test_states[:batch_size]
    benchmark(trained_tilings.encode, X)


@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_bench_p_encode(benchmark, trained_tilings, test_states, batch_size):
    X = test_states[:batch_size]
    benchmark(trained_tilings._p_encode, X)
