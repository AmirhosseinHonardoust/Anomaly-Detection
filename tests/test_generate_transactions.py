from data.generate_transactions import generate


def test_same_seed_is_fully_deterministic():
    # Regression test for a bug where one anomaly branch used the global,
    # unseeded np.random instead of the local seeded rng, breaking
    # reproducibility for a given --seed.
    df1 = generate(start="2023-01-01", end="2023-06-01", seed=7, n_customers=50)
    df2 = generate(start="2023-01-01", end="2023-06-01", seed=7, n_customers=50)
    pd_testing_equal = df1.equals(df2)
    assert pd_testing_equal


def test_different_seeds_produce_different_data():
    df1 = generate(start="2023-01-01", end="2023-03-01", seed=1, n_customers=20)
    df2 = generate(start="2023-01-01", end="2023-03-01", seed=2, n_customers=20)
    assert not df1.equals(df2)
