def assert_allclose(test, v, w):
    """Test that two vectors are approximately equal."""
    test.assertEqual(len(v), len(w))
    for i in range(len(v)):
        test.assertAlmostEqual(v[i], w[i])
