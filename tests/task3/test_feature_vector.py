from agent_code.q_learning_task_3_advanced_features.feature_vector import BaseFeatureVector


def test_feature_encoding_max():
    """Test feature encoding function."""
    unencoded_state = [4, 4, 4, 4, 15, 1]
    encoding = [5, 5, 5, 5, 16, 2]
    result = BaseFeatureVector.feature_encoding(unencoded_state, encoding)
    assert result == 20000-1


def test_feature_encoding_random():
    """Test feature encoding function."""
    unencoded_state = [1, 2, 3, 4, 12, 1]
    encoding = [5, 5, 5, 5, 16, 2]
    result = BaseFeatureVector.feature_encoding(unencoded_state, encoding)
    assert result == 18086


def test_feature_encoding_random_with_null():
    """Test feature encoding function."""
    unencoded_state = [1, 2, 3, 4, 0, 1]
    encoding = [5, 5, 5, 5, 16, 2]
    result = BaseFeatureVector.feature_encoding(unencoded_state, encoding)
    assert result == 10586


def test_feature_encoding_with_binary_encoding():
    """Test feature encoding function."""
    unencoded_state = [0, 1, 1, 0, 1, 1]
    encoding = [2, 2, 2, 2, 2, 2]
    result = BaseFeatureVector.feature_encoding(unencoded_state, encoding)
    assert result == 32+16+4+2
