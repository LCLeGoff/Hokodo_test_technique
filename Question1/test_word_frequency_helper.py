from WordFrequencyHelper import WordFrequencyHelper


class TestWordFrequencyHelper:
    """
    Tests for the code of class WordFrequencyHelper
    """
    # <editor-fold desc="split_string tests">
    @staticmethod
    def test_if_empty_str_returns_empty_list():
        word_list = WordFrequencyHelper.split_string("")
        expected_word_list = []
        assert word_list == expected_word_list

    @staticmethod
    def test_if_space_sequence_str_returns_empty_list():
        word_list = WordFrequencyHelper.split_string("     ")
        expected_word_list = []
        assert word_list == expected_word_list

    @staticmethod
    def test_if_str_of_same_word_returns_list_with_all_word_occurrence():
        word_list = WordFrequencyHelper.split_string("a a a b b b b")
        expected_word_list = ['a']*3+['b']*4
        assert word_list == expected_word_list
    # </editor-fold>

    # <editor-fold desc="get_element_frequency tests">
    @staticmethod
    def test_if_empty_list_returns_empty_dict():
        element_dict = WordFrequencyHelper.get_element_frequency([])
        expected_dict = {}
        assert element_dict == expected_dict

    @staticmethod
    def test_if_element_freq_is_correct():
        element_dict = WordFrequencyHelper.get_element_frequency(['a']*3+['b']*4+['rt']*10)
        expected_dict = {'a': 3, 'b': 4, 'rt': 10}
        assert element_dict == expected_dict
    # </editor-fold>

    # <editor-fold desc="get_element_frequency tests">
    @staticmethod
    def test_if_empty_dict_returns_empty_list():
        element_list = WordFrequencyHelper.sort_dictionary({})
        expected_element_list = []
        assert element_list == expected_element_list

    @staticmethod
    def test_alphabetic_ordering():
        element_dict = {'Aaa': 3, 'aBa': 3, 'aaA': 3,
                        'bb': 1, 'a%@': 10, '%aa': 15, '%ab': 5}
        element_list = WordFrequencyHelper.sort_dictionary(element_dict)
        expected_element_list = [
            ('%aa', 15), ('%ab', 5), ('Aaa', 3), ('a%@', 10), ('aBa', 3), ('aaA', 3), ('bb', 1)]
        assert element_list == expected_element_list
    # </editor-fold>
