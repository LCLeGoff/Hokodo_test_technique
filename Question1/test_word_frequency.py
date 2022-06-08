from WordFrequency import WordFrequency


class TestWordFrequency:
    """
    Tests for the code of class WordFrequency
    """

    @staticmethod
    def test_empty_sentence():
        word_freq = WordFrequency.get_word_frequency("")
        expected_word_freq = []
        assert word_freq == expected_word_freq

    @staticmethod
    def test_empty_word_sentence():
        word_freq = WordFrequency.get_word_frequency("     ")
        expected_word_freq = []
        assert word_freq == expected_word_freq

    @staticmethod
    def test_hello_world_sentence():
        sentence = "Hello hello world, people from another world! " \
                   "My name is %Op@. I am from another world too."
        word_freq = WordFrequency.get_word_frequency(sentence)
        expected_word_freq = [
            ('%Op@.', 1),
            ('Hello', 1),
            ('I', 1),
            ('My', 1),
            ('am', 1),
            ('another', 2),
            ('from', 2),
            ('hello', 1),
            ('is', 1),
            ('name', 1),
            ('people', 1),
            ('too.', 1),
            ('world', 1),
            ('world!', 1),
            ('world,', 1)]
        assert word_freq == expected_word_freq

    @staticmethod
    def test_if_question_example_works():
        sentence = 'How much wood would a woodchuck chuck ' \
                   'if a woodchuck could chuck wood? ' \
                   'A woodchuck would chuck as much wood as a woodchuck could chuck ' \
                   'if a woodchuck could chuck wood.'
        word_freq = WordFrequency.get_word_frequency(sentence)
        expected_word_freq = [
            ('A', 1),
            ('How', 1),
            ('a', 4),
            ('as', 2),
            ('chuck', 5),
            ('could', 3),
            ('if', 2),
            ('much', 2),
            ('wood', 2),
            ('wood.', 1),
            ('wood?', 1),
            ('woodchuck', 5),
            ('would', 2)]
        assert word_freq == expected_word_freq

