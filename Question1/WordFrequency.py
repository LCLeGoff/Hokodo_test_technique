from WordFrequencyHelper import WordFrequencyHelper as Wfh


class WordFrequency:

    """
    This class has only one main method get_word_frequency,
    which from a string returns an alphabetic ordered list of the words along with their frequencies.
    A word is simply a sequence of characters delimited by spaces.
    We do not take into account special characters and upper/lower case,
    which might duplicates the number of occurrences of a word.
    For instance, in the string "Hello hello world! I am from another world.",
    we distinguish "word." from "world!", and "Hello" from "hello".

    >>> sentence = 'Aa Aa. Ab  Aa aa'
    >>> WordFrequency.get_word_frequency(sentence)
    [('Aa', 2), ('Aa.', 1), ('Ab', 1), ('aa', 1)]
    """

    @classmethod
    def get_word_frequency(cls, sentence: str) -> list[(str, int)]:
        """
        Return from sentence, an alphabetic ordered list of the words along their frequency.
        A word is any character sequence delimited by spaces. We discard all empty words
        :param sentence: string to split in words
        :return alphabetic ordered list of the words and their frequencies
        """

        word_list = Wfh.split_string(sentence)
        word_frequency_dict = Wfh.get_element_frequency(word_list)
        sorted_word_frequency = Wfh.sort_dictionary(word_frequency_dict)

        return sorted_word_frequency
