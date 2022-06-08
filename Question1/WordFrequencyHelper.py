class WordFrequencyHelper:

    """
    This class is a helper for WordFrequency
    """

    @staticmethod
    def split_string(string: str) -> list[str]:
        """
        Split a string into a list of words. Word are character sequence delimited by spaces. We remove empty words.
        :param string: string to split
        :return: list of words

        """
        split_string = string.split(' ')
        if '' in split_string:
            split_string = [e for e in split_string if e != '']

        return split_string

    @staticmethod
    def get_element_frequency(element_list: list[str]) -> dict[str, int]:
        """
        Give a dictionary with elements of element_list as keys and frequency of these elements as items
        :param element_list: list of elements
        :return: dictionary giving frequency for each element of element_list
        """
        element_set = set(element_list)
        element_dict = {element: 0 for element in element_set}

        for e in element_list:
            element_dict[e] += 1

        return element_dict

    @staticmethod
    def sort_dictionary(dictionary: dict[str, int]) -> list[(str, int)]:
        """
        Turn a dictionary into a list of tuple (key, item), where keys are alphabetically ordered
        :param dictionary: dictionary to ordered
        :return: ordered list of tuple (key, item)
        """
        ordered_list = [(key, dictionary[key]) for key in dictionary.keys()]
        ordered_list.sort()

        return ordered_list
