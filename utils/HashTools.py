"""
This file provides methods for creating hashes, in particular hashes that
are easy to memorize for humans.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import hashlib
import json
import os


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def string_to_md5(string):
    """
    Calculate the MD5 hash (message digest) of a given string.

    Args:
        string (string): An arbitrary string.

    Returns (string): The MD5 hash (message digest) of the provided string.
    """

    return hashlib.md5(string.encode('UTF-8')).hexdigest()


# -----------------------------------------------------------------------------


def file_to_md5(file_path):
    """
    Calculate the MD5 hash (message digest) of a file from a given file path.

    Args:
        file_path (string): The to the file to be hashed.

    Returns (string): The MD5 hash (message digest) of the specified file.
    """

    hasher = hashlib.md5()
    with open(file_path, 'rb') as file:
        buf = file.read()
        hasher.update(buf)

    return hasher.hexdigest()


# -----------------------------------------------------------------------------


def mnemonify_hash(hash_string):
    """
    Takes a hash and mnemonifies ("makes easier to remember") it by mapping
    it onto a string that follows the pattern:
        [adverb]-[adjective]-[animal name]
    e.g. for example:
        'bleakly-testy-bug' (from the hash of "Gravitational Wave")

    Args:
        hash_string (string): An MD5 hash, i.e. length is 32 characters.

    Returns (string): A version of the hash that is easier to remember for
        humans.
    """

    # Split the hash into 3 chunks of length 10 (and one of size 2)
    assert len(hash_string) == 32, 'Invalid hash size!'
    hash_list = list(map(''.join, zip(*[iter(hash_string)] * 10)))

    # Convert the hash parts from hexadecimal to decimal integers
    hash_list = list(map(lambda _: int(_, 16), hash_list))

    # Read in the words from JSON file
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'HashWords.json'), 'r') as file:
        words = json.load(file)

    adverb = words['adverbs'][hash_list[0] % len(words['adverbs'])]
    adjective = words['adjectives'][hash_list[0] % len(words['adjectives'])]
    animal = words['animals'][hash_list[0] % len(words['animals'])]

    return '-'.join([adverb, adjective, animal])
