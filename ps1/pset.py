"""
6.1000 Fall 2025
Problem Set 1

Fill out the following info:
Name: Samhtia Pokkunuri
Kerberos: 921579743
Approximate time spent (HH:MM): 3 Hours
"""

import string
# NO OTHER IMPORTS ALLOWED
# ONLY USE CONTENT COVERED IN LECTURES 1, 2, AND 3

############################################################
# supplied helper function -- DO NOT MODIFY
############################################################


global all_words
global all_contractions


def collect_file_entries(filename):
    with open(filename) as file:
        return [line.strip() for line in file]


def initialize_words():
    global all_words
    global all_contractions
    all_words = set(collect_file_entries("words.txt"))
    all_contractions = set(collect_file_entries("contractions.txt"))


def is_possessive_word(text):
    global all_words
    return (
        text[-2:] == "'s" and text[:-2] in all_words
        or text[-1:] == "'" and text[:-1] in all_words
    )


def is_word(text):
    """
    Determine whether or not a given string is a word (as defined by its
    presence in words.txt or contractions.txt), accounting for possessive words.

    Parameters:
        text (str): The text to be tested. Has no punctuation or
            whitespaces and only consists of lowercase characters.

    Returns:
        bool: True if text is a valid word, False otherwise.
    """
    if not hasattr(is_word, "executed"):
        initialize_words()
        is_word.executed = True
    global all_words
    global all_contractions
    return (
        text in all_words
        or text in all_contractions
        or is_possessive_word(text)
    )


############################################################
# encryption and decryption for Beaver ciphers
############################################################


def encrypt_char(char, alphabet, shift):
    """
    This code encrypts a single character based on a chosen character, alphabet, and shift.
    Parameters:
    char (str): the characrer being encrypted.
    alphabet (str): The ordering of possible characters within which to perform the Beaver cipher.
    shift(str): The number of positions you shift forward for the cipher.
    """
    index = alphabet.find(char) # finds the character in alphabet and its index
    new_index = (index + shift) % len(alphabet) #add index
    return alphabet[new_index]


def decrypt_char(char, alphabet, shift):
    """
    This code encrypts a single character based on a chosen character, alphabet, and shift.
    Parameters:
    char (str): the characrer being encrypted.
    alphabet (str): The ordering of possible characters within which to perform the Beaver cipher.
    shift(str): The number of positions you shift forward for the cipher.
    """
    index = alphabet.find(char)  # finds the character in alphabet and its index
    old_index = ((index - shift) % len(alphabet)) #opposite - subtract index
    return alphabet[old_index]
    raise NotImplementedError



#3.1.3
def find_char(char, alphabet, shift):
    index = alphabet.find(char)
    return index

def encrypt(plaintext, alphabet, initial_shift, magic_number):
    """
    Encrypt a plaintext using a specified Beaver cipher.

    Parameters:
        plaintext (str): The message to be encrypted.
        alphabet (str): The ordering of possible characters within which
            to perform the Beaver cipher.
        initial_shift (int): The number of positions in alphabet to
            shift forward the first character of plaintext.
        magic_number (int): The magic number to be used in the Beaver
            cipher. It should be positive and smaller than the length
            of alphabet.

    Returns:
        str: Encrypted ciphertext corresponding to plaintext.
    """
    adapted_shift = initial_shift
    new_string2 = ""
    for i in plaintext:
        if i in alphabet:
            new_character = encrypt_char(i,alphabet,adapted_shift) #encrypt character
            new_string2 += new_character
            if alphabet.index(new_character) % magic_number == 0: #check if magic num rule applies
                adapted_shift += 1
        else:
            new_string2 += i
    return new_string2
    raise NotImplementedError


def decrypt(ciphertext, alphabet, initial_shift, magic_number):
    """
    Decrypt a ciphertext according to a specified Beaver cipher.

    Parameters:
        ciphertext (str): The encrypted text to be decrypted.
        alphabet (str): The same as in encrypt().
        initial_shift (int): The same as in encrypt().
        magic_number (int): The same as in encrypt().

    Returns:
        str: Decrypted plaintext corresponding to ciphertext.
    """
    adapted_shift = initial_shift
    old_string = ""
    for i in ciphertext:
        if i in alphabet:
            old_character = decrypt_char(i,alphabet,adapted_shift) #decrypt character
            old_string += old_character
            if alphabet.index(i) % magic_number == 0: #check if magic num rule applies
                adapted_shift += 1
        else:
            old_string += i



    return old_string
    raise NotImplementedError


############################################################
# breaking the Beaver cipher
############################################################


whitespace = " \n"
links = "-/"
pauses = ",;:.?!"
double_quote = '"'
separators = whitespace + links + pauses + double_quote

def count_words(text):
    """
    Return the number of valid English words in a given string.
    Words are valid English when is_word() returns True.

    Possible words identified as the characters between
    valid separators. Separators comprised of characters
    defined above (whitespace, links, pauses, double_quote).

    Parameters:
        text (str): The given text.

    Returns:
        int: The number of valid English words.
    """
    total = 0
    currentword = ""
    for char in text:
        if char in separators: #if some kind of separator, count new word
            if is_word(currentword.lower()): #check if word measured
                total += 1
            currentword = ""
        else:
            currentword += char

    if is_word(currentword.lower()):
        total += 1 #if extra word at end

    return total

def break_cipher(ciphertext, alphabet):
    """
    Decode a message encrypted by a Beaver cipher into its likely
    original source, without prior knowledge of the key.

    Parameters:
        ciphertext (str): The encrypted text.
        alphabet (str): The alphabet on which the Beaver cipher
            encryption was performed.

    Returns:
        str: A plaintext that contains the most English words out of all
            possible plaintexts. If more than one has the same number of
            English words, return any of them.
    """
    bestOption = ""
    max_words = -1
    for initial_shift in range(len(alphabet)):
        for magic_number in range(1,len(alphabet)):
            adapted_shift = initial_shift
            candidate = ""
            for char in ciphertext:
                if char in alphabet:
                    # decrypt using prev function logic
                    index = alphabet.index(char)
                    old_index = (index - adapted_shift) % len(alphabet) #calculate old index
                    decrypted_char = alphabet[old_index] #find old character
                    candidate += decrypted_char #add old character

                    # adjust shift if index % magic_number == 0
                    if index % magic_number == 0:
                        adapted_shift += 1
                else: #if NOT in alphabet, just add character as is
                    candidate += char

            if count_words(candidate) > max_words: #check if word count is larger
                max_words = count_words(candidate)
                bestOption = candidate

    return bestOption

    raise NotImplementedError



############################################################
# manual testing code
############################################################


def test_decrypt_char():
    alphabet = string.ascii_lowercase + string.ascii_uppercase

    result = decrypt_char("c", alphabet, 2)
    print(f"Expected char:  a")
    print(f"Decrypted char: {result}")
    print()

    result = decrypt_char("a", alphabet, 5)
    print(f"Expected char:  V")
    print(f"Decrypted char: {result}")
    print()


alphabet_1 = string.ascii_lowercase + string.ascii_uppercase
plaintext_1 = "easy"
ciphertext_1 = "gcuA"

alphabet_2 = alphabet_1 + string.punctuation
plaintext_2 = "This is a simple test."
ciphertext_2 = "Vjku ku c ukorng vguv:"

alphabet_3 = alphabet_2 + string.digits
plaintext_3 = "Without a doubt, 6.1000 is the best subject ever! And the staff just amazing!!"
ciphertext_3 = "Ykvjqxw d grxew/ 9;4455 nx ynk hkyA zBirmkB mDmz) Ivl Bpm ABioo sDBC jvjIrwp**"


def try_encrypt_decrypt(
    plaintext, ciphertext, alphabet, initial_shift, magic_number
):
    encrypted = encrypt(plaintext, alphabet, initial_shift, magic_number)
    decrypted = decrypt(encrypted, alphabet, initial_shift, magic_number)
    print(f"Original plaintext:  {plaintext}")
    print(f"Expected ciphertext: {ciphertext}")
    print(f"Encrypted text:      {encrypted}")
    print(f"Decrypted text:      {decrypted}")
    print()


def test_encrypt_decrypt():
    initial_shift = 2
    magic_number = 8
    print(f"{initial_shift = }, {magic_number = }")
    print()

    try_encrypt_decrypt(
        plaintext_1, ciphertext_1, alphabet_1, initial_shift, magic_number
    )
    try_encrypt_decrypt(
        plaintext_2, ciphertext_2, alphabet_2, initial_shift, magic_number
    )
    try_encrypt_decrypt(
        plaintext_3, ciphertext_3, alphabet_3, initial_shift, magic_number
    )


def test_count_words():
    text = "hello world"
    print(text)
    print(f"found {count_words(text)} words")
    print()

    text = 'Hello, "World\'s Fair"!'
    print(text)
    print(f"found {count_words(text)} words")
    print()


def try_break_cipher(ciphertext, alphabet, expected):
    decrypted = break_cipher(ciphertext, alphabet)
    print(f"Ciphertext:  {ciphertext}")
    print(f"Expected:    {expected}")
    print(f"Best guess:  {decrypted}")
    print()


def test_break_cipher():
    try_break_cipher(ciphertext_1, alphabet_1, plaintext_1)
    try_break_cipher(ciphertext_2, alphabet_2, plaintext_2)
    try_break_cipher(ciphertext_3, alphabet_3, plaintext_3)

# Part 2: Caesar Cipher and Beaver Cipher (Wasn't sure if I was supposed to code this part, but did it just in case)
# # Caesar Cipher
# letters = "abcdefghijklmnopqrstuvwxyz"
# caesar_cipher = input("Enter what you would like to caesar cipher: ")
# shift1 = int(input("Enter the shift you would like: "))
# new_string = ""
# for i in caesar_cipher:
#     for j in range(len(letters)):
#         if i == letters[j]:
#             new_string += letters[(j + shift1) % len(letters)]

# print(new_string)

# # Beaver Cipher
# letters = "abcdefg"
# beaver_cipher = input("Enter what you would like to beaver cipher: ")
# initial_shift = int(input("Enter the initial shift you would like: "))
# magic_number = int(input("Enter the magic number you would like: "))
# adapted_shift = initial_shift
# new_string2 = ""
# for i in beaver_cipher:
#     for j in range(len(letters)):
#         if i == letters[j]:
#             new_char = letters[(j + adapted_shift) % len(letters)]
#             new_string2 += new_char
#             if letters.index(new_char) % magic_number == 0:
#                 adapted_shift += 1

# print(new_string2)

if __name__ == "__main__":
    # Uncomment the function calls below to test manually.
    # Note these are not comprehensive tests.
    # Feel free to modify or extend them when debugging your code.
    # Run test.py to make sure your code passes all our test cases.

    #test_decrypt_char()
    #test_encrypt_decrypt()
    # test_count_words()
    # test_break_cipher()
    pass
