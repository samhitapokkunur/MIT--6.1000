# Caesar Cipher
letters = "abcdefghijklmnopqrstuvwxyz"
caesar_cipher = input("Enter what you would like to caesar cipher: ")
shift1 = int(input("Enter the shift you would like: "))
new_string = ""
for i in caesar_cipher:
    for j in range(len(letters)):
        if i == letters[j]:
            new_string += letters[(j + shift1) % len(letters)]

print(new_string)

# Beaver Cipher
letters = "abcdefg"
beaver_cipher = input("Enter what you would like to beaver cipher: ")
initial_shift = int(input("Enter the initial shift you would like: "))
magic_number = int(input("Enter the magic number you would like: "))
adapted_shift = initial_shift
new_string2 = ""
for i in beaver_cipher:
    for j in range(len(letters)):
        if i == letters[j]:
            new_char = letters[(j + adapted_shift) % len(letters)]
            new_string2 += new_char
            if letters.index(new_char) % magic_number == 0:
                adapted_shift += 1

print(new_string2)

# def encrypt_char(char, alphabet, shift):
#     """
#     This code encrypts a single character based on a chosen character, alphabet, and shift.
#     """
#     index = alphabet.find(char) # finds the character in alphabet and its index
#     new_index = (index + shift) % len(alphabet)
#     return alphabet[new_index]


# def decrypt_char(char, alphabet, shift):
#     """
#     This code encrypts a single character based on a chosen character, alphabet, and shift.
#     """
#     index = alphabet.find(char)
#     old_index = (index-shift % len(alphabet))
#     return alphabet[old_index]
#     raise NotImplementedError

# print(encrypt_char("m","abcdefghijklmnopqrstuvwxyz",2))
# print(decrypt_char("b","abcdefghijklmnopqrstuvwxyz",2))
