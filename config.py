import numpy as np
seed = 42
vocabulary =  "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
number_of_characters = len(vocabulary)
max_length = 150
identity_mat = np.identity(number_of_characters)