import random


def excludeCharacter(original, exclusive):
    exclusive = set(exclusive)
    filtered = ''.join(c for c in original if c not in exclusive)
    return filtered

def randomCharacterGenerator(charcterSet, size=1):
    randomChars = ''.join(random.sample(charcterSet, size))
    return randomChars
