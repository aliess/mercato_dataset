"""
Player names in plain Latin letters, the same in every game mode, so a name can be typed
on any keyboard and matches across modes: "Pascal Groß" → "Pascal Gross",
"Kenan Yıldız" → "Kenan Yildiz", "Martin Ødegaard" → "Martin Odegaard".
"""

from unidecode import unidecode


def latin_name(name):
    if not isinstance(name, str):
        return name
    return unidecode(name.replace('Æ', 'Ae')).strip()
