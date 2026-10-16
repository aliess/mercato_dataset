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


# The name players know, where Transfermarkt's API uses another (Transfermarkt player id →
# name). Chosen by the user on 2026-10-16; the API's spelling is used for everyone else.
NAME_OVERRIDES = {
    743591: 'Savinho',                 # API: Sávio
    844797: 'Flaco López',             # API: José Manuel López
    74857: 'Marc-André ter Stegen',    # API: Marc ter Stegen
    323831: 'Rafael Borré',            # API: Rafael Santos Borré
}


def player_name(player_id, api_name):
    """A player's name in every game file: the override when there is one, else the API's,
    in plain Latin letters."""
    return latin_name(NAME_OVERRIDES.get(int(player_id), api_name))
