"""
One spelling per country. The dataset tables use both "Turkey" and "Türkiye"; the app
knows the second (CountryFlags.swift), so every build writes that one.
"""

ALIASES = {'Turkey': 'Türkiye'}


def canonical(name):
    return ALIASES.get(name, name)
