"""
Which Transfermarkt clubs are first teams: shared by the transfer-history build (which drops
moves to youth and reserve sides) and the Learn build (whose routes count senior moves only).
`club` is the API's club object (TransfermarktAPI.clubs).
"""

import re

AGE_GROUP_NAME = re.compile(r'\b(U-?\d{2}|Sub-?\d{2}|Youth|Yth\.?|Jugend)$')


def is_reserve_side(club):
    """Youth, reserve and feeder sides (Castilla, Barcelona B, U19s, FC Liefering).

    clubTypeId 1 is a first team; every type above 1 is a youth or reserve side. Type 0 is
    unclassified (small clubs): a reserve side when it has a parent club, otherwise None,
    and the name decides (YOUTH_SUFFIXES). "Retired", "Without Club" and the like are
    special clubs and stay.

    The API files some youth sides of small clubs as first teams ("Legia Warsaw Youth",
    "FC Volendam U17"), so a first team whose name ends in an age group is still dropped.
    """
    if club.get('isSpecialClub'):
        return False
    base = club['baseDetails']
    club_type = base.get('clubTypeId')
    if club_type == 1:
        return bool(AGE_GROUP_NAME.search(club['name'].strip()))
    if club_type == 0:
        return True if base.get('mainClubId') not in (None, '', '0', 0, club['id']) else None
    return True
