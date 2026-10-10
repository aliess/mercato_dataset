"""The difficulty tier of a player, from his highest market value.

One place for the dataset. The same lines are in the server (`difficultyForValue` in
footballquiz_firebase/functions/src/clueValidation.ts) and the app (Models/Difficulty.swift);
they are a contract (../CONTRACTS.md §5, "Tiers from market value"), so change all three together.
"""

BEGINNER_MIN_VALUE = 80_000_000      # beginner: €80m or more
INTERMEDIATE_MIN_VALUE = 30_000_000  # intermediate: €30m up to €80m; expert: under €30m
TIERS = ('beginner', 'intermediate', 'expert')


def difficulty_for_value(market_value):
    """'beginner', 'intermediate' or 'expert'. No value counts as 0, as on the server."""
    value = market_value or 0
    if value >= BEGINNER_MIN_VALUE:
        return 'beginner'
    return 'intermediate' if value >= INTERMEDIATE_MIN_VALUE else 'expert'
