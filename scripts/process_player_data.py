#!/usr/bin/env python3
"""
Process football dataset CSV files to extract top players and enrich transfer data.

This script:
1. Sorts player market values in descending order
2. Extracts top 2500 players by market value (includes market_value column)
3. Filters transfer history for top 2500 players (removes youth team transfers)
4. Adds country information to transfer history
5. Finds and merges prestigious players (MAX value >= 10M, < 19M) who played for Barcelona, AC Milan, Real Madrid, Arsenal, or Chelsea
"""

import argparse
import pandas as pd
import os
import sys
from pathlib import Path
from unidecode import unidecode


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process football dataset CSV files to extract top players and enrich transfer data.'
    )
    parser.add_argument('player_market_value', help='Path to player_market_value.csv')
    parser.add_argument('player_profiles', help='Path to player_profiles.csv')
    parser.add_argument('team_details', help='Path to team_details.csv')
    parser.add_argument('transfer_history', help='Path to transfer_history.csv')
    parser.add_argument('output_dir', help='Directory where output files will be saved')
    
    return parser.parse_args()


def ensure_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {output_dir}")


def sort_market_values(market_value_file, output_dir):
    """
    Sort player market values in descending order.
    
    Args:
        market_value_file: Path to player_market_value.csv
        output_dir: Output directory path
        
    Returns:
        DataFrame with sorted market values
    """
    print("\n[Step 1/4] Sorting market values...")
    df = pd.read_csv(market_value_file)
    print(f"  - Loaded {len(df):,} market value records")
    
    # Sort by value in descending order
    df_sorted = df.sort_values('value', ascending=False)
    
    # Save to output directory
    output_file = os.path.join(output_dir, 'player_market_value_descending.csv')
    df_sorted.to_csv(output_file, index=False)
    print(f"  ✓ Saved sorted market values to {output_file}")
    
    return df_sorted


def normalize_player_name(name):
    """
    Normalize player name by replacing non-English special characters with ASCII equivalents.
    
    Args:
        name: Player name string (may contain special characters like é, ã, á, etc.)
        
    Returns:
        Normalized name with ASCII characters only
    """
    if pd.isna(name):
        return name
    
    return unidecode(str(name))


def is_european_country(place_of_birth):
    """
    Check if place_of_birth contains a European country name.
    
    Args:
        place_of_birth: String containing place of birth (may be city, country, or combination)
        
    Returns:
        True if European country is found, False otherwise
    """
    if pd.isna(place_of_birth):
        return False
    
    place_str = str(place_of_birth).lower()
    
    # List of European countries (case-insensitive matching)
    european_countries = [
        'albania', 'andorra', 'armenia', 'austria', 'azerbaijan', 'belarus', 'belgium',
        'bosnia', 'bulgaria', 'croatia', 'cyprus', 'czech', 'denmark', 'estonia',
        'finland', 'france', 'georgia', 'germany', 'greece', 'hungary', 'iceland',
        'ireland', 'italy', 'kazakhstan', 'kosovo', 'latvia', 'liechtenstein', 'lithuania',
        'luxembourg', 'malta', 'moldova', 'monaco', 'montenegro', 'netherlands', 'norway',
        'poland', 'portugal', 'romania', 'russia', 'san marino', 'serbia', 'slovakia',
        'slovenia', 'spain', 'sweden', 'switzerland', 'turkey', 'ukraine', 'united kingdom',
        'england', 'scotland', 'wales', 'northern ireland', 'vatican'
    ]
    
    return any(country in place_str for country in european_countries)


def get_prestigious_teams():
    """
    Return list of prestigious team names to check for.
    Only includes: Barcelona, AC Milan, Real Madrid, Arsenal, and Chelsea.
    Includes exact matches and common variations.
    """
    return [
        'Arsenal',  # Note: Will match "Arsenal" but need to exclude "Arsenal Sarandí", "Arsenal Kyiv", etc.
        'AC Milan',
        'Real Madrid',
        'FC Barcelona',
        'Barcelona',  # Note: Will match "Barcelona" but need to exclude "Barcelona SC", "Barcelona B", etc.
        'Chelsea',
        'Chelsea FC'
    ]


def normalize_team_name(team_name):
    """Normalize team name for comparison."""
    if pd.isna(team_name):
        return ''
    return str(team_name).strip()


def extract_top_players(market_value_df, player_profiles_file, output_dir):
    """
    Extract top 2500 players by market value and add their max market value.
    
    Args:
        market_value_df: Sorted DataFrame of market values
        player_profiles_file: Path to player_profiles.csv
        output_dir: Output directory path
        
    Returns:
        DataFrame with top 2500 players including their market value
    """
    print("\n[Step 2/4] Extracting top 2500 players...")
    
    # Load player profiles
    profiles_df = pd.read_csv(player_profiles_file)
    print(f"  - Loaded {len(profiles_df):,} player profiles")
    
    # Get top 30,000 rows from market value data to extract ~2500 unique players
    top_n = market_value_df.head(30000)
    
    # Calculate max market value for each player
    max_values = top_n.groupby('player_id')['value'].max().reset_index()
    max_values.columns = ['player_id', 'market_value']
    
    print(f"  - Extracted {len(max_values):,} unique top players from {30000:,} market value records")
    
    # Merge with player profiles
    top_players = profiles_df[profiles_df['player_id'].isin(max_values['player_id'])].copy()
    top_players = top_players.merge(max_values, on='player_id', how='left')
    
    # Normalize player names (replace non-English special characters)
    if 'player_name' in top_players.columns:
        top_players['player_name'] = top_players['player_name'].apply(normalize_player_name)
        print(f"  - Normalized player names (removed special characters)")
    
    # Sort by market value descending
    top_players = top_players.sort_values('market_value', ascending=False)
    
    initial_count = len(top_players)
    
    # Add European country check
    top_players['is_european'] = top_players['place_of_birth'].apply(is_european_country)
    
    # Filter logic:
    # Keep if: market_value > 20,000,000 OR Retired OR (European AND market_value <= 20,000,000)
    # Remove if: NOT European AND market_value <= 20,000,000 AND NOT Retired
    top_players = top_players[
        (top_players['market_value'] > 20000000) |
        (top_players['current_club_name'] == 'Retired') |
        (top_players['is_european'] == True)
    ]
    
    # Drop the temporary is_european column
    top_players = top_players.drop(columns=['is_european'])
    
    removed_count = initial_count - len(top_players)
    if removed_count > 0:
        print(f"  - Filtered out {removed_count:,} players:")
        print(f"    • market_value < 20,000,000 (excluding Retired)")
        print(f"    • non-European players with market_value <= 20,000,000")
    
    # Final filter: remove any player with market_value < 19,000,000 (including Retired)
    final_count = len(top_players)
    top_players = top_players[top_players['market_value'] >= 19000000]
    final_removed = final_count - len(top_players)
    
    if final_removed > 0:
        print(f"  - Final filter: removed {final_removed:,} players with market_value < 19,000,000 (including Retired)")
    
    # Save to file
    output_file = os.path.join(output_dir, 'player_profiles_top2500.csv')
    top_players.to_csv(output_file, index=False)
    
    print(f"  ✓ Extracted {len(top_players):,} players with market values")
    print(f"    Saved to player_profiles_top2500.csv")
    
    return top_players


def filter_transfer_history(transfer_history_file, top_players_df, output_dir):
    """
    Filter transfer history to only include top 2500 players.
    
    Args:
        transfer_history_file: Path to transfer_history.csv
        top_players_df: DataFrame of top 2500 players
        output_dir: Output directory path
        
    Returns:
        Filtered transfer history DataFrame
    """
    print("\n[Step 3/4] Filtering transfer history...")
    
    # Load transfer history
    transfers_df = pd.read_csv(transfer_history_file)
    print(f"  - Loaded {len(transfers_df):,} transfer records")
    
    # Get player IDs from top 2500
    top_player_ids = set(top_players_df['player_id'].unique())
    print(f"  - Filtering for {len(top_player_ids):,} unique top players")
    
    # Filter transfers
    filtered_transfers = transfers_df[transfers_df['player_id'].isin(top_player_ids)].copy()
    
    # Normalize player names (replace non-English special characters)
    if 'player_name' in filtered_transfers.columns:
        filtered_transfers['player_name'] = filtered_transfers['player_name'].apply(normalize_player_name)
        print(f"  - Normalized player names (removed special characters)")
    
    # Save to file
    output_file = os.path.join(output_dir, 'transfer_history_filtered.csv')
    filtered_transfers.to_csv(output_file, index=False)
    
    print(f"  ✓ Filtered to {len(filtered_transfers):,} transfer records")
    print(f"    Saved to transfer_history_filtered.csv")
    
    return filtered_transfers


def add_country_columns(filtered_transfers_df, team_details_file, output_dir):
    """
    Add from_team_country and to_team_country columns to transfer history.
    
    Args:
        filtered_transfers_df: Filtered transfer history DataFrame
        team_details_file: Path to team_details.csv
        output_dir: Output directory path
    """
    print("\n[Step 4/4] Adding country information...")
    
    initial_count = len(filtered_transfers_df)
    
    # Define youth team suffixes to filter out
    youth_suffixes = [
        'YTH', 'Youth', 'You', 'U19', 'U17', 'Yth', 
        'U20', 'U21', 'U18', 'U16', 'U23', 'U22', 'U24', 'II', 'Yth.', 'B'
    ]
    
    # Filter out rows where the destination (to_team_name) is a youth team,
    # regardless of whether the from_team_name is a youth team or not
    print(f"  - Filtering out transfers TO youth teams...")
    
    # Create a copy to avoid modifying the original
    filtered_transfers_df = filtered_transfers_df.copy()
    
    # Check if team name ends with any youth suffix
    def is_youth_team(team_name):
        """Check if team name ends with any youth suffix."""
        if pd.isna(team_name):
            return False
        team_name_str = str(team_name).strip()
        return any(team_name_str.endswith(suffix) for suffix in youth_suffixes)
    
    # Filter out rows where the destination team is a youth team
    mask = ~filtered_transfers_df['to_team_name'].apply(is_youth_team)
    filtered_transfers_df = filtered_transfers_df[mask]
    
    removed_count = initial_count - len(filtered_transfers_df)
    print(f"  - Removed {removed_count:,} youth team transfers")
    print(f"  - Remaining transfers: {len(filtered_transfers_df):,}")
    
    # Load team details
    teams_df = pd.read_csv(team_details_file)
    print(f"  - Loaded {len(teams_df):,} team records")
    
    # Create club_id to country_name mapping
    # Use drop_duplicates to get unique club_id mappings (in case of multiple seasons)
    team_country_map = teams_df[['club_id', 'country_name']].drop_duplicates('club_id')
    team_country_map = dict(zip(team_country_map['club_id'], team_country_map['country_name']))
    
    print(f"  - Created mapping for {len(team_country_map):,} unique clubs")
    
    # Add country columns - use empty string instead of 'Unknown' for missing values
    filtered_transfers_df['from_team_country'] = filtered_transfers_df['from_team_id'].map(team_country_map).fillna('')
    filtered_transfers_df['to_team_country'] = filtered_transfers_df['to_team_id'].map(team_country_map).fillna('')
    
    # Count how many were successfully mapped
    from_mapped = (filtered_transfers_df['from_team_country'] != '').sum()
    to_mapped = (filtered_transfers_df['to_team_country'] != '').sum()
    
    print(f"  - Mapped {from_mapped:,} from_team countries ({from_mapped/len(filtered_transfers_df)*100:.1f}%)")
    print(f"  - Mapped {to_mapped:,} to_team countries ({to_mapped/len(filtered_transfers_df)*100:.1f}%)")
    
    # Save the enriched file
    output_file = os.path.join(output_dir, 'transfer_history_filtered.csv')
    filtered_transfers_df.to_csv(output_file, index=False)
    
    print(f"  ✓ Added country columns and saved to transfer_history_filtered.csv")


def find_and_merge_prestigious_players(market_value_file, player_profiles_file, 
                                       transfer_history_file, main_players_file):
    """
    Find players with MAXIMUM market value >= 10M who have played for prestigious teams
    and merge them into the main players file.
    
    Args:
        market_value_file: Path to player_market_value.csv (historical values)
        player_profiles_file: Path to player_profiles.csv
        transfer_history_file: Path to transfer_history.csv
        main_players_file: Path to main players file (player_profiles_top2500.csv)
    """
    print("\n[Step 5/5] Finding and merging prestigious players...")
    print("  Checking for: Barcelona, AC Milan, Real Madrid, Arsenal, Chelsea")
    print("  Minimum MAX market value: 10,000,000")
    
    # Load and process market values - calculate MAX for each player
    print("  Loading market values and calculating MAX per player...")
    market_value_df = pd.read_csv(market_value_file)
    print(f"  - Loaded {len(market_value_df):,} market value records")
    
    # Calculate maximum market value for each player across all time
    max_values = market_value_df.groupby('player_id')['value'].max().reset_index()
    max_values.columns = ['player_id', 'max_market_value']
    print(f"  - Found {len(max_values):,} unique players")
    
    # Filter players with MAX market value >= 10,000,000 and < 19,000,000
    prestigious_value_players = max_values[
        (max_values['max_market_value'] >= 10000000) & 
        (max_values['max_market_value'] < 19000000)
    ].copy()
    print(f"  - Found {len(prestigious_value_players):,} players with MAX market value >= 10M and < 19M")
    
    # Load player profiles
    profiles_df = pd.read_csv(player_profiles_file, low_memory=False)
    
    # Load transfer history
    transfers_df = pd.read_csv(transfer_history_file)
    
    # Get prestigious team names
    prestigious_teams = get_prestigious_teams()
    
    # Filter transfers for prestigious value players
    prestigious_player_ids = set(prestigious_value_players['player_id'].unique())
    player_transfers = transfers_df[transfers_df['player_id'].isin(prestigious_player_ids)].copy()
    print(f"  - Found {len(player_transfers):,} transfer records for these players")
    
    # Normalize team names in transfer history
    player_transfers['from_team_normalized'] = player_transfers['from_team_name'].apply(normalize_team_name)
    player_transfers['to_team_normalized'] = player_transfers['to_team_name'].apply(normalize_team_name)
    
    # Check if any team matches prestigious teams
    def matches_prestigious_team(team_name):
        """Check if team name matches any prestigious team."""
        if not team_name:
            return False
        
        team_normalized = normalize_team_name(team_name)
        team_lower = team_normalized.lower()
        
        # Exact matches
        prestigious_lower = [t.lower() for t in prestigious_teams]
        if team_lower in prestigious_lower:
            return True
        
        # Special handling for teams that might have suffixes
        # Arsenal (but not Arsenal Sarandí, Arsenal Kyiv, Arsenal U18, etc.)
        if team_lower == 'arsenal' and not any(x in team_lower for x in ['sarandí', 'kyiv', 'u18', 'u19', 'u21', 'youth', 'b']):
            return True
        
        # Barcelona (but not Barcelona SC, Barcelona B, Barcelona C, etc.)
        if team_lower == 'barcelona' and not any(x in team_lower for x in [' sc', ' b', ' c', 'u18', 'u19', 'youth']):
            return True
        
        # AC Milan (but not AC Milan Youth, etc.)
        if team_lower == 'ac milan' and not any(x in team_lower for x in ['youth', 'u17', 'u19', 'b']):
            return True
        
        # Real Madrid (but not Real Madrid B, Real Madrid C, etc.)
        if team_lower == 'real madrid' and not any(x in team_lower for x in [' b', ' c', 'u19', 'youth']):
            return True
        
        # Chelsea / Chelsea FC
        if team_lower in ['chelsea', 'chelsea fc'] and not any(x in team_lower for x in ['u18', 'u19', 'u21', 'youth', 'b']):
            return True
        
        return False
    
    player_transfers['from_prestigious'] = player_transfers['from_team_normalized'].apply(matches_prestigious_team)
    player_transfers['to_prestigious'] = player_transfers['to_team_normalized'].apply(matches_prestigious_team)
    player_transfers['played_for_prestigious'] = player_transfers['from_prestigious'] | player_transfers['to_prestigious']
    
    # Get players who have played for prestigious teams
    players_with_prestigious = player_transfers[
        player_transfers['played_for_prestigious']
    ]['player_id'].unique()
    
    print(f"  - Found {len(players_with_prestigious):,} players who played for prestigious teams")
    
    # Get full player profiles for these players
    prestigious_players = profiles_df[profiles_df['player_id'].isin(players_with_prestigious)].copy()
    
    # Add max_market_value column
    prestigious_players = prestigious_players.merge(
        max_values[['player_id', 'max_market_value']], 
        on='player_id', 
        how='left'
    )
    
    # Rename max_market_value to market_value to match main file structure
    prestigious_players['market_value'] = prestigious_players['max_market_value']
    prestigious_players = prestigious_players.drop(columns=['max_market_value'])
    
    # Normalize player names
    if 'player_name' in prestigious_players.columns:
        prestigious_players['player_name'] = prestigious_players['player_name'].apply(normalize_player_name)
    
    print(f"  - Found {len(prestigious_players):,} prestigious players to merge")
    
    # Load main players file and merge
    if not os.path.exists(main_players_file):
        print(f"  - Main players file not found: {main_players_file}")
        print(f"  - Creating new file with prestigious players only")
        main_players = prestigious_players.copy()
    else:
        main_players = pd.read_csv(main_players_file, low_memory=False)
        print(f"  - Loaded {len(main_players):,} players from main file")
        
        # Get player IDs already in main file
        existing_player_ids = set(main_players['player_id'].unique())
        new_prestigious_ids = set(prestigious_players['player_id'].unique())
        
        # Find players to add (not already in main file)
        players_to_add = prestigious_players[
            ~prestigious_players['player_id'].isin(existing_player_ids)
        ].copy()
        
        print(f"  - Found {len(players_to_add):,} new prestigious players to add")
        print(f"  - {len(new_prestigious_ids) - len(players_to_add):,} prestigious players already in main file")
        
        # Merge: add new players
        if len(players_to_add) > 0:
            main_players = pd.concat([main_players, players_to_add], ignore_index=True)
            print(f"  - Added {len(players_to_add):,} new players")
    
    # Sort by market_value descending
    main_players = main_players.sort_values('market_value', ascending=False)
    
    # Save updated main players file
    main_players.to_csv(main_players_file, index=False)
    
    print(f"  ✓ Updated main players file: {main_players_file}")
    print(f"    Total players: {len(main_players):,}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Football Dataset Processing Script")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate input files exist
    for file_path in [args.player_market_value, args.player_profiles, 
                      args.team_details, args.transfer_history]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    print(f"\nInput files:")
    print(f"  - Market values: {args.player_market_value}")
    print(f"  - Player profiles: {args.player_profiles}")
    print(f"  - Team details: {args.team_details}")
    print(f"  - Transfer history: {args.transfer_history}")
    print(f"\nOutput directory: {args.output_dir}")
    
    # Ensure output directory exists
    ensure_output_directory(args.output_dir)
    
    # Step 1: Sort market values
    market_value_df = sort_market_values(args.player_market_value, args.output_dir)
    
    # Step 2: Extract top 2500 players
    top_players = extract_top_players(market_value_df, args.player_profiles, args.output_dir)
    
    # Remove the temporary sorted market value file
    temp_file = os.path.join(args.output_dir, 'player_market_value_descending.csv')
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"  ✓ Removed temporary file: player_market_value_descending.csv")
    
    # Step 3: Filter transfer history
    filtered_transfers = filter_transfer_history(
        args.transfer_history, 
        top_players, 
        args.output_dir
    )
    
    # Step 4: Add country columns
    add_country_columns(filtered_transfers, args.team_details, args.output_dir)
    
    # Step 5: Find and merge prestigious players
    main_players_file = os.path.join(args.output_dir, 'player_profiles_top2500.csv')
    find_and_merge_prestigious_players(
        args.player_market_value,
        args.player_profiles,
        args.transfer_history,
        main_players_file
    )
    
    print("\n" + "=" * 60)
    print("✓ Processing complete!")
    print("=" * 60)
    print(f"\nOutput files saved to: {args.output_dir}")
    print("  - player_profiles_top2500.csv (with market_value column, includes prestigious players)")
    print("  - transfer_history_filtered.csv (with country columns, youth teams filtered)")
    print()


if __name__ == '__main__':
    main()

