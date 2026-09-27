import re
import pandas as pd

# Pre-compile regex for performance
# Removing punctuation and non-alphanumeric except spaces
PUNCTUATION_RE = re.compile(r'[^a-z0-9\s]')
# Multiple spaces
MULTIPLE_SPACE_RE = re.compile(r'\s+')
# Common business suffixes to remove at word boundaries
# Using word boundaries \b to ensure we only remove standalone tokens
SUFFIX_PATTERN = r'\b(inc|llc|ltd|corp|corporation|co|company|pvt|private|limited|lp|llp|pc|pllc)\b'
SUFFIX_RE = re.compile(SUFFIX_PATTERN)

def normalize_text(text_series: pd.Series) -> pd.Series:
    """
    Apply normalization to a pandas Series of strings:
    - Lowercase
    - Remove punctuation
    - Remove common business entity suffixes
    - Remove multiple spaces
    """
    # Fill NaN with empty string
    s = text_series.fillna("").str.lower()
    
    # Remove punctuation
    s = s.str.replace(PUNCTUATION_RE, ' ', regex=True)
    
    # Remove suffixes
    s = s.str.replace(SUFFIX_RE, ' ', regex=True)
    
    # Remove extra spaces
    s = s.str.replace(MULTIPLE_SPACE_RE, ' ', regex=True)
    
    # Trim leading/trailing spaces
    s = s.str.strip()
    
    return s

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes business_name, business_address and country in the given dataframe.
    """
    if "business_name" in df.columns:
        df["business_name_norm"] = normalize_text(df["business_name"])
    if "business_address" in df.columns:
        df["business_address_norm"] = normalize_text(df["business_address"])
    if "country" in df.columns:
        df["country_norm"] = normalize_text(df["country"])
        
    # Create a concatenated field for richer TF-IDF matching if needed
    if "business_name_norm" in df.columns and "business_address_norm" in df.columns:
        df["combined_text"] = df["business_name_norm"] + " " + df["business_address_norm"]
        # Add country too
        if "country_norm" in df.columns:
            df["combined_text"] = df["combined_text"] + " " + df["country_norm"]
            
    return df
