"""
Mapping from English transliteration to Ranjana script characters
Used for label conversion in test script
"""

TRANSLITERATION_TO_RANJANA = {
    # Vowels
    'a': 'अ',
    'aa': 'आ',
    'i': 'इ',
    'ii': 'ई',
    'ee': 'ई',
    'u': 'उ',
    'uu': 'ऊ',
    'ii': 'ई',
    'ee': 'ई',
    'e': 'ए',
    'ai': 'ऐ',
    'o': 'ओ',
    'au': 'औ',
    'am': 'अं',
    'aha': 'अः',
    
    # Consonants
    'ka': 'क',
    'kha': 'ख',
    'ga': 'ग',
    'gha': 'घ',
    'nga': 'ङ',
    'cha': 'छ',
    'chha': 'छ',
    'ca': 'च',  # Alternative spelling
    'ja': 'ज',
    'jha': 'झ',
    'nya': 'ञ',
    'ta': 'ट',
    'tha': 'ठ',
    'dda': 'ड',
    'ddha': 'ध',  # Note: 'ddha' maps to 'ध' (dha) based on dataset
    'nna': 'ण',
    'tta': 'त',
    'ttha': 'थ',
    'da': 'द',
    'dha': 'ध',
    'na': 'न',
    'pa': 'प',
    'pha': 'फ',
    'ba': 'ब',
    'bha': 'भ',
    'ma': 'म',
    'ya': 'य',
    'ra': 'र',
    'la': 'ल',
    'va': 'व',
    'sha': 'श',
    'ssa': 'ष',
    'sa': 'स',
    'ha': 'ह',
    'ksa': 'क्ष',
    'tra': 'त्र',
    'jna': 'ज्ञ',
    'gyan': 'ज्ञ',  # Alternative spelling
    'lu': 'लु',  # Special case
    'wo': 'वो',  # Special case
    
    # Digits
    'zero': '०',
    'one': '१',
    'two': '२',
    'three': '३',
    'four': '४',
    'five': '५',
    'six': '६',
    'seven': '७',
    'eight': '८',
    'nine': '९',
    
    # Special characters
    'om': 'ॐ',
    'danda': '।',
    'double_danda': '॥',
}

def transliterate_to_ranjana(text):
    """
    Convert English transliteration to Ranjana script
    Returns the Ranjana character if found, otherwise returns original text
    """
    text_lower = text.lower().strip()
    
    # Direct mapping
    if text_lower in TRANSLITERATION_TO_RANJANA:
        return TRANSLITERATION_TO_RANJANA[text_lower]
    
    # Try without spaces/hyphens
    text_clean = text_lower.replace(' ', '').replace('-', '')
    if text_clean in TRANSLITERATION_TO_RANJANA:
        return TRANSLITERATION_TO_RANJANA[text_clean]
    
    # Return original if not found
    return text
