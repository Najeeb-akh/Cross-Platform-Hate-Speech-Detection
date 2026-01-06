#!/usr/bin/env python3
"""
Generate hate speech lexicon from curated sources.
Saves to data/hate_lexicon.txt for use in feature extraction.

Usage:
    python scripts/download_hate_lexicon.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import List

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def get_expanded_lexicon() -> List[str]:
    """
    Return an expanded hate speech lexicon.
    
    This is a curated list based on research literature and common hate speech patterns.
    """
    lexicon = [
        # Racial slurs (strong)
        "nigger", "nigga", "spic", "kike", "chink", "wetback", "raghead", 
        "gook", "towelhead", "coon", "paki", "curry", "sandnigger", "beaner",
        "jap", "slant", "yellow", "mud", "sand", "camel", "towel", "rag",
        "dune", "sandniga", "sandcoon", "mudslime", "muzzie", "muzlim",
        "dothead", "turban", "spick", "greaser", "border",
        
        # Racial epithets
        "monkey", "apes", "animals", "savages", "vermin", "filth", "trash",
        "subhuman", "inferior", "primitive", "barbaric", "uncivilized",
        
        # Hate rhetoric
        "exterminate", "deport", "go back", "kill", "die", "genocide",
        "gas chamber", "holocaust", "ethnic cleansing", "purge", "eradicate",
        "deporter", "country", "origin", "conspiracy", "invasion", "flood",
        "swarm", "horde", "migrant", "crisis", "refugee",
        
        # White supremacy
        "white power", "white pride", "aryan", "nazi", "fascist", "skinhead",
        "kkk", "14 words", "blood and soil", "white genocide",
        "1488", "88", "14", "words", "blood", "soil", "white", "power",
        "pride", "supremacy", "fascism", "bonehead", "klan", "stormfront",
        
        # Anti-Semitic
        "jew", "jews", "zionist", "globalist", "banker", "shekel",
        "hooked nose", "greedy", "control media", "control world",
        "jewish", "zionism", "nose", "media", "world", "holocaust",
        "denial", "revisionist",
        
        # Anti-Muslim
        "infidel", "jihadi", "terrorist", "bomber", "isis", "taliban",
        "sharia", "backward", "goat", "camel",
        "muslim", "islam", "islamic", "moslem", "mohammedan",
        "jihad", "isil", "daesh", "alqaeda", "shariah", "caliphate", "kafir",
        
        # Anti-immigrant
        "illegal", "alien", "invader", "invasion", "anchor baby",
        "replacement", "great replacement", "they don't belong",
        
        # Gender/sexuality slurs
        "faggot", "fag", "dyke", "tranny", "shemale", "trap",
        "degenerate", "pervert", "groomer", "pedo",
        "fagg", "fagot", "queer", "dike", "lesbo", "sissy", "pussy",
        "cocksucker", "dick", "sucker", "pedophile", "predator",
        
        # Misogyny
        "whore", "slut", "bitch", "cunt", "hoe", "thot",
        "roastie", "femoid", "breeding", "rape",
        "thottie", "foid", "breeder", "raped", "rapist", "incel", "cel",
        "blackpill", "redpill",
        
        # Ableist slurs
        "retard", "retarded", "spastic", "mongoloid", "midget",
        "cripple", "psycho", "crazy", "insane",
        "dwarf", "gimp", "lunatic", "moron", "idiot", "imbecile",
        
        # Dog whistles and coded language
        "dindu", "jogger", "skittles", "basketball american",
        "crime statistics", "13 50", "despite", "echo",
        "noticer", "pattern recognition", "urban", "thug",
        "basketball", "american", "crime", "statistics", "1350",
        "thirteen", "fifty", "pattern", "recognition", "gang",
        "ghetto", "hood", "welfare", "queen", "king",
        
        # Violence incitement
        "lynch", "hang", "shoot", "burn", "attack", "assault",
        "beat", "eliminate", "remove", "cleanse",
        "string", "rope", "tree", "fire", "gas", "chamber", "shower",
        "oven", "wipe", "destroy", "murder", "execute", "stab", "cut",
        
        # Dehumanization
        "cockroach", "rat", "parasite", "disease", "plague",
        "cancer", "virus", "infestation", "breeding",
        "pest", "infection", "scum", "garbage", "waste", "sewage", "rodent",
        
        # Platform-specific terms
        "kek", "pepe", "npc", "soy", "boy", "cuck", "cuckold",
        "beta", "alpha", "chad", "stacy", "normie", "reddit", "moment",
        "based", "redpilled", "blackpilled",
        
        # Other hate terms
        "gypsy", "pikey", "hick", "redneck", "trailer trash",
        "white trash", "ghetto", "hood", "savage", "thug"
    ]
    
    # Add variations
    expanded = set(lexicon)
    
    # Add plurals
    for term in lexicon:
        if not term.endswith('s'):
            expanded.add(term + 's')
    
    # Add some common misspellings/variations
    variations = {
        "nigger": ["nigg3r", "n1gger", "nig", "nigg@", "n!gger"],
        "faggot": ["f4ggot", "fag", "f@ggot", "fagg0t"],
        "kike": ["k1ke", "k!ke", "kik3"],
        "chink": ["ch1nk", "ch!nk"],
        "gook": ["g00k", "g00k"],
        "jap": ["j@p"],
        "spic": ["sp1c", "sp!c"],
        "wetback": ["wetb@ck"],
        "coon": ["c00n", "c0on"],
        "paki": ["p@ki", "p4ki"],
        "dyke": ["d1ke", "d!ke"],
        "tranny": ["tr@nny", "tr4nny"],
        "retard": ["ret@rd", "r3tard"],
        "jew": ["j3w", "j@w"],
        "muslim": ["musl1m", "muzlim"],
        "terrorist": ["terr0rist", "terr0r1st"],
    }
    
    for base, vars in variations.items():
        expanded.update(vars)
    
    return sorted(list(expanded))


def save_lexicon(terms: List[str], output_path: Path):
    """Save lexicon to file, one term per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for term in sorted(set(terms)):
            f.write(f"{term.lower()}\n")
    
    LOGGER.info(f"Saved {len(terms)} terms to {output_path}")


def main():
    output_path = Path("data/hate_lexicon.txt")
    
    LOGGER.info("="*80)
    LOGGER.info("HATE SPEECH LEXICON GENERATION")
    LOGGER.info("="*80)
    
    # Get curated lexicon
    curated_terms = get_expanded_lexicon()
    
    LOGGER.info(f"Generated {len(curated_terms)} terms from curated lexicon")
    
    # Filter to keep only meaningful terms (length > 2, alpha-only)
    filtered_terms = [
        term for term in curated_terms 
        if len(term) > 2 and (term.isalpha() or ' ' in term)
    ]
    
    save_lexicon(filtered_terms, output_path)
    
    LOGGER.info(f"\n✓ Lexicon ready with {len(filtered_terms)} hate speech terms")
    LOGGER.info(f"  Location: {output_path.absolute()}")
    LOGGER.info(f"\nNote: This lexicon is for research purposes in hate speech detection.")
    LOGGER.info("      Terms are offensive by nature. Handle with care.")


if __name__ == "__main__":
    main()
