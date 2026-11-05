"""Heuristic lexicons. All are starting points, not ground truth: extend them from real scripts."""
from __future__ import annotations

import re

# Common romanised Malayalam ("Manglish") words. Used only to separate Manglish from English.
MANGLISH_WORDS = set(
    """
    enthu entha enthaa enthinu ente ninte njan nan nee ningal ningalude avan aval avar ivan ival
    ithu athu ethu evide avide ivide enganeyaa engane aanu anu alla alle ille illa undu undo unde
    venam vendaa vende vendi cheyyum cheyyam cheyyu cheythu poyi poyo vannu vannille vaa va po
    pokam pokaam kaanam kaanaam kandu kandilla parayu paranju parayam ariyilla ariyam ariyaruthu
    mathi sheri seri pinne appo appol ippo ippol innu innale naale amme achan appa ammae chetta
    chettan chechi mone mole molu monu kutty kuttee daivame daivam aiyyo ayyo oru onnum onnu
    ellam ellaam ellavarkum ellavarum kadam kaalam mosham joli tharumo tharum odikkumo odikkam
    theerkkanam theerum ayal aara aaru nannayi valare kurachu kooduthal thirichu varatte varum
    varu veezhunnathu veezhum ninnu kittiyatha kittiyathu kittum aanallo aanenkil allenkil
    athalle ithalle enikku ninakku avanu avalkku namukku ningalkku avide ivideyum evideyum
    pennu pennine aanu paavam kollam poli machane machan mwone ente chetta ikka ikkaa mashe
    saare sir-nte kayal ariyum ariyoo kelkku kelkkoo nokku nokkoo irikku irikkoo vaayo
    sathyam thanne alle athe aahaa ayyoo ente-yum pazhaya puthiya nalla ketta chila palarum
    thaan thaanu njangal ningal naam nammal ente-yaa ullu ulla undaayirunnu aayirunnu aayirikkum
    """.split()
)

MANGLISH_SUFFIXES = (
    "aanu", "anu", "ille", "illa", "alle", "unnu", "unnathu", "ikkum", "ilekku", "ilum",
    "inte", "yude", "ode", "ariyilla", "kkum", "yilla",
)

# Scene-flag keywords. Latin entries match whole words (lowercase); Malayalam entries match as substrings.
FLAG_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "fight": {
        "latin": ("fight", "punch", "kick", "brawl", "chase", "gun", "shoot", "stab",
                  "knife", "sword", "attack", "weapon", "adi", "kuththu"),
        "ml": ("സംഘട്ടന", "അടി", "വെടി", "കത്തി", "വാൾ", "ഓടിച്ച"),
    },
    "song": {
        "latin": ("song", "paattu", "montage"),
        "ml": ("ഗാനം", "പാട്ട്"),
    },
    "crowd": {
        "latin": ("crowd", "mob", "procession", "festival", "pooram", "utsavam", "rally",
                  "gathering", "tourists", "devotees", "jathra"),
        "ml": ("ഉത്സവ", "പൂരം", "ജനക്കൂട്ടം", "ഘോഷയാത്ര"),
    },
    "water": {
        "latin": ("boat", "boats", "canoe", "backwaters", "river", "sea", "ship", "houseboat",
                  "waves", "drown", "drowning", "swim", "kayal", "vallam", "lake", "beach"),
        "ml": ("വള്ളം", "കായൽ", "കടൽ", "പുഴ", "ബോട്ട്", "തോണി"),
    },
    "weather": {
        "latin": ("rain", "storm", "flood", "cyclone", "lightning", "thunder", "monsoon"),
        "ml": ("മഴ", "കൊടുങ്കാറ്റ്", "വെള്ളപ്പൊക്കം", "ഇടി"),
    },
    "vehicle": {
        "latin": ("car", "jeep", "bus", "lorry", "truck", "bike", "train", "auto", "taxi", "van",
                  "scooter", "driving", "vandi"),
        "ml": ("ജീപ്പ്", "ബസ്", "ലോറി", "വണ്ടി", "ട്രെയിൻ", "ഓട്ടോ"),
    },
    "animal": {
        "latin": ("elephant", "dog", "snake", "cow", "bull", "horse", "buffalo", "tiger", "cat"),
        "ml": ("ആന", "നായ", "പാമ്പ്", "പശു", "കാള"),
    },
    "children": {
        "latin": ("child", "children", "kid", "kids", "baby", "school", "toddler"),
        "ml": ("കുട്ടി", "കുഞ്ഞ്", "സ്കൂൾ"),
    },
    "fire_explosion": {
        "latin": ("explosion", "explodes", "blast", "fire", "burning", "burns", "bomb"),
        "ml": ("സ്ഫോടന", "തീ", "പൊട്ടിത്തെറി"),
    },
    "stunt": {
        "latin": ("jump", "jumps", "leap", "leaps", "crash", "helicopter",
                  "drone", "cliff"),
        "ml": ("ചാടി", "വീഴ്ച"),
    },
    "flashback": {
        "latin": ("flashback",),
        "ml": ("ഫ്ലാഷ്ബാക്ക്", "ഓർമ്മ"),
    },
    "substance_review": {
        "latin": ("cigarette", "smoking", "beedi", "alcohol", "liquor", "toddy", "bar", "drinks",
                  "drunk", "whisky", "rum", "kallu", "ganja"),
        "ml": ("ബീഡി", "കള്ള്", "മദ്യം", "ബാർ", "സിഗരറ്റ്"),
    },
}

# Relative production weights per flag (uncalibrated; replace with your own budgeting data).
FLAG_WEIGHTS = {
    "fight": 2.0, "song": 3.0, "crowd": 2.0, "water": 2.0, "weather": 1.5, "vehicle": 1.0,
    "animal": 1.5, "children": 0.75, "fire_explosion": 2.5, "stunt": 1.5, "flashback": 0.5,
    "substance_review": 0.0,
}
FLAG_LABELS = {
    "fight": "Fight / chase / weapons", "song": "Song / montage", "crowd": "Crowd / festival",
    "water": "Water / boats", "weather": "Rain / storm", "vehicle": "Vehicles",
    "animal": "Animals", "children": "Children", "fire_explosion": "Fire / explosion",
    "stunt": "Stunts / heights", "flashback": "Flashback",
    "substance_review": "Tobacco / alcohol (censor review)",
}

TIME_WORDS = {
    "NIGHT": ("night", "midnight", "രാത്രി", "അർദ്ധരാത്രി"),
    "DAY": ("day", "morning", "noon", "afternoon", "പകൽ", "രാവിലെ", "ഉച്ച"),
    "TWILIGHT": ("dawn", "dusk", "evening", "sunset", "sunrise", "twilight", "പുലർച്ചെ",
                 "സന്ധ്യ", "വൈകുന്നേരം"),
}

RESERVED_CUES = {
    "FIGHT", "SONG", "FLASHBACK", "FLASH BACK", "MONTAGE", "TITLE", "SUPER", "NOTE", "NOTES",
    "SFX", "SOUND", "MUSIC", "BGM", "INSERT", "CUT TO", "FADE IN", "FADE OUT", "FADE TO",
    "DISSOLVE TO", "INTERCUT", "CUT", "SCENE", "ACT", "END", "THE END", "TRANSITION",
    "PRE-INTERVAL", "INTERVAL", "CLIMAX", "FLASHFORWARD", "SUBTITLE", "CAPTION",
}

TRANSITION_PREFIXES = ("CUT TO", "FADE", "DISSOLVE", "SMASH CUT", "MATCH CUT", "THE END", "INTERVAL")

_cache: dict[str, re.Pattern] = {}


def flag_pattern(flag: str) -> tuple[re.Pattern, tuple[str, ...]]:
    if flag not in _cache:
        words = FLAG_KEYWORDS[flag]["latin"]
        _cache[flag] = re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", re.I)
    return _cache[flag], FLAG_KEYWORDS[flag]["ml"]
