"""
generate_synthetic.py
Generates a balanced synthetic social-media dataset for 3-class suicidal-ideation
detection (high_risk / moderate_risk / low_risk) and saves it to
data/raw/synthetic_social_media_detection.csv

Classes
-------
high_risk     (label=2) – 800 samples
moderate_risk (label=1) – 800 samples
low_risk      (label=0) – 800 samples

Total: 2400 samples
Columns: text, risk_level, source (reddit/twitter), subreddit
"""

import random
import csv
import os
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------------------
# Output path  (repo_root/data/raw/...)
# ---------------------------------------------------------------------------
_THIS_FILE   = Path(__file__).resolve()
_REPO_ROOT   = _THIS_FILE.parent.parent.parent   # src/data/ -> src/ -> repo root
OUTPUT_DIR   = _REPO_ROOT / "data" / "raw"
OUTPUT_FILE  = OUTPUT_DIR / "synthetic_social_media_detection.csv"

# ---------------------------------------------------------------------------
# Shared context pools
# ---------------------------------------------------------------------------

_TIMES_OF_DAY = [
    "at 3am", "late last night", "this morning", "tonight", "earlier today",
    "lying in bed", "at midnight", "after work", "after school",
    "this afternoon", "just now", "all day", "all week",
    "in the middle of the night", "during my lunch break",
]

_LOCATIONS = [
    "in my room", "at home", "alone in my apartment", "sitting in my car",
    "in the bathroom", "on the rooftop", "outside by myself", "at the park",
    "in the dark", "in the hospital waiting room", "on the bus home",
    "at my desk", "in my childhood bedroom",
]

_REASONS = [
    "after losing my job", "after my break-up", "after failing my exams",
    "after everyone abandoned me", "after years of fighting this",
    "after my last panic attack", "after my mom died",
    "since my divorce finalised", "after being bullied again",
    "because I can't pay my bills", "because I'm so tired of everything",
    "after what happened last week", "because nothing works anymore",
    "since the accident", "after the argument with my family",
    "because I've lost everything", "after the relapse",
]

_REDDIT_OPENERS = [
    "Hey everyone, long post incoming but I need to get this out.",
    "I don't know who else to talk to.",
    "Throwaway because people know my main.",
    "Not sure why I'm posting this. Maybe I just need someone to read it.",
    "I've been lurking here for a long time and finally feel ready to share.",
    "Please don't delete this mod, I really need help.",
    "I've been bottling this up for months.",
    "TW: please be kind, I'm fragile right now.",
    "I'm sorry if this is the wrong sub.",
    "I don't expect anyone to reply but here it is anyway.",
    "First time posting here. I hope someone can understand.",
    "I don't know how to start this.",
]

_REDDIT_CLOSERS = [
    "Anyway, thanks for reading.",
    "I don't know what I'm looking for, maybe just someone to hear me.",
    "Edit: thanks for the kind words, I'm still here.",
    "I'll try to reply to comments.",
    "If you've been here too, please reach out.",
    "I'm going to try to sleep now.",
    "Sorry for the wall of text.",
    "I just needed to say it out loud somewhere.",
    "I'm not sure what I expected posting this.",
]

_TWEET_ENDINGS = [
    "#MentalHealth", "#depressed", "#alone", "#broken",
    "#cantdothisanymore", "#hopeless", "#help", "#sad",
    "", "", "", "", "", "",   # plain tweets are more common
]

# ---------------------------------------------------------------------------
# HIGH RISK  (800 samples)
# ---------------------------------------------------------------------------

_HIGH_RISK_SEEDS = [
    "I've been thinking about ending it all",
    "nobody would miss me if I was gone",
    "I've written my goodbye note",
    "I don't see any reason to keep going",
    "I've decided how I'm going to do it",
    "I want to kill myself",
    "I can't keep living like this",
    "I've said my goodbyes",
]

_HIGH_RISK_EXPANSIONS = [
    "I have a plan and I know when I'm going to do it.",
    "I've been researching methods for weeks.",
    "Tonight feels like the night.",
    "I've made peace with it.",
    "I gave away my things to my friends yesterday.",
    "I'm not afraid anymore.",
    "I just wanted someone to know.",
    "I left a note for my family.",
    "I'm so exhausted from pretending to be okay.",
    "There's nothing left for me here.",
    "I've tried everything and nothing helps.",
    "The pain is unbearable and I can't see another way out.",
    "I keep staring at the pills on my nightstand.",
    "I don't want help, I just want it to stop.",
    "I feel completely empty and done with life.",
    "I've been saying goodbye to everyone without them knowing.",
    "I drove to the bridge again last night.",
    "I can feel myself shutting down for good.",
    "I've cancelled all my plans for the future.",
    "I stopped caring about consequences.",
    "I wrote letters to the people I care about.",
    "I've been stockpiling pills.",
    "I went looking at the bridge near my house today.",
    "I can't see a version of the future where I'm still here.",
]


def _make_high_risk_rows(n: int, rng: random.Random) -> list:
    rows = []
    for _ in range(n):
        seed      = rng.choice(_HIGH_RISK_SEEDS)
        expansion = rng.choice(_HIGH_RISK_EXPANSIONS)
        time_ctx  = rng.choice(_TIMES_OF_DAY)
        location  = rng.choice(_LOCATIONS)
        reason    = rng.choice(_REASONS)

        source = rng.choice(["reddit", "reddit", "twitter"])   # 2/3 reddit
        if source == "reddit":
            opener    = rng.choice(_REDDIT_OPENERS)
            closer    = rng.choice(_REDDIT_CLOSERS)
            subreddit = rng.choice(["r/SuicideWatch", "r/depression", "r/teenagers"])
            templates = [
                f"{opener} {seed} {reason}. {expansion} I'm {time_ctx}, {location}. {closer}",
                f"{opener} {time_ctx}, {location}. {seed} {reason}. {expansion} {closer}",
                f"{opener} {reason}, {seed}. {expansion} It's {time_ctx}. {closer}",
                f"{opener} {expansion} {seed} {reason}. {time_ctx}, {location}. {closer}",
                f"{opener} I need to be honest. {seed} {reason}. {expansion} {closer}",
            ]
        else:
            subreddit = None
            ending    = rng.choice(_TWEET_ENDINGS)
            templates = [
                f"{seed} {reason} {time_ctx} {ending}".strip(),
                f"{time_ctx} {location} and {seed.lower()} {reason}. {ending}".strip(),
                f"{seed} {reason}. {expansion[:65]} {ending}".strip(),
                f"Can't do this anymore. {seed} {time_ctx}. {ending}".strip(),
                f"I'm done. {seed}. {reason}. {ending}".strip(),
            ]

        text = rng.choice(templates).strip()
        rows.append({
            "text":       text,
            "risk_level": "high_risk",
            "source":     source,
            "subreddit":  subreddit,
        })
    return rows


# ---------------------------------------------------------------------------
# MODERATE RISK  (800 samples)
# ---------------------------------------------------------------------------

_MODERATE_RISK_SEEDS = [
    "I feel so empty inside",
    "I'm such a burden to everyone around me",
    "I can't remember the last time I felt happy",
    "I've been crying for days and I don't even know why",
    "everything feels pointless",
    "I hate myself so much",
]

_MODERATE_RISK_EXPANSIONS = [
    "I don't know how much longer I can keep going like this.",
    "I wake up every morning wishing I hadn't.",
    "I'm so tired of faking smiles.",
    "Nobody really sees how much pain I'm in.",
    "I've stopped doing all the things I used to enjoy.",
    "I feel like I'm disappearing.",
    "Sometimes I just want to close my eyes and not wake up.",
    "I'm drowning and nobody notices.",
    "I've lost all motivation to do anything.",
    "I barely get out of bed most days.",
    "I cry in the shower so no one hears.",
    "I push everyone away because I don't deserve love.",
    "I feel like a ghost in my own life.",
    "The sadness never fully goes away.",
    "I'm broken in ways I can't explain.",
    "I've been isolating for months now.",
    "Everything feels gray — nothing has colour anymore.",
    "I just want the noise in my head to stop.",
    "Eating, sleeping, existing — it all feels like too much effort.",
    "I haven't felt like myself in so long I don't remember who I am.",
    "I don't enjoy anything I used to love.",
    "Every day feels the same and I hate it.",
]

_MODERATE_HASHTAGS = [
    "#depressed", "#depression", "#mentalhealth", "#anxious",
    "#MentalHealthMatters", "#sad", "#lonely", "#exhausted",
    "#broken", "#invisible", "#numb", "#struggling",
    "", "", "", "", "",   # plain posts common
]


def _make_moderate_risk_rows(n: int, rng: random.Random) -> list:
    rows = []
    for _ in range(n):
        seed      = rng.choice(_MODERATE_RISK_SEEDS)
        expansion = rng.choice(_MODERATE_RISK_EXPANSIONS)
        time_ctx  = rng.choice(_TIMES_OF_DAY)
        reason    = rng.choice(_REASONS)

        source = rng.choice(["reddit", "reddit", "twitter"])
        if source == "reddit":
            opener    = rng.choice(_REDDIT_OPENERS)
            closer    = rng.choice(_REDDIT_CLOSERS)
            subreddit = rng.choice(["r/depression", "r/teenagers", "r/mentalhealth"])
            templates = [
                f"{opener} {seed} {reason}. {expansion} {closer}",
                f"{opener} {time_ctx}, {seed}. {expansion} {closer}",
                f"{opener} {expansion} {seed} {reason}. {closer}",
                f"{opener} {seed}. {expansion} {time_ctx}. {closer}",
                f"{opener} {reason}. {seed}. {expansion} {closer}",
            ]
        else:
            subreddit = None
            htag1 = rng.choice(_MODERATE_HASHTAGS)
            htag2 = rng.choice(_MODERATE_HASHTAGS)
            tags  = f"{htag1} {htag2}".strip()
            templates = [
                f"{seed} {reason} {tags}".strip(),
                f"{time_ctx} and {seed.lower()}. {expansion[:70]} {tags}".strip(),
                f"{seed}. {expansion[:80]} {tags}".strip(),
                f"Just feeling it {time_ctx}. {seed} {tags}".strip(),
                f"{seed} {reason}. {tags}".strip(),
            ]

        text = rng.choice(templates).strip()
        rows.append({
            "text":       text,
            "risk_level": "moderate_risk",
            "source":     source,
            "subreddit":  subreddit,
        })
    return rows


# ---------------------------------------------------------------------------
# LOW RISK  (800 samples)
# ---------------------------------------------------------------------------

_LOW_RISK_TWEETS = [
    "just made the best cup of coffee I've ever had in my life",
    "Netflix just dropped another season and I'm not sleeping tonight lol",
    "my professor actually gave us extra time on the midterm, absolute legend",
    "traffic was terrible today but at least the podcast was good",
    "anyone else forget to eat lunch and then wonder why they're tired",
    "finally beat that level I've been stuck on for two weeks!!!",
    "ordered pizza and the delivery guy gave me extra garlic bread, best day ever",
    "my cat knocked my water off my desk again, zero remorse as always",
    "woke up late and still made it to work on time, feeling invincible",
    "Mondays are just evil I don't make the rules",
    "found a $20 in my old jacket, it's like a gift from past me",
    "spent 3 hours watching food videos and now I'm starving at 2am",
    "my plant is thriving and I feel like a proud parent",
    "gym was packed but I got a great workout in anyway",
    "just remembered I have leftovers in the fridge, this day just improved dramatically",
    "my friend sent me the funniest meme at the worst possible time",
    "anyone else get unreasonably excited about a good parking spot",
    "I really need a vacation but also I need to stop spending money on coffee",
    "coworker brought donuts today, productivity through the roof",
    "my team won last night! absolute madness, I completely lost my voice",
    "just finished a great book, now I have that empty post-book feeling",
    "dog park was amazing today, made three new dog friends (the humans were ok too)",
    "rain finally stopped and the whole city smells amazing",
    "study session at the library and I actually understood everything",
    "bought new sneakers and I'm irrationally happy about it honestly",
    "hot chocolate season is officially back and I am HERE for it",
    "just got a haircut and feel like a completely different person",
    "nap time on a Sunday should be a protected human right",
    "finally tried that restaurant everyone was talking about, 10/10 would recommend",
    "first day of spring and everyone in the park is just smiling at each other",
    "impulse-bought a puzzle and I'm now 300 pieces in, no regrets",
    "cleaned my whole apartment and the level of satisfaction is unreal",
    "my dog learned a new trick on the first try, she's a genius",
    "road trip this weekend with my friends, already excited",
    "made banana bread from scratch, house smells incredible",
    "scored tickets to the game next weekend!!! so hyped",
    "sometimes a good night's sleep really does fix everything",
    "the sunrise on my commute today was actually stunning",
    "just downloaded a new playlist and my commute was so much better",
    "afternoon iced coffee hitting different today",
]

_LOW_RISK_REDDIT_POSTS = [
    "Just wanted to share that I finally got a job offer after 3 months of searching. Don't give up folks!",
    "Casual vent: my landlord fixed the heater after two weeks. Small win but I'll take it.",
    "Anyone else find studying outside way more productive? The park near my school is honestly perfect.",
    "I made pasta from scratch for the first time tonight and it actually turned out great!",
    "Hot take: the second season of that show is actually better than the first. Discuss.",
    "Local sports team had an incredible game last night, the atmosphere was electric.",
    "My roommate and I finally agreed on what to watch. Only took 45 minutes of negotiation.",
    "Had the most chaotic commute today — train delayed, coffee spilled, but I laughed the whole way.",
    "Birthday dinner was amazing. I'm full and happy and going to bed early like a real adult.",
    "Just finished a 5k run, first time I've done it without stopping. Really proud of that.",
    "Weekend hiking trip was exactly what I needed. Left my phone in the car and just existed.",
    "The grocery store was out of my usual brand so I tried something new and honestly it slaps.",
    "My little sister got into her first choice college! So proud of her.",
    "I've been reducing screen time before bed and I actually slept properly for the first time in months.",
    "Tried meal prepping for the week. Spent 2 hours Sunday, saved probably 4 hours total. Worth it.",
    "My dog learned a new trick after only three days of training, I'm convinced he's a genius.",
    "Afternoon nap plus iced coffee is the combo of champions.",
    "We adopted a kitten today. There is no going back. My life belongs to her now.",
    "Finally got around to cleaning my desk and I feel like I've unlocked a new level of focus.",
    "We had the best potluck dinner with neighbours. I didn't know so many people could cook.",
    "Started learning the guitar this week. My fingers hurt but I played a full chord today.",
    "Spontaneously decided to go to a museum today and honestly it was the best decision.",
    "Made it through my first week at the new job. Nervous but also really excited.",
    "Friend surprised me with concert tickets. I've been listening to nothing else all week.",
    "Went thrift shopping and found the most incredible jacket for $6. Fashion is alive.",
    "Did a charity 5k run with my coworkers. Surprisingly really fun.",
    "The farmer's market near me has the best fresh bread on Saturdays.",
    "My book club actually had a great discussion tonight for once.",
    "Spent the afternoon volunteering at the animal shelter. 10/10 would recommend for the soul.",
    "Successfully cooked a new recipe without any disasters. Growth.",
]

_LOW_RISK_EMOJIS = [
    "☕", "😂", "🙌", "🎮", "🍕", "🐶", "🏃", "📚",
    "🎉", "😴", "🌧️", "🔥", "💯", "😅", "🤣", "👀", "🌟",
    "", "", "", "", "", "", "", "", "",  # no emoji more common
]


def _make_low_risk_rows(n: int, rng: random.Random) -> list:
    rows = []
    for _ in range(n):
        source = rng.choice(["reddit", "twitter"])

        if source == "reddit":
            subreddit = rng.choice([None, "r/AskReddit", "r/teenagers", "r/casualconversation"])
            base = rng.choice(_LOW_RISK_REDDIT_POSTS)
            # Occasionally append a short second sentence
            if rng.random() < 0.4:
                extra = rng.choice([
                    " Needed that today.",
                    " Good day overall.",
                    " Anyway, hope everyone's doing well.",
                    " Small things matter.",
                    " Shoutout to the simple pleasures.",
                    " Grateful for the little wins.",
                    " Life is good sometimes.",
                    " Hope you all have a great week.",
                ])
                base = base + extra
            text = base.strip()
        else:
            subreddit = None
            base  = rng.choice(_LOW_RISK_TWEETS)
            emoji = rng.choice(_LOW_RISK_EMOJIS)
            htag  = rng.choice(["", "", "", "#mood", "#relatable", "#mondaymotivation", "#blessed", "#vibes"])
            parts = [base]
            if emoji:
                parts.append(emoji)
            if htag:
                parts.append(htag)
            text = " ".join(parts).strip()

        rows.append({
            "text":       text,
            "risk_level": "low_risk",
            "source":     source,
            "subreddit":  subreddit,
        })
    return rows


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_dataset(
    n_per_class: int = 800,
    output_path: Path = OUTPUT_FILE,
    seed: int = SEED,
) -> list:
    """
    Generate n_per_class samples for each of the 3 risk levels (total 3 * n_per_class).

    Parameters
    ----------
    n_per_class : int
        Number of samples per class.
    output_path : Path
        CSV destination file.
    seed : int
        Random seed.

    Returns
    -------
    list of dicts (all generated rows, shuffled).
    """
    rng = random.Random(seed)

    print(f"Generating {n_per_class} samples per class ({n_per_class * 3} total) ...")

    high_rows     = _make_high_risk_rows(n_per_class, rng)
    moderate_rows = _make_moderate_risk_rows(n_per_class, rng)
    low_rows      = _make_low_risk_rows(n_per_class, rng)

    all_rows = high_rows + moderate_rows + low_rows
    rng.shuffle(all_rows)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["text", "risk_level", "source", "subreddit"]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows  →  {output_path}")
    print_stats(all_rows)
    return all_rows


# ---------------------------------------------------------------------------
# Statistics printer
# ---------------------------------------------------------------------------

def print_stats(rows: list) -> None:
    """Print class, source, and subreddit distribution."""
    total = len(rows)
    if total == 0:
        print("No rows to report.")
        return

    class_counts     = Counter(r["risk_level"] for r in rows)
    source_counts    = Counter(r["source"] for r in rows)
    subreddit_counts = Counter(r["subreddit"] for r in rows if r["subreddit"])
    reddit_total     = source_counts.get("reddit", 0)

    print("\n" + "=" * 55)
    print("  Dataset Statistics")
    print("=" * 55)

    print("\nClass Distribution:")
    for label in ["high_risk", "moderate_risk", "low_risk"]:
        cnt = class_counts.get(label, 0)
        pct = cnt / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:<18} {cnt:>5}  ({pct:5.1f}%)  {bar}")

    print("\nSource Distribution:")
    for src, cnt in source_counts.most_common():
        pct = cnt / total * 100
        print(f"  {src:<12} {cnt:>5}  ({pct:5.1f}%)")

    print("\nSubreddit Distribution (Reddit posts only):")
    for sub, cnt in subreddit_counts.most_common():
        pct = cnt / reddit_total * 100 if reddit_total else 0
        print(f"  {sub:<28} {cnt:>5}  ({pct:5.1f}% of Reddit)")

    print("\nText Length Stats:")
    lengths = [len(r["text"]) for r in rows]
    print(f"  Min chars : {min(lengths)}")
    print(f"  Max chars : {max(lengths)}")
    print(f"  Avg chars : {sum(lengths) / len(lengths):.0f}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_dataset(n_per_class=800)
