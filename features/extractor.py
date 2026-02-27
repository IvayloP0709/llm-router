import re
import spacy
import tiktoken
import textstat
from transformers import pipeline as hf_pipeline


# Load models
_nlp = spacy.load("en_core_web_sm")
_enc = tiktoken.get_encoding("cl100k_base")  

# Zero-shot NLI classifier — understands meaning, needs no keyword lists.
_classifier = hf_pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-deberta-v3-small",
)


# Label sets for zero-shot classification 
TASK_LABELS = [
    "translation",
    "code writing",
    "debugging",
    "mathematics",
    "creative writing",
    "analysis",
    "summarization",
    "question answering",
]

DOMAIN_LABELS = [
    "medical",
    "legal",
    "finance",
    "science",
    "technology",
    "education",
]

CODE_MARKERS = ["```", "def ", "function ", "class ", "import ", "=>",
                "const ", "var ", "let ", "print(", "console.log", "if __name__"]



def extract_features(query: str) -> dict:
    """
    Convert a raw query string into a flat dict of numeric features.

    Pipeline:
      spaCy      → sentence segmentation, POS ratios, NER entity signals
      tiktoken   → actual LLM token count (what APIs bill by)
      textstat   → readability score
      transformers (zero-shot NLI) → task type & domain as float probabilities,
                                     no hardcoded keyword lists
    """
    # Parse once — spaCy gives sentences, POS tags, and entities in one pass
    doc = _nlp(query)

    word_tokens = [t for t in doc if not t.is_space and not t.is_punct]
    word_count  = max(len(word_tokens), 1)
    sentences   = list(doc.sents)

    features = {}

    # ── Group 1: Structural ──────────────────────────────────────
    features["token_count"]          = len(_enc.encode(query))
    features["word_count"]           = word_count
    features["char_count"]           = len(query)
    features["sentence_count"]       = len(sentences)
    features["avg_word_length"]      = round(
        sum(len(t.text) for t in word_tokens) / word_count, 2
    )
    features["avg_sentence_length"]  = round(word_count / len(sentences), 2)
    features["question_mark_count"]  = query.count("?")
    features["has_url"]              = int(bool(re.search(r'https?://', query)))
    features["has_code_block"]       = int(any(m in query for m in CODE_MARKERS))

    # ── Group 2: Readability (textstat) ─────────────────────────
    features["flesch_kincaid_grade"] = (
        textstat.flesch_kincaid_grade(query) if word_count >= 3 else 0.0
    )

    # ── Group 3: Vocabulary richness (spaCy lemmatization) ───────
    lemmas = [t.lemma_.lower() for t in word_tokens]
    features["unique_word_ratio"] = round(len(set(lemmas)) / word_count, 3)

    # ── Group 4: POS ratios — spaCy part-of-speech tagging ──────
    total_tokens = max(len(doc), 1)
    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0, "PROPN": 0}
    for t in doc:
        if t.pos_ in pos_counts:
            pos_counts[t.pos_] += 1

    features["noun_ratio"]  = round(pos_counts["NOUN"]  / total_tokens, 3)
    features["verb_ratio"]  = round(pos_counts["VERB"]  / total_tokens, 3)
    features["adj_ratio"]   = round(pos_counts["ADJ"]   / total_tokens, 3)
    features["adv_ratio"]   = round(pos_counts["ADV"]   / total_tokens, 3)
    features["propn_ratio"] = round(pos_counts["PROPN"] / total_tokens, 3)

    # ── Group 5: Named Entity Recognition — spaCy NER
    ent_labels = {ent.label_ for ent in doc.ents}
    features["entity_count"]    = len(doc.ents)
    features["has_money_ent"]   = int("MONEY"   in ent_labels)
    features["has_person_ent"]  = int("PERSON"  in ent_labels)
    features["has_org_ent"]     = int("ORG"     in ent_labels)
    features["has_date_ent"]    = int("DATE"    in ent_labels)
    features["has_law_ent"]     = int("LAW"     in ent_labels)
    features["has_product_ent"] = int("PRODUCT" in ent_labels)

    # ── Group 6: Task type — zero-shot classification ─────────────
    # The NLI model reads the query and each label as plain English text and
    # returns a probability for each label. No keyword list, no regex —
    # "write a poem" → task_creative_writing ≈ 0.94 without any special rule.
    # multi_label=True lets a query score high on multiple tasks at once.
    task_result = _classifier(query, candidate_labels=TASK_LABELS, multi_label=True)
    for label, score in zip(task_result["labels"], task_result["scores"]):
        key = "task_" + label.replace(" ", "_")
        features[key] = round(float(score), 4)

    # ── Group 7: Domain — zero-shot classification ────────────────
    domain_result = _classifier(query, candidate_labels=DOMAIN_LABELS, multi_label=True)
    for label, score in zip(domain_result["labels"], domain_result["scores"]):
        features[f"domain_{label}"] = round(float(score), 4)

    return features


# test

if __name__ == "__main__":
    test_queries = [
        "Write a poem about autumn leaves",      
        "Translate hello to French",
        "Debug this Python function: def add(a,b): return a-b",
        "Analyze the risk-reward tradeoff of investing in Bitcoin",
        "Prove that the sum of two even numbers is always even",
    ]
    for q in test_queries:
        feats = extract_features(q)
        print(f"\nQuery: {q!r}")

        for k, v in feats.items():
            if (k.startswith("task_") or k.startswith("domain_")) and v > 0.3:
                print(f"  {k:35s} = {v:.4f}")
