import warnings
import nltk
warnings.filterwarnings("ignore", category=UserWarning, module="nltk")
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

DECAY        = 0.65
MAX_HOPS     = 6
MAX_DEPTH    = 20.0
MAX_BREADTH  = 20.0

# Anchor synsets — all values derived from WordNet's own hierarchy
# x: abstract/mental ↔ concrete/physical
_NOUN_ABSTRACT   = wn.synset('abstraction.n.06')
_NOUN_PHYSICAL   = wn.synset('physical_entity.n.01')
_VERB_ABSTRACT   = wn.synset('think.v.03')
_VERB_PHYSICAL   = wn.synset('move.v.02')
# a: event/process ↔ object/substance (does it happen or just exist?)
_NOUN_EVENT      = wn.synset('event.n.01')
_NOUN_OBJECT     = wn.synset('object.n.01')
_VERB_ACT        = wn.synset('act.v.01')
_VERB_EXIST      = wn.synset('exist.v.01')

class spatialValenceCompute:
    def __init__(self):
        print(f"initialized {self.__class__.__name__}")
        self._domain_cache = {}
        self._lemmatizer   = WordNetLemmatizer()
        self._spell        = SpellChecker()

    def _map_pos(self, treebank_tag: str):
        if treebank_tag.startswith('J'): return wn.ADJ
        if treebank_tag.startswith('V'): return wn.VERB
        if treebank_tag.startswith('N'): return wn.NOUN
        if treebank_tag.startswith('R'): return wn.ADV
        return None

    def _wup_axis(self, synset, n_pos, n_neg, v_pos, v_neg, prefix) -> float:
        key = f"{prefix}:{synset.name()}"
        if key in self._domain_cache:
            return self._domain_cache[key]
        pos = synset.pos()
        if pos == wn.NOUN:
            result = max(-1.0, min(1.0, (synset.wup_similarity(n_pos) or 0.0) - (synset.wup_similarity(n_neg) or 0.0)))
        elif pos == wn.VERB:
            result = max(-1.0, min(1.0, (synset.wup_similarity(v_pos) or 0.0) - (synset.wup_similarity(v_neg) or 0.0)))
        else:
            result = 0.0
        self._domain_cache[key] = result
        return result

    def _domain_x(self, s) -> float:
        return self._wup_axis(s, _NOUN_ABSTRACT, _NOUN_PHYSICAL, _VERB_ABSTRACT, _VERB_PHYSICAL, 'x')

    def _event_a(self, s) -> float:
        return self._wup_axis(s, _NOUN_EVENT, _NOUN_OBJECT, _VERB_ACT, _VERB_EXIST, 'a')

    def _objectivity_b(self, synset) -> float:
        try:
            ss = swn.senti_synset(synset.name())
            return ss.obj_score()
        except Exception:
            return 1.0

    def _swn_valence(self, synset) -> float:
        try:
            ss = swn.senti_synset(synset.name())
            return ss.pos_score() - ss.neg_score()
        except Exception:
            return 0.0

    def _hyponym_c(self, synset) -> float:
        return min(1.0, len(synset.hyponyms()) / MAX_BREADTH)

    def _hypernym_chain(self, synset) -> list:
        if synset is None:
            return []
        chain   = [(synset, 0)]
        current = [synset]
        visited = {synset}
        for hop in range(1, MAX_HOPS + 1):
            next_level = []
            for s in current:
                for h in s.hypernyms():
                    if h is None or h in visited:
                        continue
                    chain.append((h, hop))
                    next_level.append(h)
                    visited.add(h)
            current = next_level
            if not current:
                break
        return chain

    def computeSpatialValence(self, text: str) -> tuple:
        tokens   = nltk.word_tokenize(text.lower())
        pos_tags = nltk.pos_tag(tokens)

        x_acc = y_acc = z_acc = a_acc = b_acc = c_acc = w_total = 0.0

        for word, tag in pos_tags:
            wn_pos = self._map_pos(tag)
            if wn_pos is None:
                continue
            lemma   = self._lemmatizer.lemmatize(word, pos=wn_pos)
            synsets = wn.synsets(lemma, pos=wn_pos)
            if not synsets and wn_pos == wn.ADJ:
                synsets = wn.synsets(lemma, pos=wn.ADJ_SAT)
            if not synsets:
                for fb_pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADJ_SAT, wn.ADV]:
                    if fb_pos == wn_pos: continue
                    fb_lemma = self._lemmatizer.lemmatize(word, pos=fb_pos)
                    synsets  = wn.synsets(fb_lemma, pos=fb_pos)
                    if synsets: break
            if not synsets:
                corrected = self._spell.correction(word)
                if corrected and corrected != word:
                    for fb_pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADJ_SAT, wn.ADV]:
                        fb_lemma = self._lemmatizer.lemmatize(corrected, pos=fb_pos)
                        synsets  = wn.synsets(fb_lemma, pos=fb_pos)
                        if synsets: break
            if not synsets:
                continue

            chain = self._hypernym_chain(synsets[0])

            for ancestor, hop in chain:
                w       = DECAY ** hop
                x_acc  += self._domain_x(ancestor) * w
                y_acc  += self._swn_valence(ancestor) * w
                z_acc  += ancestor.min_depth() * w
                a_acc  += self._event_a(ancestor) * w
                b_acc  += self._objectivity_b(ancestor) * w
                c_acc  += self._hyponym_c(ancestor) * w
                w_total += w

        if w_total == 0.0:
            return (0.0,) * 6

        r = lambda v: round(max(-1.0, min(1.0, v)), 2)
        x = r( x_acc / w_total)
        y = r( y_acc / w_total)
        z = r((z_acc / w_total) / MAX_DEPTH * 2.0 - 1.0)
        a = r( a_acc / w_total)
        b = r((b_acc / w_total) * 2.0 - 1.0)
        c = r((c_acc / w_total) * 2.0 - 1.0)

        return (x, y, z, a, b, c)


if __name__ == "__main__":
    PHRASES = [
        # --- pain / hurt ---
        ("pain",    "my back is absolutely killing me"),
        ("pain",    "my feet are sore and aching badly"),
        ("pain",    "i have a splitting headache right now"),
        ("pain",    "my muscles throb from the workout"),
        # --- joy / happiness ---
        ("joy",     "i am so incredibly happy today"),
        ("joy",     "this is the best day of my entire life"),
        ("joy",     "everything feels wonderful and bright"),
        ("joy",     "my heart is overflowing with gratitude"),
        # --- grief / loss ---
        ("grief",   "i feel completely broken and hollow"),
        ("grief",   "i lost someone i truly loved"),
        ("grief",   "the sadness is slowly consuming me"),
        ("grief",   "i cannot stop crying and i don't know why"),
        # --- fear / danger ---
        ("fear",    "i am terrified and i cannot move"),
        ("fear",    "something is very wrong and i am scared"),
        ("fear",    "my heart is pounding and i am shaking"),
        ("fear",    "i feel paralysed by dread"),
        # --- food / hunger ---
        ("food",    "i am starving and i want pizza"),
        ("food",    "the meal was absolutely delicious"),
        ("food",    "she cooked an incredible dinner"),
        ("food",    "i could eat an entire cake right now"),
        # --- technology ---
        ("tech",    "the server crashed and lost all data"),
        ("tech",    "the algorithm runs in polynomial time"),
        ("tech",    "the api keeps returning timeout errors"),
        ("tech",    "we deployed the new build to production"),
    ]

    _svc = spatialValenceCompute()

    print(f"\n  {'LABEL':<8} {'PHRASE':<44} {'X':>6} {'Y':>6} {'Z':>6} {'A':>6} {'B':>6} {'C':>6}")
    print("  " + "-" * 86)

    current_label = None
    for label, phrase in PHRASES:
        if label != current_label:
            if current_label is not None:
                print()
            current_label = label
        x, y, z, a, b, c = _svc.computeSpatialValence(phrase)
        short = (phrase[:41] + "...") if len(phrase) > 44 else phrase
        print(f"  {label:<8} {short:<44} {x:+.2f} {y:+.2f} {z:+.2f} {a:+.2f} {b:+.2f} {c:+.2f}")
