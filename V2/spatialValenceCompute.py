import os
import struct
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

_CACHE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "MemoryStructures", "word_cache.lmdb")
_PACK_FMT   = "13d"  # 13 doubles = 104 bytes (6+6 accumulators + shared weight)

# Second 6D anchors — world model layer
# D: living/organic ↔ artificial/made
_NOUN_LIVING   = wn.synset('living_thing.n.01')
_NOUN_ARTIFACT = wn.synset('artifact.n.01')
_VERB_GROW     = wn.synset('grow.v.01')
_VERB_BUILD    = wn.synset('build.v.01')
# E: spatial/location ↔ process/change
_NOUN_LOCATION = wn.synset('location.n.01')
_NOUN_PROCESS  = wn.synset('process.n.06')
_VERB_TRAVEL   = wn.synset('travel.v.01')
_VERB_CHANGE   = wn.synset('change.v.01')
# F: communicative/information ↔ material/substance
_NOUN_COMMS    = wn.synset('communication.n.02')
_NOUN_MATTER   = wn.synset('substance.n.01')
_VERB_SPEAK    = wn.synset('communicate.v.01')
_VERB_CREATE   = wn.synset('create.v.02')
# G: causal agent ↔ phenomenon
_NOUN_AGENT    = wn.synset('causal_agent.n.01')
_NOUN_PHENOM   = wn.synset('phenomenon.n.01')
_VERB_CAUSE    = wn.synset('cause.v.01')
_VERB_HAPPEN   = wn.synset('happen.v.01')
# H: cognitive/mental ↔ attribute/property
_NOUN_COGNIT   = wn.synset('cognition.n.01')
_NOUN_ATTRIB   = wn.synset('attribute.n.01')
_VERB_KNOW     = wn.synset('know.v.01')
_VERB_HAVE     = wn.synset('have.v.01')
# I: relational ↔ standalone object
_NOUN_RELATION = wn.synset('relation.n.01')
_VERB_RELATE   = wn.synset('relate.v.01')
_VERB_USE      = wn.synset('use.v.01')

class spatialValenceCompute:
    def __init__(self):
        import lmdb
        self._domain_cache = {}
        self._word_cache   = {}   # session-level hot cache: (lemma, pos) → 7-tuple
        self._lemmatizer   = WordNetLemmatizer()
        self._spell        = SpellChecker()
        os.makedirs(_CACHE_DIR, exist_ok=True)
        self._lmdb = lmdb.open(_CACHE_DIR, map_size=128 * 1024 * 1024,
                               subdir=True, max_readers=4)
        nltk.word_tokenize("warmup")
        nltk.pos_tag(["warmup"])
        print(f"initialized {self.__class__.__name__}")

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

    def _living_d(self, s)  -> float:
        return self._wup_axis(s, _NOUN_LIVING,   _NOUN_ARTIFACT, _VERB_GROW,   _VERB_BUILD,  'd')

    def _spatial_e(self, s) -> float:
        return self._wup_axis(s, _NOUN_LOCATION, _NOUN_PROCESS,  _VERB_TRAVEL, _VERB_CHANGE, 'e')

    def _comms_f(self, s)   -> float:
        return self._wup_axis(s, _NOUN_COMMS,    _NOUN_MATTER,   _VERB_SPEAK,  _VERB_CREATE, 'f')

    def _agent_g(self, s)   -> float:
        return self._wup_axis(s, _NOUN_AGENT,    _NOUN_PHENOM,   _VERB_CAUSE,  _VERB_HAPPEN, 'g')

    def _cognit_h(self, s)  -> float:
        return self._wup_axis(s, _NOUN_COGNIT,   _NOUN_ATTRIB,   _VERB_KNOW,   _VERB_HAVE,   'h')

    def _relate_i(self, s)  -> float:
        return self._wup_axis(s, _NOUN_RELATION, _NOUN_OBJECT,   _VERB_RELATE, _VERB_USE,    'i')

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
            if not word.isalpha():
                continue
            wn_pos = self._map_pos(tag)
            if wn_pos is None:
                continue
            lemma     = self._lemmatizer.lemmatize(word, pos=wn_pos)
            cache_key = (lemma, wn_pos)
            if cache_key in self._word_cache:
                wx, wy, wz, wa, wb, wc, wj, wk, wl, wm, wn_, wo, ww = self._word_cache[cache_key]
            else:
                db_key = f"{lemma}\x00{wn_pos}".encode()
                with self._lmdb.begin() as txn:
                    raw = txn.get(db_key) or txn.get(f"{lemma}\x00n".encode())
                if raw and len(raw) == struct.calcsize(_PACK_FMT):
                    vals = struct.unpack(_PACK_FMT, raw)
                else:
                    # LMDB miss — fall back to WordNet computation
                    try:
                        synsets = wn.synsets(lemma, pos=wn_pos)
                        if not synsets and wn_pos == wn.ADJ:
                            synsets = wn.synsets(lemma, pos=wn.ADJ_SAT)
                        if not synsets:
                            for fb_pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADJ_SAT, wn.ADV]:
                                if fb_pos == wn_pos: continue
                                synsets = wn.synsets(self._lemmatizer.lemmatize(word, pos=fb_pos), pos=fb_pos)
                                if synsets: break
                        if not synsets:
                            continue
                        chain = self._hypernym_chain(synsets[0])
                        wx = wy = wz = wa = wb = wc = wj = wk = wl = wm = wn_ = wo = ww = 0.0
                        for ancestor, hop in chain:
                            w   = DECAY ** hop
                            wx += self._domain_x(ancestor) * w
                            wy += self._swn_valence(ancestor) * w
                            wz += ancestor.min_depth() * w
                            wa += self._event_a(ancestor) * w
                            wb += self._objectivity_b(ancestor) * w
                            wc += self._hyponym_c(ancestor) * w
                            wj += self._living_d(ancestor) * w
                            wk += self._spatial_e(ancestor) * w
                            wl += self._comms_f(ancestor) * w
                            wm += self._agent_g(ancestor) * w
                            wn_ += self._cognit_h(ancestor) * w
                            wo += self._relate_i(ancestor) * w
                            ww += w
                        vals = (wx, wy, wz, wa, wb, wc, wj, wk, wl, wm, wn_, wo, ww)
                        with self._lmdb.begin(write=True) as txn:
                            txn.put(db_key, struct.pack(_PACK_FMT, *vals))
                    except Exception:
                        continue
                self._word_cache[cache_key] = vals
                wx, wy, wz, wa, wb, wc, wj, wk, wl, wm, wn_, wo, ww = vals

            x_acc  += wx
            y_acc  += wy
            z_acc  += wz
            a_acc  += wa
            b_acc  += wb
            c_acc  += wc
            w_total += ww

        if w_total == 0.0:
            return (0.0,) * 6

        r = lambda v: round(max(-1.0, min(1.0, v)), 6)
        x = r( x_acc / w_total)
        y = r( y_acc / w_total)
        z = r((z_acc / w_total) / MAX_DEPTH * 2.0 - 1.0)
        a = r( a_acc / w_total)
        b = r((b_acc / w_total) * 2.0 - 1.0)
        c = r((c_acc / w_total) * 2.0 - 1.0)

        return (x, y, z, a, b, c)

    def extractContentWords(self, text: str) -> list:
        tokens   = nltk.word_tokenize(text.lower())
        pos_tags = nltk.pos_tag(tokens)
        words = []
        for word, tag in pos_tags:
            if not word.isalpha():
                continue
            wn_pos = self._map_pos(tag)
            if wn_pos is None:
                continue
            lemma = self._lemmatizer.lemmatize(word, pos=wn_pos)
            if lemma not in words:
                words.append(lemma)
        return words

    def computeWorldValence(self, text: str) -> tuple:
        tokens   = nltk.word_tokenize(text.lower())
        pos_tags = nltk.pos_tag(tokens)
        j_acc = k_acc = l_acc = m_acc = n_acc = o_acc = w_total = 0.0
        r = lambda v: round(max(-1.0, min(1.0, v)), 6)
        for word, tag in pos_tags:
            if not word.isalpha():
                continue
            wn_pos = self._map_pos(tag)
            if wn_pos is None:
                continue
            lemma     = self._lemmatizer.lemmatize(word, pos=wn_pos)
            cache_key = (lemma, wn_pos)
            if cache_key not in self._word_cache:
                self.computeSpatialValence(word)
            if cache_key not in self._word_cache:
                continue
            vals = self._word_cache[cache_key]
            if len(vals) < 13:
                continue
            _, _, _, _, _, _, wj, wk, wl, wm, wn_, wo, ww = vals
            j_acc   += wj;  k_acc += wk;  l_acc += wl
            m_acc   += wm;  n_acc += wn_; o_acc += wo
            w_total += ww
        if w_total == 0.0:
            return (0.0,) * 6
        return (
            r(j_acc / w_total),   # D: living vs artifact
            r(k_acc / w_total),   # E: spatial vs process
            r(l_acc / w_total),   # F: communicative vs material
            r(m_acc / w_total),   # G: agent vs phenomenon
            r(n_acc / w_total),   # H: cognitive vs attribute
            r(o_acc / w_total),   # I: relational vs object
        )


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
