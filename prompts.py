"""Prompts, as text: the v6.1 model prompts and the judge prompts.

Model prompts (what the audio model is asked):

    stage1_messages()                     the source inventory, once per scene
    stage2_messages(source_type, name)    one card per merged Stage-1 source

They carry no audio. Attach the scene audio to the user turn in whatever layout your model
expects; the benchmark sends the full 10-second mixture with every call.

Judge prompts (what the text judge is asked), filled by :func:`judge_messages`:

    PAIR_SIMILARITY   Stage 1: does a predicted source name the reference class?
    TAG_SET           Stage 2: tag-by-tag alignment counts for a tag field
    CAPTION_RUBRIC    Stage 2: four-criterion rubric for a caption

Changing any text here changes what the numbers mean; :func:`fingerprint` records which
wording a result was produced with.
"""

from __future__ import annotations

import hashlib
import string

from .format import SOURCE_TYPES

VERSION = "v6.1"

STAGE1_SYSTEM = """You are an expert audio scene analyst. You identify the distinct sound sources present in an audio recording and when each one is audible."""

STAGE1_USER = """Listen to this 10-second audio scene. It is a mixture that may contain speech, music, and sound effects, possibly overlapping.

List every DISTINCT sound source you can hear, and for each one the time span(s) during which it is audible. Rules:
- Each source has a type: speech, music, or sound effect.
- Sounds that belong to one semantic source count as ONE source (e.g. an engine and rolling wheels together are one source: a car). A constant featureless background noise bed is NOT a source; ignore it. But sustained environmental sound that characterizes a place (e.g. room tone, ventilation hum) IS a sound-effect source.
- All speech in the scene is one source; all music in the scene is one source; each distinct sound-effect source is its own entry.
- Background music counts as a music source even when it is quiet or under speech.
- Time spans are in seconds from the start of the clip (0.0 to 10.0). A source that occurs several separate times gets several spans, separated by semicolons.

Reply with ONLY a numbered list, one source per line, in this exact format:
1. <type> | <brief name of the source> | <start>-<end>; <start>-<end>
No other text."""

STAGE2_SYSTEM = {
    "speech": """You are an expert speech analyst and audio engineer with a precise ear. You describe ONLY the requested speech inside a mixed audio scene, judging with your ears: intelligibility, perceived audio quality, emotion and manner of speech are DISTINCT fields - never repeat one in another. What is said belongs ONLY in the transcript field (and, briefly, in the descriptive caption) - never in the tag fields.""",
    "music": """You are an expert music analyst, musician and audio engineer with a precise ear. You describe ONLY the requested music source inside a mixed audio scene, judging with your ears: its mood, instruments, musical descriptors and perceived audio quality are DISTINCT fields - never repeat one in another.""",
    "sfx": """You are an expert sound designer describing one sound-effect source inside a mixed audio scene by listening. Base every field on what you hear. The sounding object, its action, temporal behavior (how it occurs and evolves over the clip), plurality (how many distinct sources sound simultaneously), sonic character (how it strikes the ear) and perceived audio quality (channel/production, without naming a cause) are DISTINCT fields - never repeat one in another.""",
}

STAGE2_USER = {
    "speech": """Listen to this 10-second audio scene (a mixture that may contain speech, music and sound effects). Focus ONLY on this source: SPEECH — {SOURCE}. Ignore all other sounds in the mixture.

Fill a metadata card for the SPEECH with exactly the following fields. Judge with your ears. Where a field has a fixed vocabulary, do NOT use any word that is not in it. Keep every field distinct - do not repeat the same information across fields.

- "segments": the time span(s) where this source is audible, as [["MM:SS.mmm", "MM:SS.mmm"], ...]
- "loudness": loudness per segment as [[<LUFS value>, "<label>"], ...], label from: Extremely quiet | Very quiet | Quiet | Medium | Loud | Very loud
- "frequency_profile": energy distribution of this source per segment across ten bands (125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000 Hz), as [{"bands_label": {"125": "<level>", "250": "<level>", "500": "<level>", "1000": "<level>", "2000": "<level>", "3000": "<level>", "4000": "<level>", "6000": "<level>", "8000": "<level>", "10000": "<level>"}}, ...] with levels from: very-low | low | medium-low | medium | medium-high | high | very-high
- "median_f0_hz": the median fundamental frequency (voice pitch) per speech segment, as [[start, end, Hz], ...] with start/end in seconds
- "perceived_audio_quality": tags from this fixed vocabulary, grouped by axis - within an axis pick AT MOST ONE tag (Frequency balance: up to two if not contradictory); skip an axis if nothing clearly applies. Fidelity: low-fidelity | clear | high-fidelity. Space: dry | room reverb | reverb-heavy. Frequency balance: bass-heavy | muffled | thin | harsh. Drive: distorted. Noise: low background noise | high background noise. Do NOT tag the spoken content, the emotion, or intelligibility. Most prominent first, as a list
- "manner": tags for the MANNER of speech - delivery only: pace, rhythm, articulation, vocal effort, tone of delivery (e.g. gruff, whispered, conversational, animated, hesitant, monotonous, breathy, articulate). Do NOT use pitch words (low-pitched, high-pitched, deep - pitch belongs to median_f0_hz), recording/channel words (clear, muffled, distorted, thin, reverberant - that is perceived_audio_quality), or emotion words (calm, angry, happy, sad, stressed, neutral - that is emotion). 2-5 tags, most prominent first, as a list
- "intelligibility": how understandable the speech is and how clean its channel is, as {"words": [..]}. Exactly one of: highly intelligible | mostly intelligible | degraded intelligibility; optionally followed by one of: clean channel | poor channel quality (leave the channel word out when neither clearly applies)
- "emotion": the speaker's emotional state, as {"words": [..]} with exactly one of: neutral | excited/happy | angry/stressed | sad/subdued | calm
- "transcript": what is said, as {"language": "<ISO 639-1 code, e.g. en>", "segments": [{"start": <seconds>, "end": <seconds>, "text": "<verbatim words>"}, ...]}, one segment per utterance. If you cannot make out the words, use an empty "segments" list - never invent words
- "short_caption": 2-4 words naming the most salient character of the speech, without gender - use pitch when useful. E.g. "Angry deep voice shouting"
- "descriptive_caption": one natural sentence describing the speech: what it is about (quoted only when brief), how it sounds (including pitch as deep/low/mid/high voice), and how it unfolds over the clip - without labeling speaker gender (no man/woman/male/female/boy/girl)

Reply with ONLY a single JSON object containing exactly these keys, in this order. No sentences outside the JSON, no explanation.""",
    "music": """Listen to this 10-second audio scene (a mixture that may contain speech, music and sound effects). Focus ONLY on this source: MUSIC — {SOURCE}. Ignore all other sounds in the mixture.

Fill a metadata card for this music with exactly the following fields. Judge with your ears. Where a field has a fixed vocabulary, do NOT use any word that is not in it. Keep every field distinct - do not repeat the same information across fields.

- "segments": the time span(s) where this source is audible, as [["MM:SS.mmm", "MM:SS.mmm"], ...]
- "loudness": loudness per segment as [[<LUFS value>, "<label>"], ...], label from: Extremely quiet | Very quiet | Quiet | Medium | Loud | Very loud
- "frequency_profile": energy distribution of this source per segment across ten bands (125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000 Hz), as [{"bands_label": {"125": "<level>", "250": "<level>", "500": "<level>", "1000": "<level>", "2000": "<level>", "3000": "<level>", "4000": "<level>", "6000": "<level>", "8000": "<level>", "10000": "<level>"}}, ...] with levels from: very-low | low | medium-low | medium | medium-high | high | very-high
- "perceived_audio_quality": tags from this fixed vocabulary, grouped by axis - within an axis pick AT MOST ONE tag (Frequency balance: up to two if not contradictory); skip an axis if nothing clearly applies. Fidelity: low-fidelity | clear | high-fidelity. Space: dry | room reverb | reverb-heavy. Frequency balance: bass-heavy | muffled | thin | harsh. Drive: distorted. Noise: low background noise | high background noise. Most prominent first, as a list
- "music_descriptors": 2-5 tags for the character/texture/feel of the music (no genre, no mood, no instruments, no production quality) from this fixed vocabulary: rhythmic | layered | repetitive | sparse | sustained | dissonant | syncopated | percussive | minimalist | complex | arpeggiated | synthetic | glitchy | gritty | delicate | plucked | raw | melodic | warm | groovy | bright | abrupt | harmonic | high-pitched | rapid | intricate | punchy | orchestral | unpolished | detuned | lively | looping | evolving | static | ornamental | frenetic. Most prominent first, as a list
- "instruments": 1-6 instruments you can actually hear playing, most prominent first, from this fixed vocabulary: piano | electric piano | organ | harpsichord | synthesizer | accordion | guitar | electric guitar | acoustic guitar | bass guitar | ukulele | banjo | mandolin | lute | sitar | harp | violin | viola | cello | double bass | strings | trumpet | trombone | tuba | french horn | brass | flute | piccolo | clarinet | oboe | bassoon | saxophone | bagpipes | harmonica | woodwinds | drums | drum machine | percussion | cymbals | timpani | congas | tabla | darbuka | timbales | shaker | triangle | gong | handpan | glockenspiel | xylophone | marimba | vibraphone | bell | chimes | vocals | choir | whistling | beatboxing. Name the MOST SPECIFIC instrument you can confidently identify; family terms (guitar, strings, brass, woodwinds, percussion, choir) ONLY for whole sections or when the specific instrument cannot be determined; never both a family term and one of its members for the same sound. For guitars, commit to "acoustic guitar" or "electric guitar" whenever you can tell. "drum machine" = programmed/electronic beats; "drums" = an acoustic drum kit. As a list
- "genre": the musical genre label(s), using Free Music Archive genre names (e.g. Rock, Punk, Folk, Ambient Electronic, Hip-Hop, Experimental Pop), most specific first, as a list
- "mood": 1-2 categories, dominant first, from this fixed vocabulary (music-emotion categories): amazement (wonder and happiness) | solemnity (transcendence, inspiration, thrills) | tenderness (sensuality, affect, love) | nostalgia (dreamy, melancholic, sentimental) | calmness (relaxation, serenity, meditativeness) | power (strong, heroic, triumphant, energetic) | joyful activation (like dancing, bouncy, animated, amused) | tension (nervous, impatient, irritated, eerie, ominous, dread or menace) | sadness (depressed, sorrowful). As a list
- "tempo": the tempo as [<BPM>, "<label>"] with label from: very slow | slow | medium | fast | very fast; or the string "no tempo" if there is no discernible beat
- "short_caption": 2-4 words in the form [Adjective] [Noun] [Verb] - the adjective from the genre, mood, or tempo; the noun the music, melody, or its leading instrument; the verb what it does. E.g. "Rock music fades"; "Happy melody plays"; "Driving indie-rock blares"
- "descriptive_caption": one natural sentence describing the music - mood, genre, instruments, tempo feel, loudness character, recording character, and how it unfolds over the clip. Never echo numbers or tier labels; translate them into listener language

Reply with ONLY a single JSON object containing exactly these keys, in this order. No sentences outside the JSON, no explanation.""",
    "sfx": """Listen to this 10-second audio scene (a mixture that may contain speech, music and sound effects). Focus ONLY on this source: SOUND EFFECT — {SOURCE}. Ignore all other sounds in the mixture.

Fill a metadata card for this sound with exactly the following fields. Judge with your ears. Where a field has a fixed vocabulary, do NOT use any word that is not in it. Keep every field distinct - do not repeat the same information across fields.

- "segments": the time span(s) where this source is audible, as [["MM:SS.mmm", "MM:SS.mmm"], ...]
- "loudness": loudness per segment as [[<LUFS value>, "<label>"], ...], label from: Extremely quiet | Very quiet | Quiet | Medium | Loud | Very loud
- "frequency_profile": energy distribution of this source per segment across ten bands (125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000 Hz), as [{"bands_label": {"125": "<level>", "250": "<level>", "500": "<level>", "1000": "<level>", "2000": "<level>", "3000": "<level>", "4000": "<level>", "6000": "<level>", "8000": "<level>", "10000": "<level>"}}, ...] with levels from: very-low | low | medium-low | medium | medium-high | high | very-high
- "sounding_object": noun(s) producing the sound - generic names, no brand/model/place/personal names. E.g. dog, dogs, microwave, plate, wooden door. As a list
- "sounding_action": verb(s) for what the source does. E.g. barking, beeping, breaking. As a list
- "temporal_descriptors": the TEMPORAL BEHAVIOR of the sound over the clip, from this fixed vocabulary grouped by axis - pick EXACTLY ONE Occurrence tag; from the other axes pick AT MOST ONE tag each, only if clearly audible. Occurrence: single event | intermittent | periodic | rhythmic | continuous. Loudness dynamics: fade in | fade out | sustained | sustained with occasional bursts | impact/abrupt. Timbre dynamics: rising | falling | oscillating. A sound recurring over time is a temporal pattern, not multiple sources - describe the recurrence here. Never count repetitions. As a list
- "plurality": how many distinct sources sound SIMULTANEOUSLY - exactly one of none | one | multiple. One source that starts and stops repeatedly is still one source (that belongs to temporal behavior). Use none for a mass or diffuse phenomenon with no countable source (water, rain, wind, room ambience)
- "perceived_audio_quality": tags from this fixed vocabulary, grouped by axis - within an axis pick AT MOST ONE tag (Frequency balance: up to two if not contradictory); skip an axis if nothing clearly applies. Fidelity: low-fidelity | clear | high-fidelity. Space: dry | room reverb | reverb-heavy. Frequency balance: bass-heavy | muffled | thin | harsh. Drive: distorted. Noise: low background noise | high background noise. NOT what the sound is, NOT its mood. Most prominent first, as a list
- "sound_descriptors": tags describing ONLY the SONIC CHARACTER of the sound itself - plain adjectives or adverbs, e.g. harsh, hollow, sharp, rumbling, hurried, gentle, metallic, scraping. Do not use mood words (e.g. eerie, cheerful) or channel/production words (e.g. muffled, reverberant) - only how the sound itself strikes the ear. 2-5 tags, most prominent first, as a list
- "short_caption": 2-4 words. E.g. "Dog barks angrily"; "Hurried, loud footsteps"
- "descriptive_caption": one natural sentence. E.g. "A group of angry dogs bark repeatedly in a cathedral." Ground every word in what you hear; no brand or model names, no place names

Reply with ONLY a single JSON object containing exactly these keys, in this order. No sentences outside the JSON, no explanation.""",
}

PAIR_SIMILARITY = (
    """You rate how strongly two labels refer to the same sound source in a recording.

The two labels come from different vocabularies. One is a sound-effects library category (like "Room-Tone", "Farm", "Hand"); the other is free text written by a model describing what it heard (like "room ambience", "pig grunting", "footsteps").

Give a similarity between 0.0 and 1.0:
- 1.0  the same source, however differently worded ("Room-Tone" and "room ambience")
- 0.7  a category and an instance of it ("Farm" and "pig grunting"; "Hand" and "clapping")
- 0.3  related but not the same source ("Door" and "footsteps in a hallway")
- 0.0  unrelated sources ("Arrow" and "dog barking")

Judge the sound source, not the vocabulary: matching words are not evidence and differing words are not disqualifying.

Respond with ONLY a single JSON object in exactly this format, with no markdown fences and no text before or after it:
{"reasoning": "<one short sentence>", "similarity": <float between 0.0 and 1.0>}""",
    """Library category: "$reference"
Model label: "$prediction"
How strongly do these name the same sound source?""",
)

TAG_SET = (
    """You are an expert audio-annotation evaluator. You will be given a REFERENCE list of audio tags and a PREDICTED list of audio tags for the same sound recording. Your job is to align the two lists tag by tag and report the counts. You do not compute the final score; the harness does.

Grade every match at one of two strengths:
- EXACT: the two tags name the same concept at the same level of specificity. Spelling variants, plurals, hyphenation and true synonyms are exact ("lo-fi" / "low-fidelity", "car keys" / "keys", "excited" / "excited-happy").
- PARTIAL: the tags are related but not equivalent. This covers a broader category standing in for a specific answer ("percussion" for "handpan", "animal" for "dog"), a narrower term standing in for a general one, and loosely associated or overlapping descriptions ("crackling" for "glitching"). When in doubt between EXACT and PARTIAL, choose PARTIAL.

A tag matches at most one tag on the other side. Anything left over is a miss (reference side) or a spurious extra (prediction side); both are counted by being absent from the tallies.

Report:
- ref_exact:   how many REFERENCE tags have an EXACT match in the prediction
- ref_partial: how many REFERENCE tags have only a PARTIAL match
- pred_exact:  how many PREDICTED tags are an EXACT match for a reference tag
- pred_partial: how many PREDICTED tags are only a PARTIAL match
- holistic:    your own overall 0.0-1.0 impression of the match, ignoring the tallies

The exact counts on the two sides are normally equal; the partial counts may differ when several tags on one side collapse onto one on the other. Never count a tag twice, and never report a count larger than the length of that side's list.

Respond with ONLY a single JSON object in exactly this format, with no markdown fences and no text before or after it:
{"reasoning": "<one or two sentences naming which tags matched exactly, which only partially, and which were missed or spurious>", "ref_exact": <int>, "ref_partial": <int>, "pred_exact": <int>, "pred_partial": <int>, "holistic": <float between 0.0 and 1.0>}""",
    """Tag type: $hint
Reference tags: $reference
Predicted tags: $prediction""",
)

CAPTION_RUBRIC = (
    """You are an expert evaluator of audio understanding. You will be given a REFERENCE text and a PREDICTED text, both of which are a $hint for the same sound recording.

Score the prediction against the reference on four criteria. Each criterion is a float from 0.0 (completely fails) to 1.0 (perfect):
- correctness: How factually accurate is the prediction compared to the reference?
- relevance: How well does the prediction address the same aspect of the sound that the reference describes?
- completeness: Does the prediction cover all important aspects mentioned in the reference?
- clarity: How unambiguously does the prediction identify the sound it is describing? Judge what a reader learns about THIS recording -- the source, and what it is doing -- not the fluency of the English. A grammatical sentence that stays generic ("a sound is heard", "audio of an event", "various noises") is not clear, however well written: score it low. A prediction that names the source and its action without hedging or contradiction is clear, even if terse or ungrammatical. Clarity is about grip on the sound, not prose quality, so it is capped by how much the prediction commits to.

If the prediction is empty, missing, or the literal text "None"/"null", every criterion is 0.0 -- there is nothing to be correct, relevant, complete or clear about.

First write brief reasoning comparing the prediction to the reference, then give the four scores.

Respond with ONLY a single JSON object in exactly this format, with no markdown fences and no text before or after it:
{"reasoning": "<two or three sentences>", "correctness": <float 0.0-1.0>, "relevance": <float 0.0-1.0>, "completeness": <float 0.0-1.0>, "clarity": <float 0.0-1.0>}""",
    """Reference: "$reference"
Prediction: "$prediction\"""",
)



def stage1_messages() -> list[dict[str, str]]:
    """System and user turns for the Stage-1 inventory."""
    return [{"role": "system", "content": STAGE1_SYSTEM},
            {"role": "user", "content": STAGE1_USER}]


def stage2_messages(source_type: str, source_name: str) -> list[dict[str, str]]:
    """System and user turns for one Stage-2 card. ``source_type`` is speech, music or sound
    effect; ``source_name`` is the name from the merged inventory ("speech"/"music" for those)."""
    modality = SOURCE_TYPES[source_type.strip().lower()]
    return [{"role": "system", "content": STAGE2_SYSTEM[modality]},
            {"role": "user", "content": STAGE2_USER[modality].replace("{SOURCE}", source_name)}]


def judge_messages(prompt: tuple[str, str], **fields: object) -> list[dict[str, str]]:
    """Fill a judge prompt's ``$placeholders``; a list is joined with ", "."""
    values = {key: ", ".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)
              for key, value in fields.items()}
    return [{"role": role, "content": string.Template(text).substitute(values)}
            for role, text in zip(("system", "user"), prompt)]


def fingerprint() -> str:
    """Short hash over every prompt in this module."""
    digest = hashlib.sha256()
    texts = [STAGE1_SYSTEM, STAGE1_USER, *STAGE2_SYSTEM.values(), *STAGE2_USER.values(),
             *PAIR_SIMILARITY, *TAG_SET, *CAPTION_RUBRIC]
    for text in texts:
        digest.update(text.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:12]
