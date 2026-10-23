"""ABC notation to Standard MIDI File for the SheetSage2 nodes.

Only the notation that ``comfy.audio_encoders.sheetsage2_abc`` writes has to be
understood: header fields, per-voice ``M:``/``K:`` blocks, quoted chord symbols,
``%`` structure labels, bars, ``Z`` multi-measure rests, notes with
accidentals/octave marks, durations counted in ``L`` units and ``-`` ties.

Tick positions are accumulated as exact fractions of a quarter note and only
rounded once, when the file is encoded, so long songs cannot drift.
"""

import math
import re
import struct
from fractions import Fraction

from comfy.audio_encoders.sheetsage2_abc import get_key_accidentals

LETTERS = "CDEFGAB"
LETTER_PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
WRITTEN_ACCIDENTALS = {"^": 1, "^^": 2, "_": -1, "__": -2, "=": 0}
DEFAULT_VELOCITY = 96
NOTE_OFF_VELOCITY = 64
BASE_RESOLUTION = 480

_FIELD_RE = re.compile(r"^(?P<name>[A-Za-z])\s*:(?P<value>.*)$")
_FRACTION_RE = re.compile(r"^(?P<num>\d+)\s*/\s*(?P<den>\d+)$")
_TOKEN_RE = re.compile(
    r'"(?P<chord>[^"]*)"'
    r"|\[K:(?P<key>[^\]]+)\]"
    r"|(?P<multirest>Z)(?P<restcount>\d*)"
    r"|(?P<bar>[|:\]]+)"
    r"|(?P<accidentals>[_=^]{0,2})(?P<letter>[A-Ga-gz])(?P<octave>[,']*)(?P<units>\d*)(?P<tie>-?)"
)


class AbcParseError(ValueError):
    """Raised when the ABC text holds notation this writer does not cover."""


def _fraction(text):
    match = _FRACTION_RE.match(text.strip())
    if not match:
        raise AbcParseError(f"expected a fraction like 1/16, got {text.strip()!r}")
    return Fraction(int(match.group("num")), int(match.group("den")))


def _tempo_bpm(value):
    """Beats per minute of a quarter note from a ``Q:`` field."""
    text = value.strip()
    beat = Fraction(1, 4)
    if "=" in text:
        beat_text, text = text.split("=", 1)
        beat = _fraction(beat_text)
    return float(text.strip()) * float(beat * 4)


def _pitch(accidentals, letter, octave_marks, key_accidentals, measure_accidentals):
    index = LETTERS.index(letter.upper())
    if accidentals:
        try:
            value = WRITTEN_ACCIDENTALS[accidentals]
        except KeyError as exc:
            raise AbcParseError(f"unsupported accidental {accidentals!r} before {letter!r}") from exc
    else:
        value = measure_accidentals.get(index, key_accidentals[index])
    measure_accidentals[index] = value
    octave = 4 if letter.isupper() else 5
    for mark in octave_marks:
        octave += 1 if mark == "'" else -1
    return 12 * (octave + 1) + LETTER_PITCH_CLASSES[letter.upper()] + value


def _parse(text):
    """Split ABC text into voices, tempo changes, meters and markers.

    Each voice keeps its own clock in quarter notes because writers interleave
    voices block by block; ``M:``/``K:``/``L:`` values are global from the
    point they are declared.
    """
    voices = {}
    order = []
    tempo = []
    meters = []
    markers = []
    unit = Fraction(1, 8)
    meter = (4, 4)
    key_accidentals = get_key_accidentals("C")
    current = None
    pending_chord = None

    def clock():
        return voices[current]["clock"] if current in voices else Fraction(0)

    def add_voice(name):
        if name not in voices:
            voices[name] = {"clock": Fraction(0), "events": [], "accidentals": {}, "tie": None}
            order.append(name)
        return voices[name]

    def add_marker(tick, label):
        if label and (not markers or markers[-1] != (tick, label)):
            markers.append((tick, label))

    def add_meter(tick, value):
        if not meters or meters[-1][0] != tick or tuple(meters[-1][1:]) != value:
            meters.append((tick, *value))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("%"):
            add_marker(clock(), line[1:].strip())
            continue

        field = _FIELD_RE.match(line)
        if field:
            name, value = field.group("name").upper(), field.group("value").strip()
            if name == "V":
                current = value.split()[0] if value.split() else ""
                if not current:
                    raise AbcParseError(f"empty V: field in line {raw_line!r}")
                add_voice(current)
            elif name == "M":
                if value in ("C", "C|"):
                    meter = (4, 4) if value == "C" else (2, 2)
                else:
                    # 4/4 is a meter, not a fraction: never reduce it to 1/1.
                    match = _FRACTION_RE.match(value)
                    if not match:
                        raise AbcParseError(f"expected a meter like 4/4, got {value!r}")
                    meter = (int(match.group("num")), int(match.group("den")))
                add_meter(clock(), meter)
            elif name == "L":
                unit = _fraction(value)
            elif name == "Q":
                bpm = _tempo_bpm(value)
                if not tempo or tempo[-1][0] != clock() or tempo[-1][1] != bpm:
                    tempo.append((clock(), bpm))
            elif name == "K":
                try:
                    key_accidentals = get_key_accidentals(value)
                except ValueError as exc:
                    raise AbcParseError(f"unsupported K: field {value!r}") from exc
                for state in voices.values():
                    state["accidentals"].clear()
            continue

        state = voices[current] if current in voices else None
        if state is None:
            raise AbcParseError("music appears before any V: field")
        position = 0
        for token in _TOKEN_RE.finditer(line):
            gap = line[position:token.start()]
            if gap.strip():
                raise AbcParseError(f"cannot parse {gap.strip()!r} in {raw_line!r}")
            position = token.end()
            tick = state["clock"]

            if token.group("chord") is not None:
                pending_chord = token.group("chord")
                continue

            if token.group("key") is not None:
                try:
                    key_accidentals = get_key_accidentals(token.group("key").strip())
                except ValueError as exc:
                    raise AbcParseError(f"unsupported key {token.group('key')!r}") from exc
                for voice_state in voices.values():
                    voice_state["accidentals"].clear()
                continue

            if token.group("bar"):
                state["accidentals"].clear()
                continue

            if pending_chord is not None:
                add_marker(tick, pending_chord)
                pending_chord = None

            if token.group("multirest"):
                count = int(token.group("restcount") or 1)
                state["clock"] += count * Fraction(4 * meter[0], meter[1])
                state["tie"] = None
                continue

            length = Fraction(int(token.group("units") or 1)) * unit * 4
            letter = token.group("letter")
            if letter == "z":
                state["clock"] += length
                state["tie"] = None
                continue

            pitch = _pitch(token.group("accidentals"), letter, token.group("octave"),
                           key_accidentals, state["accidentals"])
            tied = state["tie"]
            if tied is not None and tied[0] == pitch:
                start, duration, _ = state["events"][tied[1]]
                state["events"][tied[1]] = (start, duration + length, pitch)
                index = tied[1]
            else:
                state["events"].append((tick, length, pitch))
                index = len(state["events"]) - 1
            state["tie"] = (pitch, index) if token.group("tie") else None
            state["clock"] = tick + length

        tail = line[position:]
        if tail.strip():
            raise AbcParseError(f"cannot parse {tail.strip()!r} in {raw_line!r}")

    return {"voices": [(name, voices[name]["events"]) for name in order],
            "tempo": tempo, "meters": meters, "markers": markers}


def _resolution(score):
    """Ticks per quarter note that represents every event exactly."""
    denominators = {1}
    for _, events in score["voices"]:
        for start, duration, _ in events:
            denominators.add(start.denominator)
            denominators.add(duration.denominator)
    return math.lcm(BASE_RESOLUTION, *denominators)


def _ticks(quarters, resolution):
    return int(round(quarters * resolution))


def _variable_length(value):
    if value < 0:
        raise AbcParseError(f"negative MIDI delta {value}")
    data = bytearray([value & 0x7F])
    value >>= 7
    while value:
        data.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(data))


def _track(events):
    events = sorted(events, key=lambda event: (event[0], event[1]))
    body = bytearray()
    previous = 0
    for tick, _, payload in events:
        body += _variable_length(tick - previous) + payload
        previous = tick
    body += b"\x00\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(body)) + bytes(body)


def _text_event(kind, text):
    data = text.encode("utf-8")
    return bytes([0xFF, kind]) + _variable_length(len(data)) + data


def _time_signature(numerator, denominator):
    power = denominator.bit_length() - 1
    # MIDI stores the meter as a power of two; 6/8 and 3/4 encode, 3/5 cannot.
    if 2 ** power != denominator:
        return None
    return b"\xff\x58\x04" + bytes([numerator, power, 24, 8])


def abc_to_midi(text, velocity=DEFAULT_VELOCITY):
    """Return the bytes of a type 1 MIDI file: conductor track plus one track per voice."""
    score = _parse(text)
    resolution = _resolution(score)

    conductor = []
    for tick, numerator, denominator in score["meters"]:
        payload = _time_signature(numerator, denominator)
        if payload is not None:
            conductor.append((_ticks(tick, resolution), 0, payload))
    for tick, bpm in score["tempo"]:
        micros = int(round(60_000_000 / bpm))
        conductor.append((_ticks(tick, resolution), 1, b"\xff\x51\x03" + micros.to_bytes(3, "big")))
    for tick, label in score["markers"]:
        conductor.append((_ticks(tick, resolution), 2, _text_event(0x06, label)))

    tracks = [_track(conductor)]
    for channel, (_, events) in enumerate(score["voices"]):
        status = channel % 16
        payload = []
        for start, duration, pitch in events:
            begin = _ticks(start, resolution)
            end = max(begin + 1, _ticks(start + duration, resolution))
            payload.append((begin, 2, bytes([0x90 | status, pitch, velocity])))
            payload.append((end, 1, bytes([0x80 | status, pitch, NOTE_OFF_VELOCITY])))
        tracks.append(_track(payload))

    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), resolution)
    return header + b"".join(tracks)
