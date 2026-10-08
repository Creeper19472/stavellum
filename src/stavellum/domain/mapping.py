"""Conservative name suggestions; voice identity is never inferred from colors."""

import re
import unicodedata
from collections import Counter
from dataclasses import replace

from .models import PartMapping, ProjectIR, TrackInfo

_INSTRUMENT_RULES = (
    ("double_bass", r"double\s*bass|contrabass|低音提琴|低音大提琴|コントラバス"),
    ("violin", r"\bviolins?\b|\bvln\b|小提琴|ヴァイオリン"),
    ("viola", r"\bviolas?\b|\bvla\b|中提琴"),
    ("cello", r"\bcellos?\b|\bcelli\b|\bvioloncell[oi]\b|\bvc\b|大提琴"),
    ("piano", r"\bpianos?\b|\bfl\s*keys\b|钢琴|鋼琴"),
    ("harp", r"\bharps?\b|竖琴"),
    ("flute", r"\bflutes?\b|\bpiccolo\b|长笛|短笛"),
    ("oboe", r"\boboes?\b|\benglish\s*horn\b|双簧管"),
    ("clarinet", r"\bclarinets?\b|单簧管|黑管"),
    ("bassoon", r"\bbassoons?\b|大管|巴松"),
    ("trumpet", r"\btrumpets?\b|小号"),
    ("trombone", r"\btrombones?\b|长号"),
    ("horn", r"\b(?:french\s*)?horns?\b|圆号"),
    ("tuba", r"\btubas?\b|大号"),
    ("guitar", r"\bguitars?\b|吉他"),
    ("bass", r"\bbass\b|贝斯|貝斯"),
    ("percussion", r"\b(?:drums?|percussion|timpani|fpc|kick|snare|hi[ -]?hat)\b|打击乐|鼓"),
    ("bell", r"\b(?:bells?|celest[ae]|glockenspiel|tubular\s*bells)\b|钟|鈴"),
)
_ARTICULATION = re.compile(r"\b(?:pizz(?:icato)?|spiccato|staccato|arco|normal|sustain(?:ed)?|legato)\b|拨奏|撥奏|弓奏|普通", re.I)
_STRINGS = {"violin", "viola", "cello", "double_bass"}
_REGULAR_TECHNIQUES = {"arco", "normal", "sustain", "sustained", "legato", "普通", "弓奏", "常规"}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def identify_instrument(track: TrackInfo) -> str:
    name = re.sub(r"(?<=[a-z])(?=[0-9])", " ", _normalize(track.name))
    for instrument, rule in _INSTRUMENT_RULES:
        if re.search(rule, name):
            return instrument
    # Only a known instrument or GM program supplies an additional hint. A
    # Kontakt/FLEX/Wrapper instance says nothing about its loaded patch.
    plugin = _normalize(track.plugin)
    if "general midi:" in plugin or plugin.strip() == "fl keys":
        for instrument, rule in _INSTRUMENT_RULES:
            if re.search(rule, plugin):
                return instrument
    return "unknown"


def _voice_identity(name: str) -> str:
    value = _ARTICULATION.sub("", _normalize(name))
    value = re.sub(r"[\s_\-().\[\]]+", " ", value)
    return value.strip()


def _technique(name: str) -> str:
    value = _normalize(name)
    if re.search(r"pizz(?:icato)?|拨奏|撥奏", value):
        return "pizz."
    separate_bow = re.search(r"\b(?:spiccato|staccato)\b", value)
    return separate_bow.group() if separate_bow else "arco"


def activity_color(project: ProjectIR, mapping: PartMapping) -> str:
    """Use the first regular source's Rack color within this logical part."""
    tracks = {track.track_id: track for track in project.tracks}
    sources = [tracks[track_id] for track_id in mapping.track_ids if track_id in tracks]
    if not sources:
        return "#ffffff"
    selected = sources[0]
    for track in sources:
        explicit = _normalize(mapping.articulations.get(track.track_id, "")).strip()
        technique = explicit.rstrip(".") if explicit else _technique(track.name)
        if technique in _REGULAR_TECHNIQUES:
            selected = track
            break
    return selected.color.lower() if re.fullmatch(r"#[0-9a-fA-F]{6}", selected.color) else "#ffffff"


def suggest_mappings(project: ProjectIR) -> list[PartMapping]:
    active = {note.track_id for note in project.notes}
    groups: dict[tuple[str, str], PartMapping] = {}
    identity_counts = Counter((identify_instrument(track), _voice_identity(track.name),
                               _technique(track.name)) for track in project.tracks)
    ambiguous_groups = {(instrument, identity) for (instrument, identity, _), count in identity_counts.items() if count > 1}
    mappings: list[PartMapping] = []
    for track in project.tracks:
        instrument = identify_instrument(track)
        identity = _voice_identity(track.name)
        # Remove technique labels only. Do not remove I/II, numeric suffixes,
        # family/section labels, or MIDI channel identities.
        group_key = (instrument, identity)
        articulation = _technique(track.name)
        existing = groups.get(group_key) if instrument in _STRINGS and group_key not in ambiguous_groups else None
        previous_tracks = [candidate for candidate in project.tracks
                           if existing is not None and candidate.track_id in existing.track_ids]
        has_explicit_technique = bool(_ARTICULATION.search(track.name)) or any(
            _ARTICULATION.search(candidate.name) for candidate in previous_tracks)
        different_technique = existing is not None and articulation not in existing.articulations.values()
        if existing is not None and has_explicit_technique and different_technique:
            existing.track_ids.append(track.track_id)
            existing.articulations[track.track_id] = articulation
            existing.enabled |= track.track_id in active
            continue
        if existing is not None:
            # Two ordinary voices with the same display name cannot identify
            # which voice a later Pizz channel belongs to. Keep all separate.
            ambiguous_groups.add(group_key)
        display = _ARTICULATION.sub("", track.name).strip(" -_().[]") or track.name
        mapping = PartMapping(
            part_id=f"part-{track.track_id}", name=display, track_ids=[track.track_id],
            instrument=instrument, icon="" if instrument == "unknown" else instrument,
            clef="percussion" if instrument == "percussion" else "auto",
            grand_staff=instrument == "piano", percussion=instrument == "percussion",
            enabled=track.track_id in active)
        if instrument in _STRINGS:
            mapping.articulations[track.track_id] = articulation
            if group_key not in ambiguous_groups:
                groups[group_key] = mapping
        mappings.append(mapping)
    return mappings


def split_track_by_midi_channel(project: ProjectIR, track_id: str) -> list[str]:
    """Explicit user action: split a Rack track by its retained note-channel field.

    Caller replaces the original mapping with mappings of returned track IDs.
    This mutates only the in-memory/project JSON representation, never the FLP.
    """
    track = next((track for track in project.tracks if track.track_id == track_id), None)
    if track is None:
        raise ValueError("找不到需要拆分的音轨。")
    channels = sorted({note.midi_channel for note in project.notes if note.track_id == track_id})
    if len(channels) < 2:
        return [track_id]
    identifiers = {channel: f"{track_id}-midi-{channel}" for channel in channels}
    index = project.tracks.index(track)
    project.tracks[index:index + 1] = [replace(track, track_id=identifiers[channel],
        name=f"{track.name} · MIDI {channel + 1}", midi_channel=channel) for channel in channels]
    routes = []
    for route in project.volume_routes:
        if route.track_id == track_id:
            routes.extend(replace(route, track_id=identifiers[channel],
                                  control_ids=list(route.control_ids),
                                  initial_values=dict(route.initial_values)) for channel in channels)
        else:
            routes.append(route)
    project.volume_routes = routes
    for note in project.notes:
        if note.track_id == track_id:
            note.track_id = identifiers[note.midi_channel]
    for diagnostic in project.diagnostics:
        if diagnostic.track_id == track_id:
            diagnostic.track_id = ""
    return list(identifiers.values())
