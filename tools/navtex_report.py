#!/usr/bin/env python3
# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Message-quality report for a NAVTEX decoder database
====================================================

Reads the `navtex_log` table written by navtex-cli and reports how well the
NAVTEX messages in it were received: how many have an intact header
(ZCZC B1B2B3B4), how many end with NNNN, and how noisy their text is.

The database is opened read-only and is never changed, so the script is safe
to run on a database that the decoder is still writing to.

Usage:

    python tools/navtex_report.py 490kHz.db
    python tools/navtex_report.py 490kHz.db --since 2026-10-01 --until 2026-10-08
    python tools/navtex_report.py 490kHz.db --examples 3 --csv messages.csv

    --since T, --until T   only lines from T (inclusive) to T (exclusive);
                           T is a UTC time such as 2026-10-01 or
                           "2026-10-01 06:30"
    --examples N           show up to N example messages for each kind of
                           header problem and each kind of ending (default 0)
    --csv FILE             write one row per message, to look at in a
                           spreadsheet or database viewer
    --max-gap SECONDS      a message with no new line for this long is taken
                           to have ended (default 600)

What the words mean
-------------------
A *message* runs from a ZCZC to the next NNNN. It also ends, without an NNNN,
when another ZCZC starts, when no line arrives for --max-gap seconds, or when
the data ends.

Header kinds (the four characters after ZCZC):
  ok                two letters then two digits, e.g. "BA91"
  space_lost        as ok, but the space after ZCZC is missing
  digits_as_letters the digits arrived as the letters on the same keys
                    (Q=1 W=2 E=3 R=4 T=5 Y=6 U=7 I=8 O=9 P=0), i.e. a figure
                    shift was lost, e.g. "MZOI" for "MZ98"
  extra_char        a well-formed header preceded by one stray character
  has_noise         the decoder marked a header character as unreadable (~ or !)
  symbol_in_number  the number holds symbols or one digit and one symbol
  other             anything else
  missing           nothing follows ZCZC

Noise kinds (share of the message's characters that are ~ or !):
  empty             fewer than 5 characters
  clean             under 1%
  noisy             1% to under 10%
  garbage           10% or more

A message counts as having a *good header* when its header is ok or
space_lost. Whether the letters are the right letters cannot be known: a
damaged B1 or B2 is still a letter.

Exit status: 0 on success, 1 if the database cannot be read, 2 for a mistake
on the command line.
"""

import argparse
import csv
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_MAX_GAP = 600
UNREADABLE = "~!"
# The letters whose keys carry the digits 1 2 3 4 5 6 7 8 9 0 in figures mode.
FIGURE_KEYS = "QWERTYUIOP"
TOKEN = re.compile(r"ZCZC|NNNN")

HEADER_KINDS = (
    "ok", "space_lost", "digits_as_letters", "extra_char",
    "has_noise", "symbol_in_number", "other", "missing",
)
END_KINDS = ("NNNN", "next_start", "gap", "data_end")
NOISE_KINDS = ("clean", "noisy", "garbage", "empty")
GOOD_HEADERS = ("ok", "space_lost")


@dataclass
class Line:
    id: int
    time: datetime
    quality: int
    text: str


@dataclass
class Message:
    start_id: int
    start_time: datetime
    end_id: int = 0
    end_time: datetime = None
    # pieces[0] stands for the ZCZC itself; each later piece is the text one
    # line contributed. qualities holds the line's 00-99 signal quality.
    pieces: list = field(default_factory=list)
    qualities: dict = field(default_factory=dict)
    end: str = ""


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def parse_time(text):
    """A UTC time as written in the database, or a date, or date and minutes."""
    for form in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), form)
        except ValueError:
            pass
    raise ValueError(f"{text!r} is not a time like 2026-10-01 or 2026-10-01 06:30")


def open_read_only(path):
    """A connection that cannot change the database: it is opened read-only,
    and SQLite is also told to refuse any write."""
    # as_uri() turns spaces, %, # and ? into the forms a URI needs, and a
    # Windows drive letter into file:///D:/...
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def read_lines(path, since=None, until=None):
    """Yields the lines of the database in order."""
    connection = open_read_only(path)
    try:
        query = "SELECT id, utc_timestamp, signal_strength, text_output FROM navtex_log"
        where, args = [], []
        if since is not None:
            where.append("utc_timestamp >= ?")
            args.append(since.strftime("%Y-%m-%d %H:%M:%S"))
        if until is not None:
            where.append("utc_timestamp < ?")
            args.append(until.strftime("%Y-%m-%d %H:%M:%S"))
        if where:
            query += " WHERE " + " AND ".join(where)
        for row_id, stamp, quality, text in connection.execute(query + " ORDER BY id", args):
            try:
                time = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue  # a row whose time cannot be read cannot be placed
            yield Line(row_id, time, quality, text)
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Cutting the lines into messages
# ---------------------------------------------------------------------------

def segment(lines, max_gap=DEFAULT_MAX_GAP):
    """Returns (messages, orphan_ends): the messages in order, and how many
    NNNN were seen with no message open."""
    gap_limit = timedelta(seconds=max_gap)
    messages = []
    current = None
    previous = None
    orphans = 0

    def close(kind, line):
        nonlocal current
        current.end = kind
        current.end_id = line.id
        current.end_time = line.time
        messages.append(current)
        current = None

    for line in lines:
        if current is not None and previous is not None and line.time - previous.time > gap_limit:
            close("gap", previous)
        previous = line
        position = 0
        for token in TOKEN.finditer(line.text):
            before = line.text[position:token.start()]
            position = token.end()
            if current is not None:
                current.pieces.append(before)
                current.qualities[line.id] = line.quality
            if token.group() == "ZCZC":
                if current is not None:
                    close("next_start", line)
                current = Message(line.id, line.time, pieces=[""])
            elif current is None:
                orphans += 1
            else:
                close("NNNN", line)
        if current is not None:
            # The rest of the line after the last marker (or all of a line with none).
            current.pieces.append(line.text[position:])
            current.qualities[line.id] = line.quality
    if current is not None:
        close("data_end", previous)
    return messages, orphans


# ---------------------------------------------------------------------------
# Classifying
# ---------------------------------------------------------------------------

def classify_header(text, space=None):
    """Returns (kind, header) where header is the four characters that were
    taken to be B1B2B3B4 ('' if none). `space` says whether the space after
    ZCZC arrived; if it is not given, it is read from the text."""
    if space is None:
        space = text.startswith(" ")
    rest = text[1:] if text.startswith(" ") else text
    four = rest[:4]
    if not rest.strip():
        return "missing", ""

    def well_formed(chars):
        return (len(chars) == 4 and chars[0].isascii() and chars[0].isalpha() and chars[1].isascii()
                and chars[1].isalpha() and chars[2:].isascii() and chars[2:].isdigit())

    if well_formed(four):
        return ("ok" if space else "space_lost"), four
    if (len(four) == 4 and four[:2].isalpha() and four[:2].isascii()
            and all(c in FIGURE_KEYS for c in four[2:])):
        return "digits_as_letters", four
    if well_formed(rest[1:5]):
        return "extra_char", rest[1:5]
    if any(c in UNREADABLE for c in four):
        return "has_noise", four
    if (len(four) == 4 and four[:2].isalpha() and four[:2].isascii()
            and any(not c.isalnum() or c.isdigit() for c in four[2:])):
        return "symbol_in_number", four
    return "other", four


def noise(text):
    """Returns (kind, share): the share of non-blank characters that are ~ or !."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 5:
        return "empty", 0.0
    share = sum(c in UNREADABLE for c in chars) / len(chars)
    if share < 0.01:
        return "clean", share
    if share < 0.10:
        return "noisy", share
    return "garbage", share


@dataclass
class Result:
    message: Message
    header_kind: str
    header: str
    noise_kind: str
    noise_share: float

    @property
    def good_header(self):
        return self.header_kind in GOOD_HEADERS

    @property
    def mean_quality(self):
        qualities = list(self.message.qualities.values())
        return sum(qualities) / len(qualities) if qualities else 0.0


def header_of(message):
    """Classifies the header. It normally follows ZCZC on the same line; if
    ZCZC ended its line, the first text on a later line is used, and the
    line break stands in for the space."""
    for number, piece in enumerate(message.pieces[1:]):
        if piece.strip():
            return classify_header(piece, space=True if number else None)
    return "missing", ""


def classify(messages):
    results = []
    for message in messages:
        kind, header = header_of(message)
        # Everything after ZCZC, joined by the line breaks it had.
        body = "\n".join(message.pieces[1:])
        noise_kind, share = noise(body)
        results.append(Result(message, kind, header, noise_kind, share))
    return results


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def percent(count, total):
    return f"{100 * count / total:5.1f}%" if total else "    - "


def table(title, counts, order, total):
    lines = [title]
    for name in order:
        lines.append(f"  {name:<20}{counts.get(name, 0):>8}  {percent(counts.get(name, 0), total)}")
    return lines


def summarise(results, orphans, line_count, first, last):
    total = len(results)
    out = [f"Lines read: {line_count}" + (f"   from {first} to {last} UTC" if first else ""),
           f"Messages (ZCZC seen): {total}   NNNN with no message open: {orphans}", ""]
    out += table("Header", Counter(r.header_kind for r in results), HEADER_KINDS, total) + [""]
    out += table("Ending", Counter(r.message.end for r in results), END_KINDS, total) + [""]
    out += table("Text", Counter(r.noise_kind for r in results), NOISE_KINDS, total) + [""]

    combined = Counter()
    for r in results:
        combined[("good header" if r.good_header else "bad header",
                  "NNNN" if r.message.end == "NNNN" else "no NNNN", r.noise_kind)] += 1
    out.append("Header / ending / text together")
    for key, count in sorted(combined.items(), key=lambda item: -item[1]):
        out.append(f"  {key[0]:<12}{key[1]:<9}{key[2]:<9}{count:>8}  {percent(count, total)}")
    out.append("")

    good = [r for r in results if r.good_header]
    stations = Counter(r.header[0] for r in good)
    subjects = Counter(r.header[1] for r in good)
    out.append("Station letter (B1), good headers:  " + "  ".join(f"{k}:{v}" for k, v in stations.most_common()))
    out.append("Subject letter (B2), good headers:  " + "  ".join(f"{k}:{v}" for k, v in subjects.most_common()))
    seen = Counter(r.header for r in good)
    times = Counter(min(count, 3) for count in seen.values())
    out.append(f"Distinct headers: {len(seen)}   seen once: {times[1]}   twice: {times[2]}   three or more: {times[3]}")
    return out


def examples(results, how_many):
    """A few real messages of each kind of header problem and ending."""
    out = []
    if how_many <= 0:
        return out
    groups = [("header " + kind, [r for r in results if r.header_kind == kind])
              for kind in HEADER_KINDS if kind != "ok"]
    groups += [("ending " + kind, [r for r in results if r.message.end == kind]) for kind in END_KINDS if kind != "NNNN"]
    for title, members in groups:
        if not members:
            continue
        out += ["", f"Examples: {title}"]
        step = max(1, len(members) // how_many)
        for r in members[::step][:how_many]:
            m = r.message
            text = " | ".join(p for p in m.pieces[1:] if p.strip())
            shown = text[:90].encode("unicode_escape").decode("ascii")
            out.append(f"  row {m.start_id} {m.start_time:%Y-%m-%d %H:%M:%S}  ZCZC{shown}")
    return out


def write_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["message", "start_id", "end_id", "start_time", "end_time", "header_kind",
                         "header", "end", "noise_kind", "noise_percent", "mean_quality"])
        for number, r in enumerate(results, 1):
            m = r.message
            writer.writerow([number, m.start_id, m.end_id, f"{m.start_time:%Y-%m-%d %H:%M:%S}",
                             f"{m.end_time:%Y-%m-%d %H:%M:%S}", r.header_kind, r.header, m.end,
                             r.noise_kind, f"{100 * r.noise_share:.1f}", f"{r.mean_quality:.0f}"])


def build_report(path, since=None, until=None, max_gap=DEFAULT_MAX_GAP, how_many=0, csv_path=None):
    """Returns the report as a list of lines."""
    lines = list(read_lines(path, since, until))
    messages, orphans = segment(lines, max_gap)
    results = classify(messages)
    if csv_path:
        write_csv(results, csv_path)
    first = f"{lines[0].time:%Y-%m-%d %H:%M:%S}" if lines else ""
    last = f"{lines[-1].time:%Y-%m-%d %H:%M:%S}" if lines else ""
    return summarise(results, orphans, len(lines), first, last) + examples(results, how_many)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Message-quality report for a NAVTEX decoder database.")
    parser.add_argument("database")
    parser.add_argument("--since", type=parse_time)
    parser.add_argument("--until", type=parse_time)
    parser.add_argument("--examples", type=int, default=0, metavar="N")
    parser.add_argument("--csv", metavar="FILE")
    parser.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP, metavar="SECONDS")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit:
        return exit.code if isinstance(exit.code, int) else 2
    try:
        report = build_report(args.database, args.since, args.until, args.max_gap, args.examples, args.csv)
    except (sqlite3.Error, OSError) as error:
        print(f"{args.database}: {error}", file=sys.stderr)
        return 1
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
