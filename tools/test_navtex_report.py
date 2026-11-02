# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Tests for navtex_report.py. Run from anywhere:

    python tools/test_navtex_report.py

They build small databases of their own, so nothing else is needed. Set
NAVTEX_REAL_DB to the path of a real decoder database and
NAVTEX_REAL_MESSAGES to the number of ZCZC it should contain to add a check
on real data.
"""

import contextlib
import csv
import hashlib
import io
import os
import re
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import navtex_report as report  # noqa: E402

START = datetime(2026, 10, 18, 12, 0, 0)


def make_db(rows):
    """A database like the decoder's. `rows` is a list of (seconds, quality,
    text): the line's time is START plus the seconds."""
    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE navtex_log (id INTEGER PRIMARY KEY, utc_timestamp TEXT NOT NULL, "
        "signal_strength INTEGER NOT NULL, text_output TEXT NOT NULL)")
    for seconds, quality, text in rows:
        connection.execute(
            "INSERT INTO navtex_log (utc_timestamp, signal_strength, text_output) VALUES (?, ?, ?)",
            ((START + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S"), quality, text))
    connection.commit()
    connection.close()
    return path


def lines(*texts, start=0, step=3, quality=50):
    return [(start + i * step, quality, t) for i, t in enumerate(texts)]


def analyse(rows, **options):
    path = make_db(rows)
    try:
        read = list(report.read_lines(path, options.get("since"), options.get("until")))
        messages, orphans = report.segment(read, options.get("max_gap", report.DEFAULT_MAX_GAP))
        return report.classify(messages), orphans
    finally:
        os.remove(path)


def kinds(rows, **options):
    results, _ = analyse(rows, **options)
    return [(r.header_kind, r.message.end, r.noise_kind) for r in results]


BODY = "270900 UTC SEP 26"
TEXT = "COASTAL WARNING NAVIGATION HAZARD"


class Classifying(unittest.TestCase):
    def test_a_clean_message(self):
        rows = lines("ZCZC BA91", BODY, TEXT, "2. CANCEL THIS MSG 281200 UTC SEP 26NNNN")
        self.assertEqual(kinds(rows), [("ok", "NNNN", "clean")])

    def test_the_header_kinds(self):
        cases = {
            "ZCZC BA91": "ok",
            "ZCZCBA91": "space_lost",
            "ZCZC MZOI": "digits_as_letters",
            "ZCZC OAWP": "digits_as_letters",
            "ZCZC OAKP": "other",  # K is not on a digit key
            "ZCZC OAQ0": "symbol_in_number",
            "ZCZCIOA57": "extra_char",
            "ZCZC B~91": "has_noise",
            "ZCZC BA!1": "has_noise",
            "ZCZC OA,1": "symbol_in_number",
            "ZCZC  E20": "other",
            "ZCZC KQ": "other",
        }
        for header, expected in cases.items():
            with self.subTest(header=header):
                rows = lines(header, BODY, TEXT, "NNNN")
                self.assertEqual(kinds(rows)[0][0], expected)

    def test_the_header_may_arrive_on_the_line_after_zczc(self):
        rows = lines("ZCZC", "BA92", BODY, TEXT, "NNNN")
        self.assertEqual(kinds(rows), [("ok", "NNNN", "clean")])

    def test_zczc_with_nothing_after_it_is_a_missing_header(self):
        self.assertEqual(kinds(lines("ZCZC"))[0][0], "missing")

    def test_a_good_header_is_ok_or_space_lost_only(self):
        results, _ = analyse(lines("ZCZC BA91", "NNNN", "ZCZCBA92", "NNNN", "ZCZC MZOI", "NNNN"))
        self.assertEqual([r.good_header for r in results], [True, True, False])

    def test_the_station_and_number_are_taken_from_the_header(self):
        results, _ = analyse(lines("ZCZC GB17", TEXT, "NNNN"))
        self.assertEqual(results[0].header, "GB17")

    def test_noise_kinds(self):
        def noise(text):
            return report.noise(text)[0]
        self.assertEqual(noise("ABCD"), "empty")
        self.assertEqual(noise("A" * 100), "clean")
        self.assertEqual(noise("A" * 99 + "~"), "noisy")  # exactly 1%
        self.assertEqual(noise("A" * 91 + "~" * 9), "noisy")
        self.assertEqual(noise("A" * 90 + "~" * 10), "garbage")  # exactly 10%
        self.assertEqual(noise("A B C ~ ! " * 5), "garbage")
        self.assertEqual(noise("A" * 99 + "\n\n" + "  "), "clean", "blanks are not counted")

    def test_both_unreadable_marks_count_as_noise(self):
        self.assertAlmostEqual(report.noise("A" * 8 + "~!")[1], 0.2)


class ShiftedMarkers(unittest.TestCase):
    def test_both_forms_of_each_marker_start_and_end_a_message(self):
        for start in ("ZCZC", "+:+:"):
            for end in ("NNNN", ",,,,"):
                with self.subTest(start=start, end=end):
                    results, orphans = analyse(lines(start + " BA91", BODY, TEXT, end))
                    self.assertEqual((len(results), orphans, results[0].message.end), (1, 0, "NNNN"))
                    self.assertEqual((results[0].message.start_marker, results[0].message.end_marker),
                                     (start, end))

    def test_a_shifted_end_with_no_message_open_is_counted(self):
        self.assertEqual(analyse(lines(TEXT, ",,,,"))[1], 1)

    def test_a_shifted_end_glued_to_the_last_line(self):
        results, _ = analyse(lines("+:+: 2-68", BODY, TEXT + ",,,,"))
        self.assertEqual(results[0].message.end, "NNNN")
        self.assertEqual(results[0].noise_kind, "clean")

    def test_a_shifted_start_ends_the_message_before_it(self):
        results, _ = analyse(lines("ZCZC BA91", TEXT, "+:+: 2-68", TEXT, "NNNN"))
        self.assertEqual([r.message.end for r in results], ["next_start", "NNNN"])

    def test_five_commas_are_one_end_and_a_stray_comma(self):
        results, orphans = analyse(lines("ZCZC BA91", TEXT + ",,,,,"))
        self.assertEqual((results[0].message.end, orphans), ("NNNN", 0))

    def test_the_shifted_header_is_decoded(self):
        cases = {  # real examples from the databases
            "+:+: \x07-13": ("JA13", "\x07-13"),
            "+:+: .+00": ("MZ00", ".+00"),
            "+:+: ,-72": ("NA72", ",-72"),
            "+:+: 2-68": ("WA68", "2-68"),
            "+:+: 9-91": ("OA91", "9-91"),
            "+:+:2-68": ("WA68", "2-68"),  # the space may be lost
        }
        for text, (decoded, raw) in cases.items():
            with self.subTest(text=text):
                results, _ = analyse(lines(text, BODY, TEXT, "NNNN"))
                r = results[0]
                self.assertEqual((r.header_kind, r.header, r.header_raw, r.header_repair),
                                 ("figures_shifted", decoded, raw, "figures"))
                self.assertTrue(r.good_header)

    def test_the_shifted_header_may_arrive_on_the_next_line(self):
        results, _ = analyse(lines("+:+:", "2-68", BODY, TEXT, "NNNN"))
        self.assertEqual((results[0].header_kind, results[0].header), ("figures_shifted", "WA68"))

    def test_shifted_headers_that_cannot_be_decoded_are_not_guessed(self):
        cases = {
            "+:+: ~-68": "has_noise",   # could be D, F, G or H: left alone
            "+:+: 2!68": "has_noise",
            "+:+: 2-6": "other",
            "+:+: 2-QW": "other",       # the digits must be digits
            "+:+: 2D68": "other",       # D has no figure
        }
        for text, kind in cases.items():
            with self.subTest(text=text):
                r = analyse(lines(text, BODY, TEXT, "NNNN"))[0][0]
                self.assertEqual((r.header_kind, r.header, r.header_repair), (kind, r.header_raw, "none"))
                self.assertFalse(r.good_header)

    def test_figures_in_the_header_of_an_ordinary_start_are_not_decoded(self):
        r = analyse(lines("ZCZC 2-68", BODY, TEXT, "NNNN"))[0][0]
        self.assertEqual((r.header_kind, r.header_repair), ("other", "none"))

    def test_an_ordinary_header_after_a_shifted_start_is_still_ok(self):
        r = analyse(lines("+:+: BA91", BODY, TEXT, "NNNN"))[0][0]
        self.assertEqual((r.header_kind, r.header, r.header_repair), ("ok", "BA91", "none"))

    def test_digits_as_letters_is_decoded_but_still_not_a_good_header(self):
        r = analyse(lines("ZCZC MZOI", BODY, TEXT, "NNNN"))[0][0]
        self.assertEqual((r.header, r.header_raw, r.header_repair, r.good_header),
                         ("MZ98", "MZOI", "digits_as_letters", False))

    def test_every_digit_key_decodes(self):
        for key, digit in zip("QWERTYUIOP", "1234567890"):
            with self.subTest(key=key):
                self.assertEqual(report.decode_header("digits_as_letters", "AB" + key + "Q"),
                                 ("AB" + digit + "1", "digits_as_letters"))

    def test_every_figure_glyph_decodes(self):
        for letter, figure in report.FIGURE_OF_LETTER.items():
            with self.subTest(letter=letter):
                self.assertEqual(report.decode_header("figures_shifted", figure + "-12"),
                                 (letter + "A12", "figures"))

    def test_the_raw_header_is_kept_for_headers_that_are_not_repaired(self):
        r = analyse(lines("ZCZC BA91", BODY, TEXT, "NNNN"))[0][0]
        self.assertEqual((r.header, r.header_raw, r.header_repair), ("BA91", "BA91", "none"))

    def test_the_figure_table_agrees_with_the_decoders_charset(self):
        source = Path(__file__).resolve().parent.parent / "navtex-core" / "src" / "charset.rs"
        if not source.exists():
            self.skipTest("navtex-core/src/charset.rs is not next to the tools folder")
        found = re.findall(r"\(0b[01]+, '(.)', Some\('(\\x07|\\'|.)'\)\)", source.read_text(encoding="utf-8"))
        table = {letter: "\x07" if figure == "\\x07" else figure.replace("\\'", "'") for letter, figure in found}
        self.assertEqual(len(table), 22)  # D, F, G, H have no figure
        self.assertEqual(table, report.FIGURE_OF_LETTER)

    def test_the_figure_table_is_one_to_one(self):
        self.assertEqual(len(report.LETTER_OF_FIGURE), len(report.FIGURE_OF_LETTER))
        for letter, figure in report.FIGURE_OF_LETTER.items():
            self.assertEqual(report.LETTER_OF_FIGURE[figure], letter)
        self.assertEqual("".join(report.FIGURE_OF_LETTER[c] for c in "ZCZC"), "+:+:")
        self.assertEqual("".join(report.FIGURE_OF_LETTER[c] for c in "NNNN"), ",,,,")


class DamagedStarts(unittest.TestCase):
    FORMS = {"ZCZC": "BA91", "+:+:": "2-68"}   # each form with a header that suits it

    def damaged(self, form, position, mark):
        return form[:position] + mark + form[position + 1:]

    def test_one_unreadable_character_anywhere_in_either_form_is_accepted_with_a_valid_header(self):
        for form, header in self.FORMS.items():
            for position in range(4):
                for mark in "~!":
                    text = self.damaged(form, position, mark)
                    with self.subTest(text=text):
                        results, orphans = analyse(lines(text + " " + header, BODY, TEXT, "NNNN"))
                        self.assertEqual(len(results), 1)
                        m = results[0].message
                        self.assertEqual((m.start_marker, m.start_received, m.start_damaged, m.end),
                                         (form, text, True, "NNNN"))
                        self.assertTrue(results[0].good_header)

    def test_the_header_of_a_damaged_figures_start_is_decoded(self):
        r = analyse(lines("+:+~ 2-68", BODY, TEXT, "NNNN"))[0][0]
        self.assertEqual((r.header_kind, r.header, r.header_raw), ("figures_shifted", "WA68", "2-68"))

    def test_real_examples(self):
        cases = {"~CZC FA01": "FA01", "ZCZ~ EA13": "EA13", "~:+: 2-66": "WA66", "ZCZ~ OA91": "OA91",
                 "~~~+~+~+: 1320": "QE20"}
        for text, header in cases.items():
            with self.subTest(text=text):
                results, _ = analyse(lines(text, BODY, TEXT, "NNNN"))
                self.assertEqual((len(results), results[0].header), (1, header))
                self.assertTrue(results[0].message.start_damaged)

    def test_without_a_valid_header_it_is_not_a_start(self):
        cases = ["ZC~C", "ZC~C HELLO", "ZC~C B~91", "ZC~C BA!1", "ZC~C 9", "ZC~C BA9", "+:+~ 2-6",
                 "+:+~ ~-29", "ZC~C OAKP", "ZC~C  BA91"]
        for text in cases:
            with self.subTest(text=text):
                results, orphans = analyse(lines(text, BODY, TEXT, "NNNN"))
                self.assertEqual((len(results), orphans), (0, 1))

    def test_a_header_on_the_next_line_is_not_looked_at(self):
        self.assertEqual(len(analyse(lines("ZC~C", "BA91", BODY, TEXT, "NNNN"))[0]), 0)

    def test_digits_as_letters_is_a_valid_header_for_a_damaged_start(self):
        r = analyse(lines("ZC~C MZOI", BODY, TEXT, "NNNN"))[0][0]
        self.assertEqual((r.header_kind, r.header), ("digits_as_letters", "MZ98"))

    def test_two_unreadable_characters_are_not_enough(self):
        self.assertEqual(len(analyse(lines("Z~~C BA91", BODY, TEXT, "NNNN"))[0]), 0)

    def test_mixed_letter_and_figure_forms_stay_unused(self):
        self.assertEqual(len(analyse(lines("Z:Z~ BA91", BODY, TEXT, "NNNN"))[0]), 0)
        self.assertEqual(len(analyse(lines("ZC~: BA91", BODY, TEXT, "NNNN"))[0]), 0)

    def test_a_real_start_is_never_displaced_by_a_look_alike(self):
        for text in ("Z~ZCZC BA91", "~CZCZC BA91", "ZCZ~ZCZC BA91", "ZC~ZCZC BA91"):
            with self.subTest(text=text):
                results, _ = analyse(lines(text, BODY, TEXT, "NNNN"))
                self.assertEqual([(r.message.start_marker, r.message.start_damaged) for r in results],
                                 [("ZCZC", False)])

    def test_a_look_alike_overlapping_a_real_marker_is_not_a_second_start(self):
        for text in ("ZCZC~C BA91", "+:+:+~: 2-68"):
            with self.subTest(text=text):
                results, _ = analyse(lines(text, BODY, TEXT, "NNNN"))
                self.assertEqual([(r.message.start_received, r.message.start_damaged) for r in results],
                                 [(text[:4], False)])
                self.assertEqual(report.count_markers(self.read(text))["damaged_used"], 0)

    def read(self, text):
        path = make_db(lines(text))
        self.addCleanup(os.remove, path)
        return report.read_lines(path)

    def test_a_damaged_start_ends_the_message_before_it(self):
        results, _ = analyse(lines("ZCZC BA91", TEXT, "ZCZ~ EA13", TEXT, "NNNN"))
        self.assertEqual([(r.message.end, r.header) for r in results], [("next_start", "BA91"), ("NNNN", "EA13")])

    def test_a_damaged_start_with_junk_in_front_and_a_real_start_later_on_the_line(self):
        results, _ = analyse(lines("XX~~ZCZ~ EA13", TEXT + "NNNN~ZCZC BA91", BODY, "NNNN"))
        self.assertEqual([r.header for r in results], ["EA13", "BA91"])

    def test_a_damaged_start_can_precede_an_nnnn_that_would_have_been_an_orphan(self):
        self.assertEqual(analyse(lines(TEXT, "NNNN"))[1], 1)
        results, orphans = analyse(lines("ZCZ~ EA13", TEXT, "NNNN"))
        self.assertEqual((len(results), orphans), (1, 0))

    def test_damaged_ends_are_not_used(self):
        for end in ("NNN~", "~NNN", "N~NN", ",,,~", ",,~,"):
            with self.subTest(end=end):
                results, orphans = analyse(lines("ZCZC BA91", TEXT, end))
                self.assertEqual(results[0].message.end, "data_end")

    def test_exact_starts_are_marked_exact_in_the_csv(self):
        path = make_db(lines("ZCZC BA91", BODY, TEXT, "NNNN") + lines("ZCZ~ EA13", BODY, TEXT, "NNNN", start=60))
        target = path + ".csv"
        self.addCleanup(os.remove, path)
        self.addCleanup(os.remove, target)
        report.build_report(path, csv_path=target)
        with open(target, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(r["start_kind"], r["start_received"], r["start_marker"]) for r in rows],
                         [("exact", "ZCZC", "ZCZC"), ("damaged", "ZCZ~", "ZCZC")])

    def test_the_markers_block_counts_damaged_starts_used_and_not_used(self):
        path = make_db(lines("ZCZ~ EA13", BODY, TEXT, "NNNN") + lines("ZC~C", "+:+~ 2-68", "Z!ZC", start=60))
        self.addCleanup(os.remove, path)
        text = "\n".join(report.build_report(path))
        self.assertRegex(text, r"damaged start, used\s+2\s+damaged, no header\s+2")
        self.assertIn("Messages (ZCZC seen): 2 ", text)

    def test_the_marker_counts_agree_with_what_segment_uses(self):
        rows = lines("ZCZ~ EA13", "+:+~ 2-68", "ZC~C junk", "ZCZC BA91", "NNNN", "NNN~")
        path = make_db(rows)
        self.addCleanup(os.remove, path)
        counts = report.count_markers(report.read_lines(path))
        results, _ = analyse(rows)
        self.assertEqual(counts["damaged_used"], sum(r.message.start_damaged for r in results))
        self.assertEqual(counts["damaged_unused"], 1)


class MixedMarkers(unittest.TestCase):
    def count(self, *texts):
        path = make_db(lines(*texts))
        try:
            return report.count_markers(report.read_lines(path))
        finally:
            os.remove(path)

    def test_pure_forms_are_counted(self):
        c = self.count("ZCZC BA91", "+:+: 2-68", "NNNN", ",,,,", ",,,,")
        self.assertEqual((c["ZCZC"], c["+:+:"], c["NNNN"], c[",,,,"]), (1, 1, 1, 2))
        self.assertEqual((c["mixed_start"], c["mixed_end"]), (0, 0))

    def test_mixed_forms_are_counted_but_do_not_start_or_end_anything(self):
        rows = lines("Z:ZC BA91", "ZC+: BA91", TEXT, "N,N,", "NN,,")
        c = self.count(*[r[2] for r in rows])
        self.assertEqual((c["mixed_start"], c["mixed_end"]), (2, 2))
        results, orphans = analyse(rows)
        self.assertEqual((len(results), orphans), (0, 0))

    def test_a_mixed_look_alike_cannot_hide_or_be_counted_from_a_real_marker(self):
        c = self.count("N,NNNN", "+ZCZC", "Z:+:+:")
        self.assertEqual((c["NNNN"], c["mixed_end"]), (1, 0))
        self.assertEqual((c["ZCZC"], c["+:+:"], c["mixed_start"]), (1, 1, 0))
        # and the real end is still found
        self.assertEqual(analyse(lines("ZCZC BA91", "N,NNNN"))[0][0].message.end, "NNNN")


class Segmenting(unittest.TestCase):
    def test_nnnn_on_the_end_of_a_line_ends_the_message_and_is_not_part_of_its_text(self):
        read = [report.Line(i, START, 50, t) for i, t in enumerate(["ZCZC BA91", "HELLO 26NNNN", "ZCZC BA92"], 1)]
        messages, _ = report.segment(read)
        self.assertEqual(messages[0].end, "NNNN")
        self.assertEqual(messages[0].pieces, ["", " BA91", "HELLO 26"])
        self.assertEqual((messages[0].start_id, messages[0].end_id), (1, 2))

    def test_junk_in_front_of_zczc_is_ignored(self):
        rows = lines("WZCZC BA97", TEXT, "XZCZC BA98", TEXT, "NNNN")
        self.assertEqual(kinds(rows), [("ok", "next_start", "clean"), ("ok", "NNNN", "clean")])

    def test_unreadable_marks_just_before_a_new_zczc_count_against_the_message_before(self):
        # They are most likely what is left of that message's NNNN.
        rows = lines("ZCZC BA97", TEXT, "N~NN" + "ZCZC BA98", TEXT, "NNNN")
        self.assertEqual(kinds(rows)[0], ("ok", "next_start", "noisy"))

    def test_text_after_nnnn_is_not_part_of_any_message(self):
        results, _ = analyse(lines("ZCZC BA91", TEXT, "NNNN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "garbage ~~~~~~~~~"))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].noise_kind, "clean")

    def test_a_new_zczc_ends_the_message_before_it(self):
        rows = lines("ZCZC BA93", TEXT, "ZCZC BA94", TEXT, "NNNN")
        self.assertEqual(kinds(rows), [("ok", "next_start", "clean"), ("ok", "NNNN", "clean")])

    def test_two_zczc_on_one_line_make_two_messages(self):
        rows = lines("ZCZC BA93 ZCZC BA94", TEXT, "NNNN")
        self.assertEqual([k[1] for k in kinds(rows)], ["next_start", "NNNN"])

    def test_a_message_and_the_next_start_on_the_same_line_as_nnnn(self):
        results, orphans = analyse(lines("ZCZC BA93", TEXT + "NNNNZCZC BA94", TEXT, "NNNN"))
        self.assertEqual([r.message.end for r in results], ["NNNN", "NNNN"])
        self.assertEqual([r.header for r in results], ["BA93", "BA94"])
        self.assertEqual(orphans, 0)

    def test_a_long_silence_ends_the_message(self):
        rows = lines("ZCZC BA95", TEXT) + lines("ZCZC BA96", TEXT, "NNNN", start=1200)
        self.assertEqual([k[1] for k in kinds(rows)], ["gap", "NNNN"])
        self.assertEqual([k[1] for k in kinds(rows, max_gap=2000)], ["next_start", "NNNN"])

    def test_the_gap_message_ends_at_its_last_line(self):
        results, _ = analyse(lines("ZCZC BA95", TEXT) + lines("ZCZC BA96", start=1200))
        self.assertEqual(results[0].message.end_time, START + timedelta(seconds=3))

    def test_data_that_stops_inside_a_message(self):
        results, _ = analyse(lines("ZCZC BA95", TEXT))
        self.assertEqual(results[0].message.end, "data_end")

    def test_an_nnnn_with_no_message_open_is_counted(self):
        results, orphans = analyse(lines("NNNN", TEXT, "ZCZC BA95", TEXT, "NNNN", "NNNN"))
        self.assertEqual((len(results), orphans), (1, 2))

    def test_no_lines_gives_no_messages(self):
        self.assertEqual(analyse([]), ([], 0))

    def test_text_between_messages_with_no_zczc_is_ignored(self):
        results, _ = analyse(lines("lonely line ~~~~~", "ZCZC BA95", TEXT, "NNNN"))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].noise_kind, "clean")

    def test_the_mean_quality_is_over_the_messages_own_lines(self):
        rows = [(0, 90, "ZCZC BA95"), (3, 70, TEXT), (6, 80, "NNNN"), (9, 10, "other line")]
        results, _ = analyse(rows)
        self.assertEqual(results[0].mean_quality, 80)


class Reporting(unittest.TestCase):
    def setUp(self):
        rows = (lines("ZCZC BA91", BODY, TEXT, "NNNN") +
                lines("ZCZC BA91", BODY, TEXT, "NNNN", start=60) +
                lines("ZCZC GB07", BODY, "XX~~~~~~~~~~~~~~~~~~~~~XX", "NNNN", start=120) +
                lines("ZCZC MZOI", BODY, TEXT, start=180) +
                lines("NNNN", start=1000))
        self.path = make_db(rows)
        self.addCleanup(os.remove, self.path)

    def text(self, **options):
        return "\n".join(report.build_report(self.path, **options))

    def test_the_report_gives_the_counts(self):
        text = self.text()
        self.assertIn("Lines read: 16", text)
        self.assertIn("Messages (ZCZC seen): 4   NNNN with no message open: 1", text)
        self.assertRegex(text, r"ok\s+3\s+75\.0%")
        self.assertRegex(text, r"digits_as_letters\s+1\s+25\.0%")
        self.assertRegex(text, r"NNNN\s+3\s+75\.0%")
        self.assertRegex(text, r"gap\s+1\s+25\.0%")
        self.assertRegex(text, r"garbage\s+1\s+25\.0%")
        self.assertIn("Distinct headers: 2   seen once: 1   twice: 1   three or more: 0", text)
        self.assertIn("B:2  G:1", text)

    def test_the_markers_block_and_the_shifted_header_row_are_reported(self):
        path = make_db(lines("ZCZC BA91", BODY, TEXT, "NNNN") +
                       lines("+:+: 2-68", BODY, TEXT, ",,,,", start=60) +
                       lines("Z:ZC", "N,N,", start=120))
        self.addCleanup(os.remove, path)
        text = "\n".join(report.build_report(path))
        self.assertRegex(text, r"ZCZC\s+1\s+NNNN\s+1")
        self.assertRegex(text, r"\+:\+: \(figures\)\s+1\s+,,,, \(figures\)\s+1")
        self.assertRegex(text, r"mixed, not used\s+1\s+mixed, not used\s+1")
        self.assertRegex(text, r"figures_shifted\s+1\s+50\.0%")
        self.assertIn("Messages (ZCZC seen): 2   NNNN with no message open: 0", text)
        self.assertIn("Station letter (B1), good headers:  B:1  W:1", text)

    def test_since_and_until_limit_the_lines(self):
        self.assertIn("Messages (ZCZC seen): 2 ", self.text(since=report.parse_time("2026-10-18 12:01:00")
                                                            , until=report.parse_time("2026-10-18 12:03:00")))
        self.assertIn("Messages (ZCZC seen): 0 ", self.text(since=report.parse_time("2027-01-01")))

    def test_examples_are_shown_only_when_asked_for(self):
        self.assertNotIn("Examples:", self.text())
        shown = self.text(how_many=2)
        self.assertIn("Examples: header digits_as_letters", shown)
        self.assertIn("ZCZC MZOI", shown)
        self.assertIn("Examples: ending gap", shown)

    def test_the_csv_has_one_row_per_message(self):
        target = self.path + ".csv"
        self.addCleanup(os.remove, target)
        report.build_report(self.path, csv_path=target)
        with open(target, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual([r["header"] for r in rows], ["BA91", "BA91", "GB07", "MZ98"])
        self.assertEqual([r["header_raw"] for r in rows], ["BA91", "BA91", "GB07", "MZOI"])
        self.assertEqual([r["header_repair"] for r in rows], ["none", "none", "none", "digits_as_letters"])
        self.assertEqual([r["start_marker"] for r in rows], ["ZCZC"] * 4)
        self.assertEqual([r["end_marker"] for r in rows], ["NNNN", "NNNN", "NNNN", ""])
        self.assertEqual(rows[3]["header_kind"], "digits_as_letters")
        self.assertEqual(rows[3]["end"], "gap")
        self.assertEqual(rows[2]["noise_kind"], "garbage")
        self.assertEqual((rows[0]["start_id"], rows[0]["end_id"]), ("1", "4"))

    def test_parse_time_accepts_dates_and_minutes(self):
        self.assertEqual(report.parse_time("2026-10-01"), datetime(2026, 10, 1))
        self.assertEqual(report.parse_time("2026-10-01 06:30"), datetime(2026, 10, 1, 6, 30))
        with self.assertRaises(ValueError):
            report.parse_time("yesterday")


class FormattedLog(unittest.TestCase):
    def log(self, rows, **options):
        """The formatted log of these rows, as text with \\n line ends, and the rows' ids."""
        path = make_db(rows)
        target = path + ".log"
        self.addCleanup(os.remove, path)
        self.addCleanup(lambda: os.path.exists(target) and os.remove(target))
        report.build_report(path, format_path=target, **options)
        with open(target, encoding="utf-8") as handle:  # universal newlines: \\n on any computer
            return handle.read()

    def test_the_example_from_the_specification(self):
        rows = [(0, 77, "ZCZC QA91"), (2, 85, "011840 UTC NOV 26"), (5, 85, "SPLIT RADIO"),
                (8, 82, "UNTIL FURTHER NOTICE, M/V MICOPERI TRENTA 45-03,7N 013-07,6E (WGS84)."),
                (40, 94, "NNNN")]
        self.assertEqual(self.log(rows), "\n".join([
            ">>> ZCZC, NNNN, QA91, none, exact, ZCZC",
            "",
            "[2026-10-18 12:00:00 77]\tZCZC QA91",
            "[2026-10-18 12:00:02 85]\t011840 UTC NOV 26",
            "[2026-10-18 12:00:05 85]\tSPLIT RADIO",
            "[2026-10-18 12:00:08 82]\tUNTIL FURTHER NOTICE, M/V MICOPERI TRENTA 45-03,7N 013-07,6E (WGS84).",
            "[2026-10-18 12:00:40 94]\tNNNN",
            "", ""]))

    def test_a_message_with_no_end_marker_has_an_empty_end_field_and_ends_at_its_last_row(self):
        text = self.log(lines("ZCZC BA91", TEXT, start=0) + lines("later", start=2000))
        self.assertTrue(text.startswith(">>> ZCZC, , BA91, none, exact, ZCZC\n\n"))
        block = text.split("\n\n")[1].splitlines()
        self.assertEqual(len(block), 2)
        self.assertTrue(text.endswith("\tlater\n\n"))
        self.assertIn(">>> FRAGMENT\n\n[2026-10-18 12:33:20 50]\tlater", text)

    def test_a_row_shared_by_two_messages_is_in_both(self):
        text = self.log(lines("ZCZC BA91", TEXT, "THE END NNNN ZCZC EA13", TEXT, "NNNN"))
        blocks = text.strip("\n").split("\n\n")
        self.assertEqual(blocks[0], ">>> ZCZC, NNNN, BA91, none, exact, ZCZC")
        self.assertEqual(blocks[2], ">>> ZCZC, NNNN, EA13, none, exact, ZCZC")
        self.assertEqual(blocks[1].count("THE END NNNN ZCZC EA13"), 1)
        self.assertEqual(blocks[3].splitlines()[0].split("\t")[1], "THE END NNNN ZCZC EA13")
        self.assertEqual(text.count("THE END NNNN ZCZC EA13"), 2)

    def test_rows_outside_messages_are_fragments(self):
        rows = lines("noise", "NNNN", "ZCZC BA91", TEXT, "NNNN", "more noise", "NNNN", start=0)
        text = self.log(rows)
        blocks = text.strip("\n").split("\n\n")
        self.assertEqual(blocks[0], ">>> FRAGMENT")
        self.assertEqual([l.split("\t")[1] for l in blocks[1].splitlines()], ["noise", "NNNN"])
        self.assertTrue(blocks[2].startswith(">>> ZCZC, NNNN, BA91"))
        self.assertEqual(blocks[4], ">>> FRAGMENT")
        self.assertEqual([l.split("\t")[1] for l in blocks[5].splitlines()], ["more noise", "NNNN"])

    def test_every_row_is_written_and_rows_outside_messages_exactly_once(self):
        rows = lines("x1", "ZCZC BA91", TEXT, "NNNN ZCZC EA13", TEXT, "NNNN", "x2", "x3", "ZCZC BA92", TEXT, "NNNN", "x4")
        text = self.log(rows)
        for number, (_, _, row_text) in enumerate(rows):
            count = text.count("\t" + row_text + "\n")
            expected = 2 if row_text.startswith("NNNN ZCZC") else 1 if row_text in ("x1", "x2", "x3", "x4") else None
            with self.subTest(row=row_text):
                if expected:
                    self.assertEqual(count, expected)
                else:
                    self.assertGreaterEqual(count, 1)
        order = [l[1:20] for l in text.splitlines() if l.startswith("[")]
        self.assertEqual(order, sorted(order))

    def test_since_and_until_limit_the_rows_in_the_log(self):
        rows = lines("ZCZC BA91", TEXT, "NNNN", start=0) + lines("ZCZC EA13", TEXT, "NNNN", start=120)
        text = self.log(rows, since=report.parse_time("2026-10-18 12:01:00"))
        self.assertNotIn("BA91", text)
        self.assertIn(">>> ZCZC, NNNN, EA13", text)
        self.assertEqual(self.log(rows, since=report.parse_time("2027-01-01")), "")

    def test_an_empty_database_gives_an_empty_file(self):
        self.assertEqual(self.log([]), "")

    def test_control_characters_are_written_as_escapes_in_the_text_and_the_header_line(self):
        rows = lines("+:+: \x07-13", "A\tB\x7fC \x07 \x00", "NNNN")
        text = self.log(rows)
        self.assertIn(">>> +:+:, NNNN, \\x07-13, figures, exact, +:+:\n", text)
        self.assertIn("\t+:+: \\x07-13\n", text)
        self.assertIn("\tA\\x09B\\x7fC \\x07 \\x00\n", text)
        # the only control characters left are the line ends and the one TAB after each [time quality]
        self.assertEqual(re.findall("[\x00-\x08\x0b-\x1f\x7f]", text), [])
        for row in (l for l in text.splitlines() if l.startswith("[")):
            self.assertEqual(row.count("\t"), 1)

    def test_a_comma_in_the_raw_header_is_kept(self):
        text = self.log(lines("+:+: ,-72", BODY, TEXT, ",,,,"))
        self.assertTrue(text.startswith(">>> +:+:, ,,,,, ,-72, figures, exact, +:+:\n"))

    def test_the_header_line_holds_the_same_values_as_the_csv(self):
        rows = (lines("ZCZC BA91", TEXT, "NNNN") + lines("ZCZ~ EA13", TEXT, "NNNN", start=60) +
                lines("+:+: 2-68", TEXT, start=120) + lines("ZCZC MZOI", TEXT, "NNNN", start=2000))
        path = make_db(rows)
        target = path + ".csv"
        self.addCleanup(os.remove, path)
        self.addCleanup(os.remove, target)
        log_path = path + ".log"
        self.addCleanup(os.remove, log_path)
        report.build_report(path, csv_path=target, format_path=log_path)
        with open(target, newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        with open(log_path, encoding="utf-8") as handle:
            headers = [l for l in handle.read().splitlines() if l.startswith(">>> ") and "FRAGMENT" not in l]
        expected = [">>> " + ", ".join([r["start_marker"], r["end_marker"], r["header_raw"], r["header_repair"],
                                        r["start_kind"], r["start_received"]]) for r in csv_rows]
        self.assertEqual(headers, expected)
        self.assertIn(">>> ZCZC, NNNN, EA13, none, damaged, ZCZ~", headers)
        self.assertIn(">>> ZCZC, NNNN, MZOI, digits_as_letters, exact, ZCZC", headers)

    def test_the_database_is_not_changed_and_the_report_is_unaffected(self):
        rows = lines("ZCZC BA91", TEXT, "NNNN")
        path = make_db(rows)
        self.addCleanup(os.remove, path)
        self.addCleanup(lambda: os.path.exists(path + ".log") and os.remove(path + ".log"))
        before = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        plain = report.build_report(path)
        with_log = report.build_report(path, format_path=path + ".log")
        self.assertEqual(plain, with_log)
        self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), before)

    def test_an_existing_file_is_replaced(self):
        path = make_db(lines("ZCZC BA91", TEXT, "NNNN"))
        target = path + ".log"
        self.addCleanup(os.remove, path)
        self.addCleanup(os.remove, target)
        Path(target).write_text("old contents that are longer than the new log would be? " * 400)
        report.build_report(path, format_path=target)
        self.assertTrue(Path(target).read_text(encoding="utf-8").startswith(">>> ZCZC"))
        self.assertNotIn("old contents", Path(target).read_text(encoding="utf-8"))

    def test_the_command_line_option_writes_the_log_and_still_prints_the_report(self):
        path = make_db(lines("ZCZC BA91", TEXT, "NNNN"))
        target = path + ".log"
        self.addCleanup(os.remove, path)
        self.addCleanup(os.remove, target)
        code, out, err = CommandLine.run_main(self, path, "--format", target)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("Messages (ZCZC seen): 1", out)
        self.assertTrue(Path(target).read_text(encoding="utf-8").startswith(">>> ZCZC, NNNN, BA91"))

    def test_a_log_that_cannot_be_written_gives_status_1_and_names_the_file(self):
        path = make_db(lines("ZCZC BA91", TEXT, "NNNN"))
        self.addCleanup(os.remove, path)
        bad = str(Path(tempfile.gettempdir()) / "no_such_folder" / "log.txt")
        code, out, err = CommandLine.run_main(self, path, "--format", bad)
        self.assertEqual((code, out), (1, ""))
        self.assertIn("log.txt", err)
        self.assertNotIn(Path(path).name, err)

    def test_format_needs_a_file_name(self):
        self.assertEqual(CommandLine.run_main(self, "x.db", "--format")[0], 2)


class Safety(unittest.TestCase):
    def test_the_database_is_not_changed_by_a_report(self):
        path = make_db(lines("ZCZC BA91", TEXT, "NNNN"))
        self.addCleanup(os.remove, path)
        before = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        report.build_report(path, how_many=3, csv_path=path + ".csv")
        os.remove(path + ".csv")
        self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), before)

    def test_the_database_can_be_read_while_the_decoder_has_it_open_in_wal_mode(self):
        path = make_db(lines("ZCZC BA91", TEXT))
        self.addCleanup(lambda: [os.path.exists(path + s) and os.remove(path + s) for s in ("", "-wal", "-shm")])
        writer = sqlite3.connect(path)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO navtex_log (utc_timestamp, signal_strength, text_output) VALUES "
                       "('2026-10-18 12:00:09', 50, 'NNNN')")
        writer.commit()  # the new row is in the -wal file, not yet in the database file
        self.addCleanup(writer.close)
        self.assertEqual(len(list(report.read_lines(path))), 3)
        writer.execute("INSERT INTO navtex_log (utc_timestamp, signal_strength, text_output) VALUES "
                       "('2026-10-18 12:00:12', 50, 'later')")
        writer.commit()  # the decoder can still write after the report has read

    def test_the_connection_used_for_reading_refuses_every_kind_of_write(self):
        path = make_db(lines("ZCZC BA91"))
        self.addCleanup(os.remove, path)
        connection = report.open_read_only(path)
        self.addCleanup(connection.close)
        for statement in ("DELETE FROM navtex_log", "CREATE TABLE t (x)", "DROP TABLE navtex_log",
                          "INSERT INTO navtex_log (utc_timestamp, signal_strength, text_output) VALUES ('x', 1, 'y')"):
            with self.subTest(statement=statement), self.assertRaises(sqlite3.OperationalError):
                connection.execute(statement)


class Paths(unittest.TestCase):
    def test_paths_with_spaces_and_odd_characters_can_be_opened(self):
        folder = tempfile.mkdtemp(prefix="odd name %41 #1 ")
        source = make_db(lines("ZCZC BA91", TEXT, "NNNN"))
        # Windows does not allow ? in a file name, so it is only tried elsewhere.
        target = os.path.join(folder, "my log#1.db" if os.name == "nt" else "my log?#1.db")
        os.replace(source, target)
        self.addCleanup(lambda: (os.remove(target), os.rmdir(folder)))
        self.assertEqual(len(list(report.read_lines(target))), 3)


class CommandLine(unittest.TestCase):
    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = report.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_a_report_is_printed_and_the_status_is_zero(self):
        path = make_db(lines("ZCZC BA91", TEXT, "NNNN"))
        self.addCleanup(os.remove, path)
        code, out, err = self.run_main(path)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("Messages (ZCZC seen): 1", out)

    def test_a_missing_database_is_reported_with_status_1(self):
        code, out, err = self.run_main(str(Path(tempfile.gettempdir()) / "no_such_folder" / "x.db"))
        self.assertEqual((code, out), (1, ""))
        self.assertIn("x.db", err)

    def test_a_file_that_is_not_a_database_gives_status_1(self):
        handle, path = tempfile.mkstemp(suffix=".db")
        os.write(handle, b"this is not a database at all" * 20)
        os.close(handle)
        self.addCleanup(os.remove, path)
        code, _, err = self.run_main(path)
        self.assertEqual(code, 1)
        self.assertIn(path, err)

    def test_a_database_without_the_table_gives_status_1(self):
        handle, path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        sqlite3.connect(path).close()
        self.addCleanup(os.remove, path)
        self.assertEqual(self.run_main(path)[0], 1)

    def test_a_mistake_on_the_command_line_gives_status_2(self):
        self.assertEqual(self.run_main()[0], 2)
        self.assertEqual(self.run_main("x.db", "--since", "yesterday")[0], 2)
        self.assertEqual(self.run_main("x.db", "--no-such-option")[0], 2)


@unittest.skipUnless(os.environ.get("NAVTEX_REAL_DB"), "set NAVTEX_REAL_DB to check a real database")
class RealData(unittest.TestCase):
    def test_every_zczc_becomes_a_message_and_the_percentages_add_up(self):
        path = os.environ["NAVTEX_REAL_DB"]
        connection = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
        expected = sum(t.count("ZCZC") for (t,) in connection.execute("SELECT text_output FROM navtex_log"))
        connection.close()
        if os.environ.get("NAVTEX_REAL_MESSAGES"):
            self.assertEqual(expected, int(os.environ["NAVTEX_REAL_MESSAGES"]))
        results, _ = analyse_path(path)
        self.assertEqual(len(results), expected)
        self.assertEqual(sum(1 for r in results if r.message.end), expected)


def analyse_path(path):
    messages, orphans = report.segment(report.read_lines(path))
    return report.classify(messages), orphans


if __name__ == "__main__":
    unittest.main(verbosity=1)
