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
        self.assertEqual([r["header"] for r in rows], ["BA91", "BA91", "GB07", "MZOI"])
        self.assertEqual(rows[3]["header_kind"], "digits_as_letters")
        self.assertEqual(rows[3]["end"], "gap")
        self.assertEqual(rows[2]["noise_kind"], "garbage")
        self.assertEqual((rows[0]["start_id"], rows[0]["end_id"]), ("1", "4"))

    def test_parse_time_accepts_dates_and_minutes(self):
        self.assertEqual(report.parse_time("2026-10-01"), datetime(2026, 10, 1))
        self.assertEqual(report.parse_time("2026-10-01 06:30"), datetime(2026, 10, 1, 6, 30))
        with self.assertRaises(ValueError):
            report.parse_time("yesterday")


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
