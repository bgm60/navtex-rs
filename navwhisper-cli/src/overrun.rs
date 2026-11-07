//! Reporting of audio buffer overruns without flooding the console.
//!
//! An overrun means the sound system had to throw audio away because the
//! program did not collect it in time. They tend to come in clusters, so the
//! first one in a quiet period is reported at once, and the rest of the
//! cluster is counted and reported together when a minute has passed:
//!
//! ```text
//! 2026-10-28 03:14:07 UTC: warning: audio buffer overrun: some audio may have been lost
//! 2026-10-28 03:15:07 UTC: warning: audio buffer overrun: 6 more in the last 60 s
//! ```
//!
//! The reporter does not look at the clock itself. The caller says what time
//! it is, which keeps it simple to test.

use crate::utc::UtcTime;
use std::time::{Duration, SystemTime};

/// How long a cluster is counted for before it is summarised.
pub const WINDOW: Duration = Duration::from_secs(60);

/// `YYYY-MM-DD HH:MM:SS UTC`, the start of every audio warning.
pub fn stamp(now: SystemTime) -> String {
    format!("{} UTC", UtcTime::from_system_time(now).db_stamp())
}

pub struct OverrunReporter {
    window: Duration,
    /// When the first overrun of the current cluster happened.
    started: Option<SystemTime>,
    /// Overruns since then that have not been reported yet.
    pending: u32,
}

impl Default for OverrunReporter {
    fn default() -> Self {
        OverrunReporter::new(WINDOW)
    }
}

impl OverrunReporter {
    pub fn new(window: Duration) -> Self {
        OverrunReporter { window, started: None, pending: 0 }
    }

    /// An overrun has just happened. Returns the lines to show now: the first
    /// overrun of a cluster is shown at once, and so is the summary of an
    /// earlier cluster whose minute has run out.
    pub fn overrun(&mut self, now: SystemTime) -> Vec<String> {
        let mut lines: Vec<String> = self.summary_if_due(now).into_iter().collect();
        if self.started.is_none() {
            self.started = Some(now);
            self.pending = 0;
            lines.push(format!("{}: warning: audio buffer overrun: some audio may have been lost", stamp(now)));
        } else {
            self.pending += 1;
        }
        lines
    }

    /// Called often, so that a summary appears when its minute is up even if
    /// no further overrun happens.
    pub fn tick(&mut self, now: SystemTime) -> Option<String> {
        self.summary_if_due(now)
    }

    /// Called when the audio ends: reports anything still being counted.
    pub fn finish(&mut self, now: SystemTime) -> Option<String> {
        let started = self.started.take()?;
        let count = std::mem::take(&mut self.pending);
        (count > 0).then(|| summary(now, count, now.duration_since(started).unwrap_or_default()))
    }

    fn summary_if_due(&mut self, now: SystemTime) -> Option<String> {
        let started = self.started?;
        // If the clock was set back, the cluster is treated as over.
        let elapsed = now.duration_since(started).unwrap_or(self.window);
        if elapsed < self.window {
            return None;
        }
        self.started = None;
        let count = std::mem::take(&mut self.pending);
        (count > 0).then(|| summary(now, count, elapsed))
    }
}

fn summary(now: SystemTime, count: u32, elapsed: Duration) -> String {
    format!("{}: warning: audio buffer overrun: {count} more in the last {} s", stamp(now), elapsed.as_secs().max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil;
    use std::time::UNIX_EPOCH;

    /// 2026-10-18 17:36:36 UTC plus the given seconds.
    fn at(seconds: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_micros(testutil::START_MICROS) + Duration::from_secs(seconds)
    }

    #[test]
    fn the_first_overrun_is_reported_at_once_with_its_time() {
        let mut reporter = OverrunReporter::default();
        assert_eq!(
            reporter.overrun(at(0)),
            ["2026-10-18 17:36:36 UTC: warning: audio buffer overrun: some audio may have been lost"]
        );
    }

    #[test]
    fn the_rest_of_a_cluster_is_counted_and_summarised_after_a_minute() {
        let mut reporter = OverrunReporter::default();
        reporter.overrun(at(0));
        for second in [1, 2, 3, 20, 59] {
            assert!(reporter.overrun(at(second)).is_empty(), "overrun at {second} s should be counted quietly");
        }
        assert_eq!(reporter.tick(at(59)), None);
        assert_eq!(
            reporter.tick(at(60)),
            Some("2026-10-18 17:37:36 UTC: warning: audio buffer overrun: 5 more in the last 60 s".to_string())
        );
        assert_eq!(reporter.tick(at(61)), None, "a summary is given once");
    }

    #[test]
    fn a_single_overrun_needs_no_summary() {
        let mut reporter = OverrunReporter::default();
        reporter.overrun(at(0));
        assert_eq!(reporter.tick(at(60)), None);
        assert_eq!(reporter.tick(at(600)), None);
    }

    #[test]
    fn after_a_quiet_minute_the_next_overrun_is_reported_at_once_again() {
        let mut reporter = OverrunReporter::default();
        reporter.overrun(at(0));
        assert_eq!(reporter.tick(at(60)), None);
        let lines = reporter.overrun(at(300));
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("2026-10-18 17:41:36 UTC: warning: audio buffer overrun: some audio"), "{lines:?}");
    }

    #[test]
    fn an_overrun_after_the_minute_reports_the_old_cluster_first() {
        // No tick happened in between, so the late overrun does the summarising.
        let mut reporter = OverrunReporter::default();
        reporter.overrun(at(0));
        reporter.overrun(at(5));
        reporter.overrun(at(6));
        let lines = reporter.overrun(at(90));
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines[0].contains("2 more in the last 90 s"), "{lines:?}");
        assert!(lines[1].contains("some audio may have been lost"), "{lines:?}");
    }

    #[test]
    fn the_end_of_the_audio_reports_what_is_still_being_counted() {
        let mut reporter = OverrunReporter::default();
        reporter.overrun(at(0));
        reporter.overrun(at(4));
        reporter.overrun(at(9));
        assert_eq!(
            reporter.finish(at(12)),
            Some("2026-10-18 17:36:48 UTC: warning: audio buffer overrun: 2 more in the last 12 s".to_string())
        );
        assert_eq!(reporter.finish(at(13)), None);
        // and with only one overrun there is nothing to add
        reporter.overrun(at(20));
        assert_eq!(reporter.finish(at(21)), None);
    }

    #[test]
    fn a_clock_set_back_does_not_freeze_the_reporter() {
        let mut reporter = OverrunReporter::default();
        reporter.overrun(at(1000));
        reporter.overrun(at(1001));
        let summary = reporter.tick(at(10)).expect("the cluster is treated as over");
        assert!(summary.contains("1 more in the last 60 s"), "{summary}");
        assert_eq!(reporter.overrun(at(11)).len(), 1);
    }

    #[test]
    fn the_window_can_be_changed() {
        let mut reporter = OverrunReporter::new(Duration::from_secs(5));
        reporter.overrun(at(0));
        reporter.overrun(at(1));
        assert!(reporter.tick(at(4)).is_none());
        assert!(reporter.tick(at(5)).is_some());
    }
}
