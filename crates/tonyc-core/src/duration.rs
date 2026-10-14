//! Human durations such as `45m`, `2h30m`, `90s`, `1d`.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid duration `{input}`: {reason}")]
pub struct DurationError {
    pub input: String,
    pub reason: &'static str,
}

/// Parses a duration into milliseconds. Accepts one or more `<int><unit>`
/// groups with units `ms`, `s`, `m`, `h`, `d`; a bare integer means seconds.
pub fn parse_millis(input: &str) -> Result<i64, DurationError> {
    let err = |reason| DurationError { input: input.to_string(), reason };
    let s = input.trim();
    if s.is_empty() {
        return Err(err("empty"));
    }
    if let Ok(secs) = s.parse::<i64>() {
        return secs.checked_mul(1000).filter(|v| *v > 0).ok_or(err("must be positive"));
    }
    let bytes = s.as_bytes();
    let mut total: i64 = 0;
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if start == i {
            return Err(err("expected a number"));
        }
        let n: i64 = s[start..i].parse().map_err(|_| err("number too large"))?;
        let ustart = i;
        while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
            i += 1;
        }
        let unit = match &s[ustart..i] {
            "ms" => 1,
            "s" => 1_000,
            "m" => 60_000,
            "h" => 3_600_000,
            "d" => 86_400_000,
            "" => return Err(err("missing unit (ms, s, m, h, d)")),
            _ => return Err(err("unknown unit (use ms, s, m, h, d)")),
        };
        total = n.checked_mul(unit).and_then(|v| total.checked_add(v)).ok_or(err("too large"))?;
    }
    if total <= 0 {
        return Err(err("must be positive"));
    }
    Ok(total)
}

/// Formats milliseconds compactly: `7m12s`, `450ms`, `2h`.
pub struct Human(pub i64);

impl fmt::Display for Human {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ms = self.0;
        if ms < 0 {
            return write!(f, "-{}", Human(-ms));
        }
        if ms < 1000 {
            return write!(f, "{ms}ms");
        }
        let mut secs = ms / 1000;
        let parts = [(86_400, "d"), (3_600, "h"), (60, "m"), (1, "s")];
        let mut wrote = false;
        for (size, unit) in parts {
            if secs >= size {
                write!(f, "{}{unit}", secs / size)?;
                secs %= size;
                wrote = true;
            }
        }
        if !wrote {
            write!(f, "0s")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_units_and_compounds() {
        assert_eq!(parse_millis("45m"), Ok(45 * 60_000));
        assert_eq!(parse_millis("2h30m"), Ok(150 * 60_000));
        assert_eq!(parse_millis("90"), Ok(90_000));
        assert_eq!(parse_millis("250ms"), Ok(250));
        assert_eq!(parse_millis("1d"), Ok(86_400_000));
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_millis("").is_err());
        assert!(parse_millis("10x").is_err());
        assert!(parse_millis("m").is_err());
        assert!(parse_millis("0s").is_err());
        assert!(parse_millis("-5").is_err());
    }

    #[test]
    fn formats() {
        assert_eq!(Human(432_000).to_string(), "7m12s");
        assert_eq!(Human(450).to_string(), "450ms");
        assert_eq!(Human(7_200_000).to_string(), "2h");
    }
}
