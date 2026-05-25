//! Lexer for the topology DSL.
//!
//! Hand-rolled because the grammar is tiny and pull-based iteration maps cleanly to it.
//! We keep a `Peekable<Chars>` and drive it character-by-character to avoid the
//! `take_while` / `by_ref` footgun where the adapter borrows the iterator mutably
//! and can leave us past the first digit of a numeric suffix.

/// Tokens produced by the topology DSL lexer.
///
/// Grammar terminal symbols for expressions like `TP8_NVL12+PP4_IB2|TP8_NVL6`.
#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    /// Tensor-parallel dim keyword.
    Tp,
    /// Pipeline-parallel dim keyword.
    Pp,
    /// Data-parallel dim keyword.
    Dp,
    /// Degree integer following a dim keyword.
    Number(u32),
    /// `_NVL<n>`: minimum NVLink bandwidth in GB/s.
    NvlMin(f32),
    /// `_IB<n>`: maximum InfiniBand hops allowed.
    IbMax(u32),
    /// `_NUMA`: NUMA-locality constraint.
    Numa,
    /// `+` conjunction operator.
    Plus,
    /// `|` disjunction operator.
    Pipe,
}

/// Consume a run of alphabetic characters from `chars`.
fn collect_alpha(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut s = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_alphabetic() {
            s.push(c);
            chars.next();
        } else {
            break;
        }
    }
    s
}

/// Consume a run of ASCII digit characters from `chars`.
fn collect_digits(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut s = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            s.push(c);
            chars.next();
        } else {
            break;
        }
    }
    s
}

/// Tokenise a topology DSL string into a flat token stream.
///
/// # Errors
/// Returns `Err(String)` on the first unknown keyword or malformed number.
pub fn lex(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            '+' => {
                tokens.push(Token::Plus);
                chars.next();
            }
            '|' => {
                tokens.push(Token::Pipe);
                chars.next();
            }
            '_' => {
                chars.next(); // consume '_'
                let kw = collect_alpha(&mut chars);
                match kw.as_str() {
                    "NVL" => {
                        let n = collect_digits(&mut chars);
                        if n.is_empty() {
                            return Err("_NVL must be followed by a number".to_string());
                        }
                        let bw: f32 = n.parse().map_err(|_| "bad NVL number".to_string())?;
                        tokens.push(Token::NvlMin(bw));
                    }
                    "IB" => {
                        let n = collect_digits(&mut chars);
                        if n.is_empty() {
                            return Err("_IB must be followed by a number".to_string());
                        }
                        let hops: u32 = n.parse().map_err(|_| "bad IB number".to_string())?;
                        tokens.push(Token::IbMax(hops));
                    }
                    "NUMA" => tokens.push(Token::Numa),
                    other => return Err(format!("unknown constraint _{other}")),
                }
            }
            c if c.is_alphabetic() => {
                // Collect the keyword (e.g. "TP", "PP", "DP") then its degree number.
                let kw = collect_alpha(&mut chars);
                let num_str = collect_digits(&mut chars);
                if num_str.is_empty() {
                    return Err(format!("expected number after {kw}"));
                }
                let num: u32 = num_str
                    .parse()
                    .map_err(|_| format!("expected number after {kw}"))?;
                match kw.as_str() {
                    "TP" => {
                        tokens.push(Token::Tp);
                        tokens.push(Token::Number(num));
                    }
                    "PP" => {
                        tokens.push(Token::Pp);
                        tokens.push(Token::Number(num));
                    }
                    "DP" => {
                        tokens.push(Token::Dp);
                        tokens.push(Token::Number(num));
                    }
                    other => return Err(format!("unknown parallel dim {other}")),
                }
            }
            c => return Err(format!("unexpected character '{c}'")),
        }
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_tp8_nvl12() {
        let tokens = lex("TP8_NVL12").unwrap();
        assert_eq!(
            tokens,
            vec![Token::Tp, Token::Number(8), Token::NvlMin(12.0)]
        );
    }

    #[test]
    fn lex_operators() {
        let tokens = lex("+|").unwrap();
        assert_eq!(tokens, vec![Token::Plus, Token::Pipe]);
    }

    #[test]
    fn lex_unknown_dim_errors() {
        assert!(lex("XP8").is_err());
    }
}
