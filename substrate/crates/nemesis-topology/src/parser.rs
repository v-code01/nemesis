//! Recursive-descent parser for the topology DSL.
//!
//! Grammar (EBNF):
//! ```text
//! spec        := disjunction
//! disjunction := conjunction ('|' conjunction)*
//! conjunction := atom ('+' atom)*
//! atom        := dim_kw number constraint*
//! dim_kw      := 'TP' | 'PP' | 'DP'
//! constraint  := NvlMin | IbMax | Numa
//! ```

use crate::lexer::{lex, Token};

/// A parallelism dimension with its degree.
#[derive(Debug, PartialEq, Clone)]
pub enum ParallelDim {
    /// Tensor parallelism — all-reduce inside NVLink domain.
    Tp(u32),
    /// Pipeline parallelism — point-to-point across IB.
    Pp(u32),
    /// Data parallelism — gradient all-reduce.
    Dp(u32),
}

/// Hardware placement / bandwidth constraint attached to a dim.
#[derive(Debug, PartialEq, Clone)]
pub enum Constraint {
    /// Minimum NVLink bandwidth required in GB/s.
    NvlMin(f32),
    /// Maximum InfiniBand hop count permitted.
    IbMax(u32),
    /// All ranks must be NUMA-local.
    Numa,
}

/// A typed topology specification node.
#[derive(Debug, PartialEq, Clone)]
pub enum TopologySpec {
    /// A single parallelism axis with zero or more constraints.
    Atom(ParallelDim, Vec<Constraint>),
    /// Both axes are simultaneously required (e.g. TP inside PP).
    Conjunction(Box<TopologySpec>, Box<TopologySpec>),
    /// Any one of the alternatives is acceptable.
    Disjunction(Vec<TopologySpec>),
}

impl TopologySpec {
    /// Total GPU count implied by this spec.
    ///
    /// - Atom: degree of its dim.
    /// - Conjunction: product of left × right counts.
    /// - Disjunction: count of the first alternative (all must agree — enforced by type-checker).
    pub fn gpu_count(&self) -> u32 {
        match self {
            Self::Atom(dim, _) => match dim {
                ParallelDim::Tp(n) | ParallelDim::Pp(n) | ParallelDim::Dp(n) => *n,
            },
            Self::Conjunction(l, r) => l.gpu_count() * r.gpu_count(),
            Self::Disjunction(alts) => alts.first().map(|a| a.gpu_count()).unwrap_or(0),
        }
    }
}

/// Parse a topology DSL string into a `TopologySpec`.
///
/// # Errors
/// Returns `Err(String)` if the input is lexically or syntactically invalid.
pub fn parse(input: &str) -> Result<TopologySpec, String> {
    let tokens = lex(input)?;
    let mut pos = 0usize;
    let spec = parse_disjunction(&tokens, &mut pos)?;
    if pos != tokens.len() {
        return Err(format!(
            "unexpected tokens starting at position {pos}: {:?}",
            &tokens[pos..]
        ));
    }
    Ok(spec)
}

/// Parse a disjunction: one or more conjunctions separated by `|`.
fn parse_disjunction(tokens: &[Token], pos: &mut usize) -> Result<TopologySpec, String> {
    let first = parse_conjunction(tokens, pos)?;
    let mut alts = vec![first];
    while *pos < tokens.len() && tokens[*pos] == Token::Pipe {
        *pos += 1;
        alts.push(parse_conjunction(tokens, pos)?);
    }
    if alts.len() == 1 {
        Ok(alts.remove(0))
    } else {
        Ok(TopologySpec::Disjunction(alts))
    }
}

/// Parse a conjunction: one or two atoms separated by `+`.
///
/// Conjunction is left-associative and binary for now; chaining would require
/// a loop, but the DSL currently describes two-axis (TP × PP) layouts.
fn parse_conjunction(tokens: &[Token], pos: &mut usize) -> Result<TopologySpec, String> {
    let first = parse_atom(tokens, pos)?;
    if *pos < tokens.len() && tokens[*pos] == Token::Plus {
        *pos += 1;
        let second = parse_atom(tokens, pos)?;
        Ok(TopologySpec::Conjunction(Box::new(first), Box::new(second)))
    } else {
        Ok(first)
    }
}

/// Parse an atom: dim keyword + degree + zero or more constraints.
fn parse_atom(tokens: &[Token], pos: &mut usize) -> Result<TopologySpec, String> {
    let dim = match tokens.get(*pos) {
        Some(Token::Tp) => {
            *pos += 1;
            ParallelDim::Tp(expect_number(tokens, pos)?)
        }
        Some(Token::Pp) => {
            *pos += 1;
            ParallelDim::Pp(expect_number(tokens, pos)?)
        }
        Some(Token::Dp) => {
            *pos += 1;
            ParallelDim::Dp(expect_number(tokens, pos)?)
        }
        other => {
            return Err(format!(
                "expected TP/PP/DP at position {pos}, got {other:?}"
            ))
        }
    };

    let mut constraints = Vec::new();
    while let Some(tok) = tokens.get(*pos) {
        match tok {
            Token::NvlMin(bw) => {
                constraints.push(Constraint::NvlMin(*bw));
                *pos += 1;
            }
            Token::IbMax(h) => {
                constraints.push(Constraint::IbMax(*h));
                *pos += 1;
            }
            Token::Numa => {
                constraints.push(Constraint::Numa);
                *pos += 1;
            }
            // Any other token ends this atom's constraint list.
            _ => break,
        }
    }
    Ok(TopologySpec::Atom(dim, constraints))
}

/// Consume a `Number` token and return its value.
fn expect_number(tokens: &[Token], pos: &mut usize) -> Result<u32, String> {
    match tokens.get(*pos) {
        Some(Token::Number(n)) => {
            let v = *n;
            *pos += 1;
            Ok(v)
        }
        other => Err(format!(
            "expected degree number at position {pos}, got {other:?}"
        )),
    }
}
