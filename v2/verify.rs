#!/usr/bin/env cargo
---
[package]
edition = "2024"

[dependencies]
shakmaty = "0.30"

[profile.dev]
opt-level = 3
debug = false
debug-assertions = false
overflow-checks = false
---

use shakmaty::{
    attacks::{bishop_attacks, rook_attacks},
    Bitboard, File, Rank, Role, Square,
};

use std::fmt;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct VerifiedMagic {
    max_index: usize,
    magic: u64,
}

impl fmt::Debug for VerifiedMagic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "magic: {:#x} (max index: {})",
            self.magic, self.max_index
        )
    }
}

struct Reference {
    occupied: Bitboard,
    attacks: Bitboard,
}

struct Scratchpad {
    real_shift: u32,
    references: Vec<Reference>,
    attacks: [Bitboard; 1 << 12],
    age: [u64; 1 << 12],
}

impl Scratchpad {
    pub fn new(role: Role, sq: Square, shift: u32) -> Self {
        let attacks = |occupied: Bitboard| -> Bitboard {
            match role {
                Role::Bishop => bishop_attacks(sq, occupied),
                Role::Rook => rook_attacks(sq, occupied),
                _ => panic!("bishops/rooks only"),
            }
        };

        let full_attacks = attacks(Bitboard::EMPTY);

        let edges = ((Bitboard::from(Rank::First) | Bitboard::from(Rank::Eighth))
            & !Bitboard::from(sq.rank()))
            | ((Bitboard::from(File::A) | Bitboard::from(File::H))
                & !Bitboard::from(sq.file()));

        let mask = full_attacks & !edges;

        Self {
            real_shift: 64 - shift,
            references: mask.carry_rippler().map(|occupied| Reference {
                occupied,
                attacks: attacks(occupied),
            }).collect(),
            attacks: [Bitboard::EMPTY; 1 << 12],
            age: [0; 1 << 12],
        }
    }

    fn test(&mut self, magic: u64) -> Option<VerifiedMagic> {
        let mut max_index = 0;

        for reference in &self.references {
            let index = (u64::from(reference.occupied).wrapping_mul(magic) >> self.real_shift) as usize;
            max_index = max_index.max(index);
            if self.age[index] != magic {
                self.attacks[index] = reference.attacks;
                self.age[index] = magic;
            } else if self.attacks[index] != reference.attacks {
                return None;
            }
        }

        Some(VerifiedMagic { magic, max_index })
    }
}

fn main() {
    let role = Role::Rook;
    let sq = Square::G3;
    let c = 10;

    //let mut fixed_shift = Scratchpad::new(role, sq, 12);
    //let mut best_fixed: Option<VerifiedMagic> = None;

    let mut easy_shift = Scratchpad::new(role, sq, c);
    let mut best_easy: Option<VerifiedMagic> = None;

    let mut hard_shift = Scratchpad::new(role, sq, c - 1);
    let mut best_hard: Option<VerifiedMagic> = None;

    let mut seen_candidates = 0;
    let mut formatting_errors = 0;

    for line in std::io::stdin().lines() {
        let line = line.unwrap();
        if let Some(hex) = line.strip_prefix("0x") {
            let candidate = match u64::from_str_radix(hex, 16) {
                Ok(candidate) => candidate,
                Err(err) => {
                    println!("{err}: {line:?}");
                    formatting_errors += 1;
                    continue;
                }
            };

            seen_candidates += 1;

            //let verified = fixed_shift.test(candidate).unwrap_or_else(|| panic!("not a fixed shift magic: {candidate:#x}"));
            //if best_fixed.is_none_or(|b| verified < b) {
            //    println!("improved fixed: {:?}", verified);
            //    best_fixed = Some(verified);
            //}

            let verified = easy_shift.test(candidate).unwrap_or_else(|| panic!("not an easy shift magic: {candidate:#x}"));
            if best_easy.is_none_or(|b| verified < b) {
                println!("improved easy (c = {}): {:?}", c, verified);
                best_easy = Some(verified);
            }

            if let Some(verified) = hard_shift.test(candidate) &&
                best_hard.is_none_or(|b| verified < b)
            {
                println!("improved hard (c - 1 = {}): {:?}", c - 1, verified);
                best_hard = Some(verified);
            }
        }
    }

    println!("--- {}{} ---", role.upper_char(), sq);
    println!("seen candidates (no deduplication): {}", seen_candidates);
    println!("formatting errors: {}", formatting_errors);
    //if let Some(best_fixed) = best_fixed {
    //    println!("best fixed: {:?}", best_fixed);
    //}
    if let Some(best_easy) = best_easy {
        println!("best easy (c = {}): {:?}", c, best_easy);
    }
    if let Some(best_hard) = best_hard {
        println!("best hard (c - 1 = {}): {:?}", c - 1, best_hard);
    }
    println!("goodbye")
}
