use std::collections::HashMap;
use std::io::{self, BufRead, Read};
use std::fs::File;
use std::path::Path;
use std::str::FromStr;
use ark_ff::PrimeField;
use byteorder::{ByteOrder, LittleEndian};
use log::debug;
use num_bigint::BigUint;
use num_traits::{FromBytes, Num};
use regex;
use serde::{Deserialize, Serialize};
use serde_yaml;
use crate::common::{PATTERN_SPDZ_RANDOM_POINT, PATTERN_SPDZ_OUTPUT_SUM, PATTERN_SPDZ_EVAL, PATTERN_SPDZ_RANDOMNESS};
use crate::mpspdz::FormatType::Sfix;

pub trait MPSPDZBinaryInput {
    fn parse<R: Read>(reader: &mut R) -> io::Result<Self> where Self: Sized;
}

impl MPSPDZBinaryInput for u32 {
    fn parse<R: Read>(reader: &mut R) -> io::Result<Self> {

        const length: usize = 4;
        let mut secret = [0u8; length];
        reader.read_exact(&mut secret)?;

        let result = u32::from_le_bytes(secret.try_into().unwrap());

        Ok(result)
    }
}
impl MPSPDZBinaryInput for u64 {
    fn parse<R: Read>(reader: &mut R) -> io::Result<Self> {
        const length: usize = 8;
        let mut secret = [0u8; length];
        reader.read_exact(&mut secret)?;

        let result = u64::from_le_bytes(secret.try_into().unwrap());

        Ok(result)
    }
}
impl MPSPDZBinaryInput for i64 {
    fn parse<R: Read>(reader: &mut R) -> io::Result<Self> {
        const length: usize = 8;
        let mut secret = [0u8; length];
        reader.read_exact(&mut secret)?;

        let result = i64::from_le_bytes(secret.try_into().unwrap());

        Ok(result)
    }
}
impl MPSPDZBinaryInput for f32 {
    fn parse<R: Read>(reader: &mut R) -> io::Result<Self> {
        const length: usize = 4;
        let mut secret = [0u8; length];
        reader.read_exact(&mut secret)?;

        let result = f32::from_le_bytes(secret.try_into().unwrap());

        Ok(result)
    }
}
impl MPSPDZBinaryInput for f64 {
    fn parse<R: Read>(reader: &mut R) -> io::Result<Self> {
        const length: usize = 8;
        let mut secret = [0u8; length];
        reader.read_exact(&mut secret)?;

        let result = f64::from_le_bytes(secret.try_into().unwrap());

        Ok(result)
    }
}

pub fn parse_shares_from_file<P: AsRef<Path>, Tfix: MPSPDZBinaryInput, Tint: MPSPDZBinaryInput>(path: P, format_path: P)
    -> io::Result<Vec<Vec<ShareList<Tfix, Tint>>>> {
    let format_file_res = parse_format(format_path);
    if format_file_res.is_err() {
        println!("Format file not found: {:?}", format_file_res.err().unwrap());
        return Ok(Vec::new());
    }
    let format_file = format_file_res.unwrap();

    let mut file = File::open(path)?;
    let mut all_objects: Vec<Vec<ShareList<Tfix, Tint>>> = Vec::new();

    println!("Reading format file {:?}", format_file);
    for format_obj in format_file {
        println!("Reading format object {:?}", format_obj.object_type);
        let mut shares: Vec<ShareList<Tfix, Tint>> = Vec::new();
        for format in format_obj.items {
            match format.format_type {
                FormatType::Sfix => {
                    let mut fixList = Vec::new();
                    for _ in 0..format.length {
                        let share = Tfix::parse(&mut file)?;
                        fixList.push(share);
                    }
                    shares.push(ShareList::Sfix(fixList));
                },
                FormatType::Sint => {
                    let mut intList = Vec::new();
                    for _ in 0..format.length {
                        let share = Tint::parse(&mut file)?;
                        intList.push(share);
                    }
                    shares.push(ShareList::Sint(intList));
                }
            }
        }
        all_objects.push(shares);
    }

    Ok(all_objects)
}

pub fn parse_randomness_from_log_file<P: AsRef<Path>, E: PrimeField>(path: P, current_player_id: u64) -> io::Result<Vec<(u64, E)>> {
    let mut file = File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut randomnesses: Vec<(u64, E)> = Vec::new();

    for line in reader.lines() {
        let line = line?;

        if let Some((player_id, rand_str)) = parse_line_rand(&line) {
            if player_id != current_player_id {
                debug!("Found evaluation for player {:?} which is not the current player id. Skipping.", player_id);
                continue;
            }
            let rand_biguint = BigUint::from_str_radix(rand_str.as_str(), 10).expect("Failed to parse BigUInt!");
            let rand = E::from(E::BigInt::try_from(rand_biguint).expect(format!("Failed to convert BigUInt to BigInt: {} ee", rand_str).as_str()));
            randomnesses.push((player_id, rand));
        }
    }
    Ok(randomnesses)
}
fn parse_line_rand(line: &str) -> Option<(u64, String)> {
    let re = regex::Regex::new(PATTERN_SPDZ_RANDOMNESS).unwrap();
    let captures = re.captures(line);
    if let Some(captures) = captures {
        let player_id = captures.get(1).unwrap().as_str().parse::<u64>().unwrap();
        let rand_str = captures.get(2).unwrap().as_str().to_string();
        Some((player_id, rand_str))
    } else {
        None
    }
}

pub fn parse_evaluations_from_log_file<P: AsRef<Path>, E: PrimeField>(path: P, n_players: usize) -> io::Result<Vec<Vec<(E, E)>>> {
    let mut file = File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut points: Vec<Vec<(E, E)>> = vec![Vec::new(); n_players];

    for line in reader.lines() {
        let line = line?;
        if let Some((player_id, point1_str, point2_str)) = parse_line_eval(&line) {
            let point1_biguint = BigUint::from_str_radix(point1_str.as_str(), 10).expect("Failed to parse BigUInt!");
            let point2_biguint = BigUint::from_str_radix(point2_str.as_str(), 10).expect("Failed to parse BigUInt!");
            let point1 = E::try_from(E::BigInt::try_from(point1_biguint).expect(format!("Failed to convert BigUInt to BigInt: {} ee", point1_str).as_str()))
                .expect(format!("Failed to convert BigInt to FieldElement: {} ee", point1_str).as_str());
            let point2 = E::try_from(E::BigInt::try_from(point2_biguint).expect(format!("Failed to convert BigUInt to BigInt: {} ee", point2_str).as_str()))
                .expect(format!("Failed to convert BigInt to FieldElement: {} ee", point2_str).as_str());

            points[player_id as usize].push((point1, point2));
        }
    }
    Ok(points)
}

fn parse_line_eval(line: &str) -> Option<(u64, String, String)> {
    let re = regex::Regex::new(PATTERN_SPDZ_EVAL).unwrap();
    let captures = re.captures(line);
    if let Some(captures) = captures {
        let player_id = captures.get(1).unwrap().as_str().parse::<u64>().unwrap();
        let point1_str = captures.get(2).unwrap().as_str().to_string();
        let point2_str = captures.get(3).unwrap().as_str().to_string();
        Some((player_id, point1_str, point2_str))
    } else {
        None
    }
}

enum FormatType {
    Sint,
    Sfix,
}

struct ShareFormat {
    #[serde(rename = "type")]
    format_type: FormatType,
    length: u32,
}

pub enum ShareList<Sfix, Sint> {
    Sfix(Vec<Sfix>),
    Sint(Vec<Sint>),
}

pub enum ShareFormatObjectType {
    M,D,X,Y,
    Test_X,
    Test_Y
}

struct ShareFormatObject {
    items: ShareFormatList,
    object_type: ShareFormatObjectType
}
type ShareFormatObjects = Vec<ShareFormatObject>;
type ShareFormatList = Vec<ShareFormat>;

fn parse_format<P: AsRef<Path>>(format_path: P) -> io::Result<ShareFormatObjects> {

    let file_content = std::fs::read_to_string(format_path)?;
    println!("File content: {:?}", file_content);

    let format_list: ShareFormatObjects = serde_yaml::from_str(&file_content)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok(format_list)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::io::Cursor;
    use ark_bls12_377::Bls12_377;
    use ark_ec::pairing::Pairing;

    #[test]
    fn test_parse_u32() {
        let test_cases = vec![
            (vec![0x78, 0x56, 0x34, 0x12], 0x12345678),
            (vec![0xFF, 0xFF, 0xFF, 0xFF], u32::MAX),
            (vec![0x00, 0x00, 0x00, 0x00], u32::MIN),
        ];

        for (input, expected) in test_cases {
            let mut cursor = Cursor::new(input);
            let parsed = u32::parse(&mut cursor).expect("Failed to parse u32");
            assert_eq!(parsed, expected, "Parsed value did not match expected");
        }
    }

    #[test]
    fn test_parse_u32_incomplete_input() {
        let incomplete_input = vec![0x78, 0x56, 0x34];
        let mut cursor = Cursor::new(incomplete_input);
        let parse_result = u32::parse(&mut cursor);
        assert!(parse_result.is_err(), "Should error on incomplete input");
    }

    #[test]
    fn test_parse_file_inputs() {
        let filename = "/Users/hidde/Documents/PhD/auditing/cryptographic-auditing-mpc/MP-SPDZ/Player-Data/Input-Binary-P0-0";
        let x: Vec<f32> = parse_shares_from_file(filename).expect("Failed to parse shares");
        println!("Size: {:?}", x.len());
        println!("First Value: {:?}", x[0]);
    }

    #[test]
    fn test_parse_file_eval() {
        type E = Bls12_377;
        type F = <E as Pairing>::ScalarField;

        let filename = "/Users/hidde/Documents/PhD/auditing/cryptographic-auditing-mpc/output.txt";
        let x: Vec<(F, F)> = parse_evaluations_from_log_file(filename).expect("Failed to parse file");
        println!("Size: {:?}", x.len());
        println!("First Eval: {:?} {:?}", x[0].0.into_bigint().to_string(), x[0].1.into_bigint().to_string());
    }
}