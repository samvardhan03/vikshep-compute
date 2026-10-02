//! Canonical tensor hashing (`spec/VDS-1.md` section 9).
//!
//! An object identifier (OID) is the lowercase hexadecimal encoding of the
//! first 14 bytes of the SHA3-256 digest of a tensor's canonical bytes
//! (little-endian, row-major, contiguous): 28 hex characters. This matches
//! the existing Vikshep contract and must not change.

use sha3::{Digest, Sha3_256};

/// Number of digest bytes kept in an OID.
pub const OID_BYTES: usize = 14;

/// Full SHA3-256 digest.
#[must_use]
pub fn sha3_256(bytes: &[u8]) -> [u8; 32] {
    Sha3_256::digest(bytes).into()
}

/// Lowercase hexadecimal encoding.
#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from(DIGITS[usize::from(b >> 4)]));
        s.push(char::from(DIGITS[usize::from(b & 0x0f)]));
    }
    s
}

/// OID of raw canonical tensor bytes.
#[must_use]
pub fn oid(bytes: &[u8]) -> String {
    hex(&sha3_256(bytes)[..OID_BYTES])
}

/// Canonical bytes of a binary32 tensor (little-endian, in the given order).
#[must_use]
pub fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha3_256_known_answers() {
        // FIPS 202 / NIST example values for SHA3-256.
        assert_eq!(
            hex(&sha3_256(b"")),
            "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
        );
        assert_eq!(
            hex(&sha3_256(b"abc")),
            "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
        );
    }

    #[test]
    fn oid_is_28_hex_chars_of_digest_prefix() {
        assert_eq!(oid(b""), "a7ffc6f8bf1ed76651c14756a061");
        assert_eq!(oid(b"abc").len(), 28);
        let bytes = f32_le_bytes(&[1.0, -2.5]);
        assert_eq!(bytes, [0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0xc0]);
    }
}
