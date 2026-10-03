//! RFC 8785 number formatting against ECMAScript (`String(x)` in Node),
//! fixture from `oracles/jcs_numbers/gen.sh`.

#[test]
fn matches_ecmascript_number_to_string() {
    let text = include_str!("fixtures/ecmascript_numbers.txt");
    let mut n = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let (bits, want) = line.split_once(' ').unwrap();
        let x = f64::from_bits(u64::from_str_radix(bits, 16).unwrap());
        assert_eq!(
            vikshep_numerics::jcs::format_number(x).unwrap(),
            want,
            "{bits}"
        );
        n += 1;
    }
    assert!(n > 5000);
}
