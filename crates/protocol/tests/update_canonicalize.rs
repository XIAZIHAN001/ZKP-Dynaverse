//! Update canonicalization tests (day-1 scope; protocol update tests come day 5).
use ark_bn254::Fr;
use dynaverse_protocol::Update;

#[test]
fn update_canonicalize_sorts_dedups_and_rejects_out_of_range() {
    let u = Update {
        indices: vec![9, 3, 3, 0],
        values: vec![Fr::from(9u64), Fr::from(3u64), Fr::from(33u64), Fr::from(0u64)],
    };
    let c = u.canonicalize(10).unwrap();
    assert_eq!(c.indices, vec![0, 3, 9]);
    assert_eq!(c.values, vec![Fr::from(0u64), Fr::from(33u64), Fr::from(9u64)]);
    assert_eq!(c.k(), 3);

    let bad = Update {
        indices: vec![10],
        values: vec![Fr::from(1u64)],
    };
    assert!(bad.canonicalize(10).is_err());

    let mismatch = Update {
        indices: vec![1, 2],
        values: vec![Fr::from(1u64)],
    };
    assert!(mismatch.canonicalize(10).is_err());
}
