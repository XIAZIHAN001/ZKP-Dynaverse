use ark_bn254::Fr;
use dynaverse_protocol::domain::{
    interpolate_at_indices, pad_to_power_of_two, radix2_domain, vanishing_derivative_values,
    vanishing_polynomial,
};

#[test]
fn domain_rejects_non_power_of_two_sizes() {
    assert!(radix2_domain(0).is_err());
    assert!(radix2_domain(3).is_err());
    assert!(radix2_domain(4).is_ok());
}

#[test]
fn vanishing_polynomial_and_derivative_match_two_roots() {
    let domain = radix2_domain(4).unwrap();
    // Domain points 0 and 2 are +1 and -1, so A_I(X) = X^2 - 1.
    let a = vanishing_polynomial(&domain, &[0, 2]).unwrap();
    assert_eq!(a, vec![-Fr::from(1u64), Fr::from(0u64), Fr::from(1u64)]);
    assert_eq!(
        vanishing_derivative_values(&domain, &[0, 2]).unwrap(),
        vec![Fr::from(2u64), -Fr::from(2u64)]
    );
    assert_eq!(
        vanishing_polynomial(&domain, &[]).unwrap(),
        vec![Fr::from(1u64)]
    );
}

#[test]
fn interpolation_recovers_a_hand_calculated_line() {
    let domain = radix2_domain(4).unwrap();
    // R(1)=5, R(-1)=1 implies R(X)=3+2X.
    let r = interpolate_at_indices(&domain, &[0, 2], &[Fr::from(5u64), Fr::from(1u64)]).unwrap();
    assert_eq!(r, vec![Fr::from(3u64), Fr::from(2u64)]);
    assert_eq!(
        interpolate_at_indices(&domain, &[], &[]).unwrap(),
        vec![Fr::from(0u64)]
    );
}

#[test]
fn domain_inputs_reject_duplicates_and_out_of_range_indices() {
    let domain = radix2_domain(4).unwrap();
    assert!(vanishing_polynomial(&domain, &[1, 1]).is_err());
    assert!(vanishing_polynomial(&domain, &[4]).is_err());
    assert!(interpolate_at_indices(&domain, &[0, 2], &[Fr::from(1u64)]).is_err());
}

#[test]
fn padding_is_explicit_and_keeps_existing_values() {
    let input = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
    assert_eq!(
        pad_to_power_of_two(&input).unwrap(),
        vec![
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(0u64)
        ]
    );
    assert_eq!(pad_to_power_of_two(&[]).unwrap(), vec![Fr::from(0u64)]);
}
