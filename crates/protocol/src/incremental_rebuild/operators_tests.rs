use super::*;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup, Group, VariableBaseMSM};
use ark_ff::{One, Zero};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

fn values(len: usize) -> Vec<Fr> {
    (0..len).map(|i| Fr::from((i * i + 3 * i + 11) as u64)).collect()
}

fn points(len: usize) -> Vec<G1Affine> {
    values(len).iter().map(|x| (G1Projective::generator() * x).into_affine()).collect()
}

fn drain(plan: &mut FourStepPlan, backend: &mut CpuRebuildBackend) -> usize {
    let mut calls = 0;
    loop {
        calls += 1;
        if plan.next_step(backend).unwrap() {
            return calls;
        }
        assert!(calls < 10_000);
    }
}

// Removing a transpose or using the wrong twiddle/root must change this result.
#[test]
fn four_step_fields_match_natural_arkworks_forward_and_inverse() {
    for len in [1, 2, 4, 8, 16, 32, 64, 128] {
        for inverse in [false, true] {
            let mut backend = CpuRebuildBackend::new();
            let input = backend.alloc(BufferKind::Fr, len, false).unwrap();
            let scratch = backend.alloc(BufferKind::Fr, len, true).unwrap();
            let output = backend.alloc(BufferKind::Fr, len, true).unwrap();
            let source = values(len);
            backend.write_fields(input, 0, &source).unwrap();
            backend.seal();
            let mut plan = FourStepPlan::new(BufferKind::Fr, len, input, scratch, output, 3, 2, inverse).unwrap();
            let expected_steps = plan.total_steps();
            assert_eq!(drain(&mut plan, &mut backend), expected_steps);
            let domain = Radix2EvaluationDomain::<Fr>::new(len).unwrap();
            let expected = if inverse { domain.ifft(&source) } else { domain.fft(&source) };
            let mut actual = vec![Fr::zero(); len];
            backend.read_fields(output, 0, &mut actual).unwrap();
            assert_eq!(actual, expected, "len={len}, inverse={inverse}");
            assert_eq!(backend.stats().fr_transform_elements, (2 * len) as u64);
        }
    }
}

// Group transforms must act on scalars identically and normalize the inverse once.
#[test]
fn four_step_points_match_arkworks_group_fft() {
    for len in [1, 2, 4, 8, 16, 32, 64, 128] {
        for inverse in [false, true] {
            let mut backend = CpuRebuildBackend::new();
            let input = backend.alloc(BufferKind::G1, len, false).unwrap();
            let scratch = backend.alloc(BufferKind::G1, len, true).unwrap();
            let output = backend.alloc(BufferKind::G1, len, true).unwrap();
            let source = points(len);
            backend.write_points(input, 0, &source).unwrap();
            backend.seal();
            let mut plan = FourStepPlan::new(BufferKind::G1, len, input, scratch, output, 3, 2, inverse).unwrap();
            let expected_steps = plan.total_steps();
            assert_eq!(drain(&mut plan, &mut backend), expected_steps);
            let domain = Radix2EvaluationDomain::<Fr>::new(len).unwrap();
            let mut expected: Vec<G1Projective> = source.iter().map(|p| p.into_group()).collect();
            if inverse { domain.ifft_in_place(&mut expected); } else { domain.fft_in_place(&mut expected); }
            let expected = G1Projective::normalize_batch(&expected);
            let mut actual = vec![G1Affine::zero(); len];
            backend.read_points(output, 0, &mut actual).unwrap();
            assert_eq!(actual, expected, "len={len}, inverse={inverse}");
            assert_eq!(backend.stats().g1_transform_elements, (2 * len) as u64);
        }
    }
}

// Row batching must leave rows outside the selected interval unchanged.
#[test]
fn transform_rows_respects_selected_batch() {
    let mut backend = CpuRebuildBackend::new();
    let source = values(16);
    let input = backend.alloc(BufferKind::Fr, 16, false).unwrap();
    let output = backend.alloc(BufferKind::Fr, 16, true).unwrap();
    backend.write_fields(input, 0, &source).unwrap();
    backend.seal();
    backend.transform_rows(BufferKind::Fr, input, output, 4, 1, 2, false).unwrap();
    let domain = Radix2EvaluationDomain::<Fr>::new(4).unwrap();
    let mut expected = vec![Fr::zero(); 16];
    expected[4..8].copy_from_slice(&domain.fft(&source[4..8]));
    expected[8..12].copy_from_slice(&domain.fft(&source[8..12]));
    let mut actual = vec![Fr::zero(); 16];
    backend.read_fields(output, 0, &mut actual).unwrap();
    assert_eq!(actual, expected);
}

// A rollback must restore the checkpoint even when the same entries were written twice.
#[test]
fn transaction_restores_multiple_operations_and_commit_preserves_new_checkpoint() {
    let mut backend = CpuRebuildBackend::new();
    let fields = backend.alloc(BufferKind::Fr, 16, true).unwrap();
    let other = backend.alloc(BufferKind::Fr, 16, true).unwrap();
    let group = backend.alloc(BufferKind::G1, 16, true).unwrap();
    let group_out = backend.alloc(BufferKind::G1, 16, true).unwrap();
    let source = values(16);
    let source_points = points(16);
    backend.write_fields(fields, 0, &source).unwrap();
    backend.write_fields(other, 0, &source).unwrap();
    backend.write_points(group, 0, &source_points).unwrap();
    backend.write_points(group_out, 0, &source_points).unwrap();
    backend.seal();
    backend.begin_transaction().unwrap();
    backend.write_fields(fields, 3, &[Fr::one(); 3]).unwrap();
    backend.copy_buffer(other, 2, fields, 4, 3).unwrap();
    backend.field_zip(FieldBinaryOp::Mul, fields, 3, other, 3, fields, 3, 3).unwrap();
    backend.scale_points(group, 0, fields, 0, group_out, 0, 5).unwrap();
    assert_eq!(backend.transaction_journal_len(), 9);
    backend.rollback_transaction().unwrap();
    let mut actual = vec![Fr::zero(); 16];
    backend.read_fields(fields, 0, &mut actual).unwrap();
    assert_eq!(actual, source);
    let mut actual_points = vec![G1Affine::zero(); 16];
    backend.read_points(group_out, 0, &mut actual_points).unwrap();
    assert_eq!(actual_points, source_points);
    backend.begin_transaction().unwrap();
    backend.write_fields(fields, 7, &[Fr::one()]).unwrap();
    backend.commit_transaction();
    backend.begin_transaction().unwrap();
    backend.write_fields(fields, 7, &[Fr::from(19u64)]).unwrap();
    backend.rollback_transaction().unwrap();
    backend.read_fields(fields, 7, &mut actual[..1]).unwrap();
    assert_eq!(actual[0], Fr::one());
}

// Invalid aliases, ranges, kinds and read-only writes must fail before changing output.
#[test]
fn validation_and_seal_reject_invalid_mutation_without_partial_writes() {
    let mut backend = CpuRebuildBackend::new();
    let fields = backend.alloc(BufferKind::Fr, 8, true).unwrap();
    let readonly = backend.alloc(BufferKind::Fr, 8, false).unwrap();
    let group = backend.alloc(BufferKind::G1, 8, true).unwrap();
    let source = values(8);
    backend.write_fields(fields, 0, &source).unwrap();
    backend.seal();
    assert!(backend.alloc(BufferKind::Fr, 1, true).is_err());
    assert!(backend.write_fields(readonly, 0, &[Fr::one()]).is_err());
    assert!(backend.write_fields(fields, usize::MAX, &[Fr::one()]).is_err());
    assert!(backend.write_fields(group, 0, &[Fr::one()]).is_err());
    assert!(backend.transpose(BufferKind::Fr, fields, fields, 2, 4, 0, 8, None).is_err());
    assert!(backend.field_zip(FieldBinaryOp::Add, fields, 0, readonly, 0, fields, 1, 7).is_err());
    assert!(backend.copy_buffer(fields, 0, fields, 1, 7).is_err());
    assert!(backend.transform_rows(BufferKind::Fr, fields, readonly, 4, 0, 2, false).is_err());
    assert!(backend.transform_rows(BufferKind::Fr, fields, fields, 3, 0, 2, false).is_err());
    let mut actual = vec![Fr::zero(); 8];
    backend.read_fields(fields, 0, &mut actual).unwrap();
    assert_eq!(actual, source);
}

// Nonzero scalar and point offsets must be honored by both group MSMs.
#[test]
fn resident_field_operations_point_scaling_and_msms_match_arkworks() {
    let mut backend = CpuRebuildBackend::new();
    let fields = backend.alloc(BufferKind::Fr, 16, false).unwrap();
    let output = backend.alloc(BufferKind::Fr, 16, true).unwrap();
    let group = backend.alloc(BufferKind::G1, 16, false).unwrap();
    let group_out = backend.alloc(BufferKind::G1, 16, true).unwrap();
    let source = values(16);
    let source_points = points(16);
    backend.write_fields(fields, 0, &source).unwrap();
    backend.write_points(group, 0, &source_points).unwrap();
    backend.seal();
    for op in [FieldBinaryOp::Add, FieldBinaryOp::Sub, FieldBinaryOp::Mul] {
        backend.field_zip(op, fields, 2, fields, 5, output, 3, 4).unwrap();
        let expected: Vec<Fr> = (0..4).map(|i| match op {
            FieldBinaryOp::Add => source[i+2] + source[i+5],
            FieldBinaryOp::Sub => source[i+2] - source[i+5],
            FieldBinaryOp::Mul => source[i+2] * source[i+5],
        }).collect();
        let mut actual = vec![Fr::zero(); 4];
        backend.read_fields(output, 3, &mut actual).unwrap();
        assert_eq!(actual, expected);
    }
    backend.scale_points(group, 3, fields, 5, group_out, 1, 4).unwrap();
    let mut actual = vec![G1Affine::zero(); 4];
    backend.read_points(group_out, 1, &mut actual).unwrap();
    for i in 0..4 { assert_eq!(actual[i], (source_points[i+3] * source[i+5]).into_affine()); }
    let expected = G1Projective::msm_unchecked(&source_points[3..7], &source[5..9]).into_affine();
    assert_eq!(backend.msm_g1(group, 3, fields, 5, 4).unwrap(), expected);
    let g2: Vec<G2Affine> = source[1..5].iter().map(|x| (G2Projective::generator() * x).into_affine()).collect();
    let expected_g2 = G2Projective::msm_unchecked(&g2, &source[5..9]).into_affine();
    assert_eq!(backend.msm_g2(&g2, fields, 5).unwrap(), expected_g2);
    assert_eq!(backend.stats().msm_g1_points, 4);
    assert_eq!(backend.stats().msm_g2_points, 4);
}

#[test]
fn partial_transpose_twiddle_uses_original_source_coordinates() {
    let mut backend = CpuRebuildBackend::new();
    let fields = backend.alloc(BufferKind::Fr, 8, false).unwrap();
    let output = backend.alloc(BufferKind::Fr, 8, true).unwrap();
    let source = values(8);
    let root = Radix2EvaluationDomain::<Fr>::new(8).unwrap().group_gen;
    backend.write_fields(fields, 0, &source).unwrap();
    backend.seal();
    backend.transpose(BufferKind::Fr, fields, output, 2, 4, 3, 3, Some(root)).unwrap();
    let mut actual = vec![Fr::zero(); 8];
    backend.read_fields(output, 0, &mut actual).unwrap();
    let mut expected = vec![Fr::zero(); 8];
    expected[6] = source[3];
    expected[1] = source[4];
    expected[3] = source[5] * root;
    assert_eq!(actual, expected);
}

// A handle from another resident owner must never mutate an unrelated pool.
#[test]
fn buffer_handles_are_bound_to_the_allocating_owner() {
    let mut first = CpuRebuildBackend::new();
    let mut second = CpuRebuildBackend::new();
    let first_id = first.alloc(BufferKind::Fr, 4, true).unwrap();
    let second_id = second.alloc(BufferKind::Fr, 4, true).unwrap();
    assert!(second.write_fields(first_id, 0, &[Fr::one()]).is_err());
    assert!(first.write_fields(second_id, 0, &[Fr::one()]).is_err());
    let mut actual = [Fr::zero(); 4];
    second.read_fields(second_id, 0, &mut actual).unwrap();
    assert_eq!(actual, [Fr::zero(); 4]);
}

// Offset or reversal mistakes affect polynomial convolution prefixes.
#[test]
fn reverse_copy_respects_both_offsets_for_fields_and_points() {
    let mut backend = CpuRebuildBackend::new();
    let fields = backend.alloc(BufferKind::Fr, 8, false).unwrap();
    let reversed_fields = backend.alloc(BufferKind::Fr, 8, true).unwrap();
    let group = backend.alloc(BufferKind::G1, 8, false).unwrap();
    let reversed_group = backend.alloc(BufferKind::G1, 8, true).unwrap();
    let source = values(8);
    let source_points = points(8);
    backend.write_fields(fields, 0, &source).unwrap();
    backend.write_points(group, 0, &source_points).unwrap();
    backend.seal();
    backend.reverse_copy(BufferKind::Fr, fields, 2, reversed_fields, 3, 4).unwrap();
    backend.reverse_copy(BufferKind::G1, group, 2, reversed_group, 3, 4).unwrap();
    let mut actual = [Fr::zero(); 4];
    let mut actual_points = [G1Affine::zero(); 4];
    backend.read_fields(reversed_fields, 3, &mut actual).unwrap();
    backend.read_points(reversed_group, 3, &mut actual_points).unwrap();
    assert_eq!(actual, [source[5], source[4], source[3], source[2]]);
    assert_eq!(actual_points, [source_points[5], source_points[4], source_points[3], source_points[2]]);
    assert!(backend.reverse_copy(BufferKind::Fr, reversed_fields, 0, reversed_fields, 0, 8).is_err());
}

// Empty bounded work may occur at convolution tails and must be harmless.
#[test]
fn zero_length_operations_and_empty_msms_are_safe() {
    let mut backend = CpuRebuildBackend::new();
    let fields = backend.alloc(BufferKind::Fr, 8, true).unwrap();
    let other = backend.alloc(BufferKind::Fr, 8, true).unwrap();
    let group = backend.alloc(BufferKind::G1, 8, true).unwrap();
    backend.seal();
    backend.begin_transaction().unwrap();
    backend.write_fields(fields, 8, &[]).unwrap();
    backend.read_fields(fields, 8, &mut []).unwrap();
    backend.write_points(group, 8, &[]).unwrap();
    backend.read_points(group, 8, &mut []).unwrap();
    backend.copy_buffer(fields, 8, fields, 8, 0).unwrap();
    backend.reverse_copy(BufferKind::G1, group, 8, group, 8, 0).unwrap();
    backend.transpose(BufferKind::Fr, fields, fields, 2, 4, 8, 0, None).unwrap();
    backend.transform_rows(BufferKind::Fr, fields, fields, 4, 2, 0, false).unwrap();
    backend.field_zip(FieldBinaryOp::Mul, fields, 8, other, 8, fields, 8, 0).unwrap();
    backend.scale_points(group, 8, fields, 8, group, 8, 0).unwrap();
    assert_eq!(backend.msm_g1(group, 8, fields, 8, 0).unwrap(), G1Affine::zero());
    assert_eq!(backend.msm_g2(&[], fields, 8).unwrap(), G2Affine::zero());
    assert_eq!(backend.transaction_journal_len(), 0);
    backend.rollback_transaction().unwrap();
}

// Reusing an advanced cursor after whole-call rollback would skip required work.
#[test]
fn four_step_committed_cursor_can_be_retried_after_call_rollback() {
    let mut backend = CpuRebuildBackend::new();
    let input = backend.alloc(BufferKind::Fr, 32, false).unwrap();
    let scratch = backend.alloc(BufferKind::Fr, 32, true).unwrap();
    let output = backend.alloc(BufferKind::Fr, 32, true).unwrap();
    let source = values(32);
    backend.write_fields(input, 0, &source).unwrap();
    backend.seal();
    let committed = FourStepPlan::new(BufferKind::Fr, 32, input, scratch, output, 3, 2, false).unwrap();
    backend.begin_transaction().unwrap();
    let mut tentative = committed.clone();
    for _ in 0..15 { assert!(!tentative.next_step(&mut backend).unwrap()); }
    assert!(backend.write_fields(output, 32, &[Fr::one()]).is_err());
    backend.rollback_transaction().unwrap();
    let mut actual = vec![Fr::zero(); 32];
    backend.read_fields(output, 0, &mut actual).unwrap();
    assert_eq!(actual, vec![Fr::zero(); 32]);
    backend.begin_transaction().unwrap();
    let mut retry = committed;
    drain(&mut retry, &mut backend);
    backend.commit_transaction();
    backend.read_fields(output, 0, &mut actual).unwrap();
    let expected = Radix2EvaluationDomain::<Fr>::new(32).unwrap().fft(&source);
    assert_eq!(actual, expected);
}
