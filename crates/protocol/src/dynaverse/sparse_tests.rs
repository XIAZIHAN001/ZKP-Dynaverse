use super::*;
use ark_std::UniformRand;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn sparse_delta_matches_every_dense_field_and_both_public_verifiers() {
    let mut rng = StdRng::seed_from_u64(15401);
    for n in [2usize, 16, 32] {
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, anchor) = circuit.valid_instance();
        let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
        let (initial, aux) = params.prove(&instance, &anchor).unwrap();
        // Includes empty I, non-power-of-two I, n-1 and the full domain.
        for affected_count in [0usize, 1, 3.min(n), n - 1, n] {
            let mut current = anchor.clone();
            for i in 0..affected_count {
                current.wires[4 * n + i] += Fr::rand(&mut rng);
                current.wires[5 * n + i] = current.wires[3 * n + i] * current.wires[4 * n + i];
            }
            if affected_count == 0 {
                current.wires[1] += Fr::one();
                current.wires[n + 1] -= Fr::one();
            }
            let mut new_instance = instance.clone();
            new_instance.public_inputs[1] = current.wires[1];
            let entries: Vec<(usize, Fr)> = current.wires.iter().zip(&anchor.wires).enumerate()
                .filter_map(|(i, (&new, &old))| (new != old).then_some((i, new - old))).collect();
            let dense = params.prove_delta(&anchor, &current, &initial.anchor.wires, &aux.kzg,
                AnchorOpeningPolicy::Precompute, BatchOpenMode::Prepared, None,
                &mut CpuTrace::disabled()).unwrap();
            let sparse = params.prove_delta_sparse(&anchor, &entries, &initial.anchor.wires,
                &aux.kzg, &ArkReferenceBackend).unwrap();
            assert_eq!(sparse, dense, "n={n} a={affected_count}");
            let proof = DynaverseProof { anchor: initial.anchor.clone(), delta: Some(sparse),
                anchor_generation: 0, generation: 1 };
            assert!(params.verifying.verify(&new_instance, &proof).unwrap());
            assert!(params.verifying.verify_reference(&new_instance, &proof).unwrap());
        }
    }
}

#[test]
fn sparse_consistency_matches_dense_for_each_slice() {
    let mut rng = StdRng::seed_from_u64(15402);
    let circuit = ToyCircuit::new(8).unwrap();
    let srs = SharedSrs::setup(64, 8, circuit.permutation(), &mut rng).unwrap();
    let mut full = vec![Fr::zero(); 64];
    for i in [0, 7, 8, 15, 23, 31, 39, 47, 63] { full[i] = Fr::rand(&mut rng); }
    let sparse = sparse_from_dense(&full).unwrap();
    for t in 0..6 {
        let small = &full[t * 8..(t + 1) * 8];
        let sparse_small = sparse_from_dense(small).unwrap();
        assert_eq!(srs.consistency[t].prove_sparse(&sparse, &sparse_small, &ArkReferenceBackend).unwrap(),
            srs.consistency[t].prove(&full, small, &ArkReferenceBackend).unwrap());
    }
}

#[test]
fn session_matches_dense_through_hotspots_reverts_public_changes_and_rebuilds() {
    let n = 64;
    let circuit = ToyCircuit::new(n).unwrap();
    let (mut instance, witness) = circuit.valid_instance();
    let params = DynaverseParams::setup(circuit, &mut StdRng::seed_from_u64(15301)).unwrap();
    let (mut proof, mut aux) = params.prove(&instance, &witness).unwrap();
    let mut session = params.start_update_session(&instance, &witness).unwrap();
    for step in 0..12 {
        let old_instance = instance.clone();
        let old_witness = aux.current_witness().clone();
        let update;
        let mut public = Update { indices: vec![], values: vec![] };
        if step == 4 {
            instance.public_inputs[1] += Fr::one();
            public = Update { indices: vec![1], values: vec![instance.public_inputs[1]] };
            update = Update { indices: vec![1, n + 1],
                values: vec![old_witness.wires[1] + Fr::one(), old_witness.wires[n + 1] - Fr::one()] };
        } else {
            let i = if step < 4 { 1 } else { step - 4 };
            let value = if step == 3 { witness.wires[4 * n + i] }
                else { old_witness.wires[4 * n + i] + Fr::one() };
            update = Update { indices: vec![4 * n + i, 5 * n + i],
                values: vec![value, old_witness.wires[3 * n + i] * value] };
        }
        proof = params.update_proof(&old_instance, &instance, &old_witness, &update, &proof, &mut aux).unwrap();
        let candidate = session.update_with_public_inputs(&update, &public).unwrap();
        assert_eq!(candidate, proof, "step={step}");
        assert_eq!(session.current_witness(), aux.current_witness());
        assert_eq!(session.current_instance(), &instance);
        assert!(params.verifying.verify_reference(&instance, &candidate).unwrap());
    }
    assert!(proof.anchor_generation > 0, "sequence must cover a synchronous reference rebuild");
}

#[test]
fn session_rejects_invalid_updates_atomically_and_preserves_noops() {
    let n = 32;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let params = DynaverseParams::setup(circuit, &mut StdRng::seed_from_u64(15302)).unwrap();
    let mut session = params.start_update_session(&instance, &witness).unwrap();
    let before = session.proof().clone();
    for bad in [Update { indices: vec![6 * n], values: vec![Fr::one()] },
        Update { indices: vec![0], values: vec![] },
        Update { indices: vec![n + 1], values: vec![Fr::zero()] },
        Update { indices: vec![3 * n + 1], values: vec![Fr::zero()] },
        Update { indices: vec![5 * n + 1], values: vec![Fr::zero()] }] {
        assert!(session.update(&bad).is_err());
        assert_eq!(session.proof(), &before);
        assert_eq!(session.current_witness(), &witness);
        assert_eq!(session.current_instance(), &instance);
        assert_eq!(session.anchor_distance(), 0);
    }
    let noop = Update { indices: vec![4 * n + 1, 4 * n + 1],
        values: vec![Fr::zero(), witness.wires[4 * n + 1]] };
    assert_eq!(session.update(&noop).unwrap(), before);
    let bad_public = Update { indices: vec![1], values: vec![Fr::zero()] };
    assert!(session.update_with_public_inputs(&Update { indices: vec![], values: vec![] }, &bad_public).is_err());
    assert_eq!(session.current_instance(), &instance);
}

#[test]
fn fixed_two_wire_session_work_is_independent_of_domain_size() {
    let mut reference = None;
    for n in [16usize, 32, 64, 128] {
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let params = DynaverseParams::setup(circuit, &mut StdRng::seed_from_u64(15303)).unwrap();
        let mut session = params.start_update_session(&instance, &witness).unwrap();
        let value = witness.wires[4 * n + 1] + Fr::one();
        let update = Update { indices: vec![4 * n + 1, 5 * n + 1],
            values: vec![value, witness.wires[3 * n + 1] * value] };
        let (result, work) = crate::workcount::measure(|| session.update(&update));
        let proof = result.unwrap();
        assert!(proof.delta.is_some());
        assert!(params.verifying.verify(&instance, &proof).unwrap());
        println!("sparse n={n} d=2 a=1 {work:?} total={}", work.total());
        assert!(work.msm_points > 0 && work.scan_elements > 0 && work.gate_checks == 1);
        if let Some(expected) = reference { assert_eq!(work, expected, "n={n}"); }
        else { reference = Some(work); }
    }
}

struct FailingCpuBackend {
    calls: std::cell::Cell<usize>,
    fail_at: usize,
}
impl FailingCpuBackend {
    fn tick(&self) -> Result<()> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        ensure!(call != self.fail_at, "injected backend failure at call {call}");
        Ok(())
    }
}
impl OperatorBackend for FailingCpuBackend {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        self.tick()?;
        ensure!(points.len() == scalars.len(), "test backend MSM shape mismatch");
        Ok(G1Projective::msm_unchecked(points, scalars).into_affine())
    }
    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        let mut output = Vec::new();
        for i in 0..request.point_sets.len() {
            let offset = request.offsets[i];
            let len = request.lengths[i];
            output.push(self.msm(&request.point_sets[i][..len], &request.shared_scalars[offset..offset + len])?);
        }
        Ok(output)
    }
    fn ntt(&self, _: &mut [Fr], _: bool) -> Result<()> {
        anyhow::bail!("sparse fixed-anchor path must not dispatch a full-domain NTT")
    }
}

#[test]
fn session_backend_errors_do_not_publish_partial_state_or_hide_cpu_fallback() {
    let n = 32;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let params = DynaverseParams::setup(circuit, &mut StdRng::seed_from_u64(15304)).unwrap();
    let mut session = params.start_update_session(&instance, &witness).unwrap();
    let proof = session.proof().clone();
    let value = witness.wires[4 * n + 1] + Fr::one();
    let update = Update { indices: vec![4 * n + 1, 5 * n + 1],
        values: vec![value, witness.wires[3 * n + 1] * value] };
    let no_public = Update { indices: vec![], values: vec![] };
    for fail_at in [1usize, 4, 12, 30] {
        let backend = FailingCpuBackend { calls: std::cell::Cell::new(0), fail_at };
        let error = session.update_with_backend(&update, &no_public, &backend).unwrap_err();
        assert!(error.to_string().contains("injected backend failure"));
        assert_eq!(backend.calls.get(), fail_at);
        assert_eq!(session.proof(), &proof);
        assert_eq!(session.current_witness(), &witness);
        assert_eq!(session.current_instance(), &instance);
        assert_eq!(session.anchor_distance(), 0);
    }
    let backend = FailingCpuBackend { calls: std::cell::Cell::new(0), fail_at: usize::MAX };
    let (gpu_shape_result, selected_work) = crate::workcount::measure(||
        session.update_with_backend(&update, &no_public, &backend));
    let selected_proof = gpu_shape_result.unwrap();
    assert!(backend.calls.get() > 30);
    let mut default_session = params.start_update_session(&instance, &witness).unwrap();
    let (default_result, default_work) = crate::workcount::measure(|| default_session.update(&update));
    assert_eq!(selected_proof, default_result.unwrap());
    assert_eq!(selected_work, default_work, "work accounting must not depend on backend dispatch");
}

#[test]
fn sessions_match_dense_reference_at_small_rebuild_boundaries() {
    for n in [1usize, 2, 4, 8, 16] {
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let params = DynaverseParams::setup(circuit, &mut StdRng::seed_from_u64(15305)).unwrap();
        let (proof, mut aux) = params.prove(&instance, &witness).unwrap();
        let mut session = params.start_update_session(&instance, &witness).unwrap();
        let value = witness.wires[4 * n] + Fr::one();
        // Gate zero's sum is zero: the effective Hamming distance is one,
        // although the submitted update contains two entries.
        let update = Update { indices: vec![4 * n, 5 * n], values: vec![value, Fr::zero()] };
        let expected = params.update_proof(&instance, &instance, &witness, &update, &proof, &mut aux).unwrap();
        assert_eq!(session.update(&update).unwrap(), expected);
        assert!(params.verifying.verify_reference(&instance, session.proof()).unwrap());
    }
}

#[test]
fn larger_affected_sets_match_dense_and_record_selected_operand_sizes() {
    let n = 128;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, anchor) = circuit.valid_instance();
    let params = DynaverseParams::setup(circuit, &mut StdRng::seed_from_u64(15403)).unwrap();
    let (initial, aux) = params.prove(&instance, &anchor).unwrap();
    for a in [1usize, 2, 4, 8, 16, 32, 64, 127, 128] {
        let mut current = anchor.clone();
        for i in 0..a {
            current.wires[4 * n + i] += Fr::one();
            current.wires[5 * n + i] = current.wires[3 * n + i] * current.wires[4 * n + i];
        }
        let entries: Vec<(usize, Fr)> = current.wires.iter().zip(&anchor.wires).enumerate()
            .filter_map(|(i, (&new, &old))| (new != old).then_some((i, new - old))).collect();
        let (candidate, work) = crate::workcount::measure(|| params.prove_delta_sparse(&anchor,
            &entries, &initial.anchor.wires, &aux.kzg, &FailingCpuBackend {
                calls: std::cell::Cell::new(0), fail_at: usize::MAX }));
        let expected = params.prove_delta(&anchor, &current, &initial.anchor.wires, &aux.kzg,
            AnchorOpeningPolicy::Precompute, BatchOpenMode::Prepared, None, &mut CpuTrace::disabled()).unwrap();
        assert_eq!(candidate.unwrap(), expected, "a={a}");
        println!("sparse-scale n={n} d={} a={a} {work:?} total={}", entries.len(), work.total());
        assert!(work.max_operand <= (4 * a + 4) as u64);
        assert!(work.total() <= 2000 * (a as u64 + 1) * ((a + 1).ilog2() as u64 + 1).pow(2));
    }
}

struct RejectPublicCommitment {
    public_basis: G1Affine,
    seen: std::cell::Cell<bool>,
}
impl OperatorBackend for RejectPublicCommitment {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        if points == [self.public_basis] {
            self.seen.set(true);
            anyhow::bail!("injected public commitment failure");
        }
        Ok(G1Projective::msm_unchecked(points, scalars).into_affine())
    }
    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        let mut output = Vec::new();
        for i in 0..request.point_sets.len() {
            let start = request.offsets[i];
            let len = request.lengths[i];
            output.push(self.msm(&request.point_sets[i][..len], &request.shared_scalars[start..start + len])?);
        }
        Ok(output)
    }
    fn ntt(&self, _: &mut [Fr], _: bool) -> Result<()> { anyhow::bail!("unexpected NTT") }
}

#[test]
fn public_commitment_backend_failure_rolls_back_cached_public_and_private_state() {
    let n = 32;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let params = DynaverseParams::setup(circuit, &mut StdRng::seed_from_u64(15306)).unwrap();
    let mut session = params.start_update_session(&instance, &witness).unwrap();
    let proof = session.proof().clone();
    let wire_update = Update { indices: vec![1, n + 1],
        values: vec![witness.wires[1] + Fr::one(), witness.wires[n + 1] - Fr::one()] };
    let public_update = Update { indices: vec![1], values: vec![instance.public_inputs[1] + Fr::one()] };
    let backend = RejectPublicCommitment { public_basis: params.verifying.public_h_bases[1],
        seen: std::cell::Cell::new(false) };
    assert!(session.update_with_backend(&wire_update, &public_update, &backend).is_err());
    assert!(backend.seen.get(), "public-input commitment must pass through the selected backend");
    assert_eq!(session.proof(), &proof);
    assert_eq!(session.current_witness(), &witness);
    assert_eq!(session.current_instance(), &instance);
    assert_eq!(session.anchor_distance(), 0);
    let result = session.update_with_public_inputs(&wire_update, &public_update).unwrap();
    let mut new_instance = instance;
    new_instance.public_inputs[1] += Fr::one();
    assert!(params.verifying.verify(&new_instance, &result).unwrap());
    assert!(params.verifying.verify_reference(&new_instance, &result).unwrap());
}

#[test]
fn scattered_mixed_wire_deltas_and_zero_reversion_match_dense_proofs() {
    let mut rng = StdRng::seed_from_u64(15404);
    for n in [16usize, 64, 128] {
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, anchor) = circuit.valid_instance();
        let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
        let (initial, aux) = params.prove(&instance, &anchor).unwrap();
        for count in [1usize, 3, 7, n - 1] {
            let mut current = anchor.clone();
            let mut new_instance = instance.clone();
            for j in 0..count {
                // Odd stride permutes a radix-2 domain and scatters the set.
                let i = (9 * j + 3) % n;
                if j % 3 != 1 {
                    current.wires[n + i] += Fr::rand(&mut rng);
                    current.wires[2 * n + i] = current.wires[i] + current.wires[n + i];
                    current.wires[3 * n + i] = current.wires[2 * n + i];
                }
                if j % 3 != 0 { current.wires[4 * n + i] += Fr::rand(&mut rng); }
                if j == count - 1 {
                    current.wires[i] += Fr::one();
                    new_instance.public_inputs[i] = current.wires[i];
                    current.wires[2 * n + i] = current.wires[i] + current.wires[n + i];
                    current.wires[3 * n + i] = current.wires[2 * n + i];
                }
                current.wires[5 * n + i] = current.wires[3 * n + i] * current.wires[4 * n + i];
            }
            params.circuit.validate(&new_instance, &current).unwrap();
            let entries: Vec<(usize, Fr)> = current.wires.iter().zip(&anchor.wires).enumerate()
                .filter_map(|(i, (&new, &old))| (new != old).then_some((i, new - old))).collect();
            let expected = params.prove_delta(&anchor, &current, &initial.anchor.wires, &aux.kzg,
                AnchorOpeningPolicy::Precompute, BatchOpenMode::Prepared, None,
                &mut CpuTrace::disabled()).unwrap();
            let actual = params.prove_delta_sparse(&anchor, &entries, &initial.anchor.wires,
                &aux.kzg, &ArkReferenceBackend).unwrap();
            assert_eq!(actual, expected, "n={n} selected={count}");
            let proof = DynaverseProof { anchor: initial.anchor.clone(), delta: Some(actual),
                anchor_generation: 0, generation: 1 };
            assert!(params.verifying.verify(&new_instance, &proof).unwrap());
            assert!(params.verifying.verify_reference(&new_instance, &proof).unwrap());
        }
        let reverted = params.prove_delta_sparse(&anchor, &[], &initial.anchor.wires,
            &aux.kzg, &ArkReferenceBackend).unwrap();
        let dense_zero = params.prove_delta(&anchor, &anchor, &initial.anchor.wires, &aux.kzg,
            AnchorOpeningPolicy::Precompute, BatchOpenMode::Prepared, None,
            &mut CpuTrace::disabled()).unwrap();
        assert_eq!(reverted, dense_zero);
    }
}

#[test]
#[ignore = "explicit larger-domain CPU correctness/work sweep; dense setup is deliberately outside the update measurement"]
fn larger_domain_session_work_and_proofs_match_dense_reference() {
    let mut reference_work = None;
    for n in [16usize, 256, 512, 1024] {
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let params = DynaverseParams::setup(circuit, &mut StdRng::seed_from_u64(15307)).unwrap();
        let (proof, mut aux) = params.prove(&instance, &witness).unwrap();
        let mut session = params.start_update_session(&instance, &witness).unwrap();
        let value = witness.wires[4 * n + 1] + Fr::one();
        let update = Update { indices: vec![4 * n + 1, 5 * n + 1],
            values: vec![value, witness.wires[3 * n + 1] * value] };
        let (candidate, work) = crate::workcount::measure(|| session.update(&update));
        let expected = params.update_proof(&instance, &instance, &witness, &update, &proof, &mut aux).unwrap();
        let candidate = candidate.unwrap();
        assert_eq!(candidate, expected);
        assert!(params.verifying.verify(&instance, &candidate).unwrap());
        assert!(params.verifying.verify_reference(&instance, &candidate).unwrap());
        println!("sparse-domain n={n} d=2 a=1 {work:?} total={}", work.total());
        if let Some(previous) = reference_work { assert_eq!(work, previous, "n={n}"); }
        else { reference_work = Some(work); }
    }
}
