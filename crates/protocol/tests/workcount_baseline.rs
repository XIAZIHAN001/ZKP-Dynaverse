//! Stage 0 baseline: operation-size counters for the current (dense) update.
//!
//! A fixed d=2 update (one multiplication gate: its multiplier and output)
//! is proved at several n. The dense reference path still runs length-m
//! transforms and MSMs, so its recorded work must grow with n. Later stages
//! add a sparse path whose counters are expected to stay flat at fixed d.
use ark_ff::One;
use dynaverse_protocol::dynaverse::DynaverseParams;
use dynaverse_protocol::toy_circuit::ToyCircuit;
use dynaverse_protocol::workcount::{measure, WorkCount};
use dynaverse_protocol::Update;
use rand::{rngs::StdRng, SeedableRng};

fn dense_update_work(n: usize) -> WorkCount {
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let params = DynaverseParams::setup(circuit, &mut StdRng::seed_from_u64(7000 + n as u64)).unwrap();
    let (proof, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
    // Gate 1: multiplier +1 and output += left input keeps s4*s5=s6.
    let update = Update {
        indices: vec![4 * n + 1, 5 * n + 1],
        values: vec![
            witness.wires[4 * n + 1] + ark_bn254::Fr::one(),
            witness.wires[5 * n + 1] + witness.wires[3 * n + 1],
        ],
    };
    // measure takes a closure; it borrows aux mutably only while it runs.
    let (updated, work) = measure(|| {
        params.update_proof(&instance, &instance, &witness, &update, &proof, &mut aux)
    });
    let updated = updated.unwrap();
    assert!(updated.delta.is_some(), "d=2 must stay below the sqrt(n) rebuild bound");
    assert!(params.verifying_key().verify(&instance, &updated).unwrap());
    work
}

#[test]
fn measure_isolates_nested_work() {
    let (_, outer) = measure(|| {
        let (_, inner) = measure(|| {
            let circuit = ToyCircuit::new(4).unwrap();
            let (instance, witness) = circuit.valid_instance();
            circuit.validate(&instance, &witness).unwrap();
        });
        assert_eq!(inner.gate_checks, 4);
    });
    assert_eq!(outer.gate_checks, 4);
    let (_, empty) = measure(|| ());
    assert_eq!(empty, WorkCount::default());
}

#[test]
fn dense_update_work_grows_with_n() {
    // n=16 is the smallest size where ceil(sqrt(n))=4 > d=2.
    let sizes = [16usize, 32, 64, 128];
    let mut rows = Vec::new();
    for &n in &sizes {
        let work = dense_update_work(n);
        println!("n={n:4} m={:5} ntt={:7} msm={:7} g2_msm={:6} poly={:9} scan={:6} gates={:4} max_operand={:5} total={}",
            8 * n, work.ntt_elements, work.msm_points, work.g2_msm_points, work.poly_elements,
            work.scan_elements, work.gate_checks, work.max_operand, work.total());
        rows.push((n, work));
    }
    for i in 1..rows.len() {
        let (n_prev, prev) = rows[i - 1];
        let (n, work) = rows[i];
        // Dense path: an m=8n IFFT, a length-m MSM and full-witness scans.
        assert!(work.max_operand >= 8 * n as u64, "n={n}: no length-m operand recorded");
        assert!(work.ntt_elements >= 2 * prev.ntt_elements, "n={n}: NTT work did not scale with n");
        assert!(work.msm_points >= 2 * prev.msm_points - 64, "n={n}: MSM work did not scale with n");
        assert!(work.scan_elements >= 2 * prev.scan_elements, "n={n}: scan work did not scale with n");
        assert!(work.gate_checks == n as u64, "n={n}: full validation should check every gate once");
        assert!(work.total() > prev.total(), "n={n} vs {n_prev}: total work did not grow");
    }
}
