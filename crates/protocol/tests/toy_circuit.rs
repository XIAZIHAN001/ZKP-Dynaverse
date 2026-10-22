use ark_bn254::Fr;
use ark_ff::Zero;
use dynaverse_protocol::{toy_circuit::ToyCircuit, Instance, Witness};

#[test]
fn fixture_has_valid_gates_copy_edges_and_padding_at_three_sizes() {
    for n in [8usize, 16, 32] {
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        assert_eq!(instance.public_inputs.len(), n);
        assert_eq!(witness.wires.len(), 6 * n);
        assert_eq!(circuit.m(), 8 * n);
        assert_eq!(witness.wires[3 * n], Fr::zero());
        circuit.validate(&instance, &witness).unwrap();

        let sigma = circuit.permutation();
        let sigma = sigma.as_slice();
        for i in 0..n {
            assert_eq!(sigma[i], 6 * n + i);
            assert_eq!(sigma[6 * n + i], i);
            assert_eq!(sigma[2 * n + i], 3 * n + i);
            assert_eq!(sigma[3 * n + i], 2 * n + i);
        }
        for (index, &image) in sigma.iter().enumerate().skip(7 * n) {
            assert_eq!(image, index);
        }
    }
}

#[test]
fn malformed_circuit_shape_and_witness_relations_are_rejected() {
    assert!(ToyCircuit::new(0).is_err());
    assert!(ToyCircuit::new(3).is_err());
    let circuit = ToyCircuit::new(8).unwrap();
    let (instance, witness) = circuit.valid_instance();

    let mut short_instance = Instance {
        public_inputs: instance.public_inputs.clone(),
    };
    short_instance.public_inputs.pop();
    assert!(circuit.validate(&short_instance, &witness).is_err());
    let mut short_witness = Witness {
        wires: witness.wires.clone(),
    };
    short_witness.wires.pop();
    assert!(circuit.validate(&instance, &short_witness).is_err());

    let mut bad_add = witness.clone();
    bad_add.wires[2 * 8 + 1] += Fr::from(1u64);
    assert!(circuit.validate(&instance, &bad_add).is_err());
    let mut bad_mul = witness.clone();
    bad_mul.wires[5 * 8 + 1] += Fr::from(1u64);
    assert!(circuit.validate(&instance, &bad_mul).is_err());
    let mut bad_copy = witness.clone();
    bad_copy.wires[3 * 8 + 1] += Fr::from(1u64);
    assert!(circuit.validate(&instance, &bad_copy).is_err());
    let mut bad_public = instance.clone();
    bad_public.public_inputs[1] += Fr::from(1u64);
    assert!(circuit.validate(&bad_public, &witness).is_err());
}
