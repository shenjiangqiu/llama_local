use std::collections::BTreeMap;

use candle_core::Tensor;
use itertools::Itertools;
fn main() {
    // let device = candle_core::Device::new_metal(0).unwrap();
    let device = candle_core::Device::Cpu;
    let mut _all_tensors: Vec<_> = (0..2048)
        .map(|count| format!("tensors/{count}.safetensors"))
        .map(|name| {
            candle_core::safetensors::load(name, &device)
                .unwrap()
                .into_values()
                .next()
                .unwrap()
        })
        .collect();
    println!("len: {}", _all_tensors.len());
    // pading some zeros
    _all_tensors.iter_mut().for_each(|tensor| {
        let dims = tensor.dims();
        let history = dims[3];
        let q = dims[2];
        *tensor = tensor.reshape((32, q, history)).unwrap();
        *tensor = tensor.pad_with_zeros(2, 0, 97 - history).unwrap();
    });

    // stack each layer
    let all_tensors = _all_tensors
        .into_iter()
        .chunks(32)
        .into_iter()
        .map(|chunk| {
            let chunk = chunk.into_iter().collect_vec();
            assert_eq!(chunk.len(), 32);
            Tensor::stack(&chunk, 0).unwrap()
        })
        .collect_vec();

    // the dimension of the tensor is [1,head,q_size,v_size]
    for tensor in all_tensors.iter() {
        let dim: &[usize] = tensor.dims();
        println!("{:?}", dim);
    }
    let tensor = Tensor::cat(&all_tensors, 2).unwrap();

    println!("{:?}", tensor.dims());

    tensor.write_npy("output.npy").unwrap();
}

// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
// [1, 32, 1, 97]
