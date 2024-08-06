use super::{Align64, L1_SIZE, L2_SIZE, L3_SIZE, QA, QAB};

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
pub fn activate_ft(us: &Align64<[i16; L1_SIZE]>, them: &Align64<[i16; L1_SIZE]>, output: &mut Align64<[u8; L1_SIZE]>) {
    // this is just autovec'd for the moment.
    for (a, acc) in [us, them].into_iter().enumerate() {
        for i in 0..L1_SIZE / 2 {
            let l = acc.0[i];
            let r = acc.0[L1_SIZE / 2 + i];
            let cl = i32::clamp(i32::from(l), 0, QA);
            let cr = i32::clamp(i32::from(r), 0, QA);
            let r = (cl * cr) / QA;
            output.0[i + a * L1_SIZE / 2] = r as u8;
        }
    }
}

#[allow(clippy::needless_range_loop, clippy::cast_precision_loss)]
pub fn propagate_l1(
    inputs: &Align64<[u8; L1_SIZE]>,
    weights: &Align64<[i8; L1_SIZE * L2_SIZE]>,
    biases: &Align64<[f32; L2_SIZE]>,
    output: &mut Align64<[f32; L2_SIZE]>,
) {
    const SUM_DIV: f32 = QAB as f32;
    // this is just autovec'd for the moment.
    let mut sums = [0; L2_SIZE];
    for i in 0..L1_SIZE {
        for j in 0..L2_SIZE {
            sums[j] += i32::from(inputs.0[i]) * i32::from(weights.0[j * L1_SIZE + i]);
        }
    }

    for i in 0..L2_SIZE {
        // convert to f32 and activate L1
        let clipped = f32::clamp((sums[i] as f32) / SUM_DIV + biases.0[i], 0.0, 1.0);
        output.0[i] = clipped * clipped;
    }
}

#[allow(clippy::needless_range_loop)]
pub fn propagate_l2(
    inputs: &Align64<[f32; L2_SIZE]>,
    weights: &Align64<[f32; L2_SIZE * L3_SIZE]>,
    biases: &Align64<[f32; L3_SIZE]>,
    output: &mut Align64<[f32; L3_SIZE]>,
) {
    // this is just autovec'd for the moment.
    let mut sums = biases.0;

    // affine transform for l2
    for i in 0..L2_SIZE {
        for j in 0..L3_SIZE {
            sums[j] += inputs.0[i] * weights.0[j * L2_SIZE + i];
        }
    }

    // activate l2
    for i in 0..L3_SIZE {
        let clipped = f32::clamp(sums[i], 0.0, 1.0);
        output.0[i] = clipped * clipped;
    }
}

pub fn propagate_l3(inputs: &Align64<[f32; L3_SIZE]>, weights: &Align64<[f32; L3_SIZE]>, bias: f32, output: &mut f32) {
    let mut sum = bias;

    for i in 0..L3_SIZE {
        sum += inputs.0[i] * weights.0[i];
    }

    *output = sum;
}
