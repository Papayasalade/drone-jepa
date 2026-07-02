//! Differential test of the Rust SE3 controller vs Python RotorPy `SE3Control`.
//! Same (params, state, flat ref) -> same rotor forces. (Pure linear algebra, no
//! integrator, so it should match to ~1e-9.) Fixtures: scripts/export_se3_goldens.py.

use std::path::Path;

use rotor_rs::{FlatRef, Quat, QuadParamsInput, Se3Control, State, Vec3};
use serde::Deserialize;

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    params: PJson,
    state: SJson,
    flat: FJson,
    forces: Vec<f64>,
}
#[derive(Deserialize)]
#[allow(non_snake_case)]
struct PJson {
    mass: f64, Ixx: f64, Iyy: f64, Izz: f64, Ixy: f64, Iyz: f64, Ixz: f64,
    rotor_pos: Vec<[f64; 3]>, rotor_directions: Vec<f64>,
    c_Dx: f64, c_Dy: f64, c_Dz: f64,
    k_eta: f64, k_m: f64, k_d: f64, k_z: f64, k_h: f64, k_flap: f64,
    tau_m: f64, rotor_speed_min: f64, rotor_speed_max: f64, k_w: f64,
}
#[derive(Deserialize)]
struct SJson {
    x: [f64; 3], v: [f64; 3], q: [f64; 4], w: [f64; 3],
}
#[derive(Deserialize)]
struct FJson {
    x: [f64; 3], x_dot: [f64; 3], x_ddot: [f64; 3], yaw: f64, yaw_dot: f64,
}

impl PJson {
    fn to_input(&self) -> QuadParamsInput {
        QuadParamsInput {
            mass: self.mass, ixx: self.Ixx, iyy: self.Iyy, izz: self.Izz,
            ixy: self.Ixy, iyz: self.Iyz, ixz: self.Ixz,
            rotor_pos: [self.rotor_pos[0], self.rotor_pos[1], self.rotor_pos[2], self.rotor_pos[3]],
            rotor_directions: [
                self.rotor_directions[0], self.rotor_directions[1],
                self.rotor_directions[2], self.rotor_directions[3],
            ],
            c_dx: self.c_Dx, c_dy: self.c_Dy, c_dz: self.c_Dz,
            k_eta: self.k_eta, k_m: self.k_m, k_d: self.k_d, k_z: self.k_z,
            k_h: self.k_h, k_flap: self.k_flap, tau_m: self.tau_m,
            rotor_speed_min: self.rotor_speed_min, rotor_speed_max: self.rotor_speed_max,
            k_w: self.k_w,
        }
    }
}

#[test]
fn se3_matches_rotorpy() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let txt = std::fs::read_to_string(root.join("../fixtures/se3/goldens.json"))
        .expect("se3 goldens (run scripts/export_se3_goldens.py)");
    let g: Goldens = serde_json::from_str(&txt).unwrap();

    let mut worst = 0.0_f64;
    for (ci, c) in g.cases.iter().enumerate() {
        let se3 = Se3Control::new(&c.params.to_input());
        let state = State {
            x: Vec3::new(c.state.x[0], c.state.x[1], c.state.x[2]),
            v: Vec3::new(c.state.v[0], c.state.v[1], c.state.v[2]),
            q: Quat::new(c.state.q[0], c.state.q[1], c.state.q[2], c.state.q[3]),
            w: Vec3::new(c.state.w[0], c.state.w[1], c.state.w[2]),
            wind: Vec3::zero(),
            rotor_speeds: [0.0; 4],
        };
        let flat = FlatRef {
            x: Vec3::new(c.flat.x[0], c.flat.x[1], c.flat.x[2]),
            x_dot: Vec3::new(c.flat.x_dot[0], c.flat.x_dot[1], c.flat.x_dot[2]),
            x_ddot: Vec3::new(c.flat.x_ddot[0], c.flat.x_ddot[1], c.flat.x_ddot[2]),
            yaw: c.flat.yaw,
            yaw_dot: c.flat.yaw_dot,
        };
        let got = se3.update(&state, &flat);
        for j in 0..4 {
            let e = (got[j] - c.forces[j]).abs();
            worst = worst.max(e);
            assert!(e < 1e-6, "case {ci} rotor {j}: rust={} py={} (|d|={e:.2e})", got[j], c.forces[j]);
        }
    }
    println!("SE3 worst force error vs RotorPy = {worst:.2e} N");
}
