//! End-to-end trajectory check: take each RotorPy-generated trajectory (the
//! golden fixtures), free-run the Rust sim from ONLY the initial state plus the
//! recorded command stream, and compare the two full trajectories.
//!
//!   cargo run --example replay
//!
//! Prints, per fixture, the position/velocity/attitude/rotor-speed divergence
//! over the whole rollout, and writes side-by-side CSVs to
//! `web-demo/fixtures/compare/<name>.csv` for plotting.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use rotor_rs::{
    Ctbr, CtbrCmd, ControlLaw, Multirotor, Quat, QuadParamsInput, RotorForce, State, Vec3,
};

const N_SUB: usize = 8;

#[derive(Deserialize)]
struct Fixture {
    name: String,
    abstraction: String,
    dt: f64,
    params: PJson,
    initial_state: SJson,
    steps: Vec<StepJson>,
}
#[derive(Deserialize)]
struct StepJson {
    cmd: serde_json::Value,
    state: SJson,
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
    x: [f64; 3], v: [f64; 3], q: [f64; 4], w: [f64; 3], wind: [f64; 3], rotor_speeds: [f64; 4],
}

impl PJson {
    fn to_input(&self) -> QuadParamsInput {
        let mut rotor_pos = [[0.0; 3]; 4];
        let mut rotor_directions = [0.0; 4];
        for r in 0..4 {
            rotor_pos[r] = self.rotor_pos[r];
            rotor_directions[r] = self.rotor_directions[r];
        }
        QuadParamsInput {
            mass: self.mass, ixx: self.Ixx, iyy: self.Iyy, izz: self.Izz,
            ixy: self.Ixy, iyz: self.Iyz, ixz: self.Ixz, rotor_pos, rotor_directions,
            c_dx: self.c_Dx, c_dy: self.c_Dy, c_dz: self.c_Dz,
            k_eta: self.k_eta, k_m: self.k_m, k_d: self.k_d, k_z: self.k_z,
            k_h: self.k_h, k_flap: self.k_flap, tau_m: self.tau_m,
            rotor_speed_min: self.rotor_speed_min, rotor_speed_max: self.rotor_speed_max,
            k_w: self.k_w,
        }
    }
}
impl SJson {
    fn to_state(&self) -> State<f64> {
        State {
            x: Vec3::new(self.x[0], self.x[1], self.x[2]),
            v: Vec3::new(self.v[0], self.v[1], self.v[2]),
            q: Quat::new(self.q[0], self.q[1], self.q[2], self.q[3]),
            w: Vec3::new(self.w[0], self.w[1], self.w[2]),
            wind: Vec3::new(self.wind[0], self.wind[1], self.wind[2]),
            rotor_speeds: self.rotor_speeds,
        }
    }
}

fn rotor_force_cmd(v: &serde_json::Value) -> [f64; 4] {
    let a = v["cmd_motor_thrusts"].as_array().unwrap();
    core::array::from_fn(|i| a[i].as_f64().unwrap())
}
fn ctbr_cmd(v: &serde_json::Value) -> CtbrCmd<f64> {
    let w = v["cmd_w"].as_array().unwrap();
    CtbrCmd {
        thrust: v["cmd_thrust"].as_f64().unwrap(),
        w_cmd: Vec3::new(w[0].as_f64().unwrap(), w[1].as_f64().unwrap(), w[2].as_f64().unwrap()),
    }
}

struct Diff {
    pos_max: f64,
    pos_rms: f64,
    vel_max: f64,
    att_max_deg: f64,
    rotor_max: f64,
    final_pos_rs: [f64; 3],
    final_pos_py: [f64; 3],
}

/// Angle (deg) between two unit quaternions.
fn quat_angle_deg(a: [f64; 4], b: [f64; 4]) -> f64 {
    let d = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]).abs().min(1.0);
    2.0 * d.acos().to_degrees()
}

fn run<C, F>(fx: &Fixture, mk_cmd: F) -> (Diff, Vec<(f64, [f64; 3], [f64; 3])>)
where
    C: ControlLaw<f64>,
    F: Fn(&serde_json::Value) -> C::Command,
{
    let veh: Multirotor<f64, C> = Multirotor::with_substeps(&fx.params.to_input(), N_SUB);
    let mut s = fx.initial_state.to_state();

    let mut pos_max = 0.0_f64;
    let mut pos_sq = 0.0_f64;
    let mut vel_max = 0.0_f64;
    let mut att_max = 0.0_f64;
    let mut rotor_max = 0.0_f64;
    let mut trace = Vec::with_capacity(fx.steps.len());

    for (t, step) in fx.steps.iter().enumerate() {
        let cmd = mk_cmd(&step.cmd);
        s = veh.step(&s, &cmd, fx.dt);
        let py = &step.state;

        let rs_x = [s.x.x, s.x.y, s.x.z];
        let dp = ((rs_x[0] - py.x[0]).powi(2)
            + (rs_x[1] - py.x[1]).powi(2)
            + (rs_x[2] - py.x[2]).powi(2))
        .sqrt();
        pos_max = pos_max.max(dp);
        pos_sq += dp * dp;
        let dv = ((s.v.x - py.v[0]).powi(2) + (s.v.y - py.v[1]).powi(2) + (s.v.z - py.v[2]).powi(2)).sqrt();
        vel_max = vel_max.max(dv);
        att_max = att_max.max(quat_angle_deg(s.q.to_array(), py.q));
        for r in 0..4 {
            rotor_max = rotor_max.max((s.rotor_speeds[r] - py.rotor_speeds[r]).abs());
        }
        trace.push(((t as f64 + 1.0) * fx.dt, rs_x, py.x));
    }

    let n = fx.steps.len() as f64;
    let last = &fx.steps.last().unwrap().state;
    (
        Diff {
            pos_max,
            pos_rms: (pos_sq / n).sqrt(),
            vel_max,
            att_max_deg: att_max,
            rotor_max,
            final_pos_rs: [s.x.x, s.x.y, s.x.z],
            final_pos_py: last.x,
        },
        trace,
    )
}

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/sim");
    let cmp_dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/compare");
    fs::create_dir_all(&cmp_dir).unwrap();

    let mut paths: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {dir:?}: {e} (run scripts/export_fixtures.py)"))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .collect();
    paths.sort();

    println!(
        "Free-running Rust rollout vs RotorPy trajectory (over the full ~2 s, fine rate, N_SUB={N_SUB})\n"
    );
    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>9} {:>10}",
        "fixture", "pos_max[m]", "pos_rms[m]", "vel_max", "att[deg]", "rotor_max"
    );
    println!("{}", "-".repeat(82));

    let mut worst_pos = 0.0_f64;
    for path in paths {
        let fx: Fixture = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let (d, trace) = match fx.abstraction.as_str() {
            "cmd_motor_thrusts" => run::<RotorForce, _>(&fx, rotor_force_cmd),
            "cmd_ctbr" => run::<Ctbr, _>(&fx, ctbr_cmd),
            other => panic!("unknown abstraction {other}"),
        };
        worst_pos = worst_pos.max(d.pos_max);
        println!(
            "{:<28} {:>10.2e} {:>10.2e} {:>10.2e} {:>9.2e} {:>10.2e}",
            fx.name, d.pos_max, d.pos_rms, d.vel_max, d.att_max_deg, d.rotor_max
        );

        // CSV: t, rust pos, python pos
        let mut csv = String::from("t,rs_x,rs_y,rs_z,py_x,py_y,py_z\n");
        for (t, rs, py) in &trace {
            csv.push_str(&format!(
                "{t:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                rs[0], rs[1], rs[2], py[0], py[1], py[2]
            ));
        }
        fs::write(cmp_dir.join(format!("{}.csv", fx.name)), csv).unwrap();

        // also show the final-position agreement explicitly
        println!(
            "    final pos  rust=[{:.4}, {:.4}, {:.4}]  rotorpy=[{:.4}, {:.4}, {:.4}]",
            d.final_pos_rs[0], d.final_pos_rs[1], d.final_pos_rs[2],
            d.final_pos_py[0], d.final_pos_py[1], d.final_pos_py[2],
        );
    }

    println!("\nworst position divergence across all fixtures: {worst_pos:.3e} m");
    println!("side-by-side CSVs written to {}", cmp_dir.display());
}
