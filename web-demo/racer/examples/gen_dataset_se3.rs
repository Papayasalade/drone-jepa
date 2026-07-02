//! Generate ROTOR-FORCE training data (the paper's action space) fast in Rust:
//! a domain-randomized drone tracks a random sum-of-sines reference with the SE3
//! controller. Mirrors the Python pipeline's quality recipe — fine-rate (200 Hz)
//! control subsampled to 20 Hz with the per-window MEAN action (smooth ZOH).
//!
//!   cargo run --release --example gen_dataset_se3 -- <n_traj> <steps> <out.bin> <seed> [fourier|gp] [smooth|recovery]

use std::io::Write;

use racer::rng::Rng;
use racer::se3::FlatRef;
use racer::{FourierRef, GpRef, Multirotor, QuadParamsInput, Quat, RotorForce, Se3Control, State, Vec3};

/// Either reference family, behind one `.at(t)` interface.
enum Ref {
    Fourier(FourierRef),
    Gp(GpRef),
}
impl Ref {
    fn at(&self, t: f64) -> FlatRef {
        match self {
            Ref::Fourier(f) => f.at(t),
            Ref::Gp(g) => g.at(t),
        }
    }
}

#[derive(Clone, Copy)]
enum RefKind {
    Fourier,
    Gp,
}

impl RefKind {
    fn parse(s: Option<&String>) -> Self {
        match s.map(|v| v.as_str()) {
            Some("gp") => Self::Gp,
            Some("fourier") | None => Self::Fourier,
            Some(other) => panic!("unknown ref kind '{other}', expected fourier|gp"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Profile {
    Smooth,
    Recovery,
}

impl Profile {
    fn parse(s: Option<&String>) -> Self {
        match s.map(|v| v.as_str()) {
            Some("recovery") => Self::Recovery,
            Some("smooth") | None => Self::Smooth,
            Some(other) => panic!("unknown profile '{other}', expected smooth|recovery"),
        }
    }
}

const HUM_IXX: f64 = 3.65e-3;
const HUM_IYY: f64 = 3.68e-3;
const HUM_IZZ: f64 = 7.03e-3;
const HUM_KETA: f64 = 5.57e-6;
const HUM_KM: f64 = 1.36e-7;

fn lerp(r: &mut Rng, a: f64, b: f64) -> f64 {
    a + (b - a) * r.uniform()
}

fn denv(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// DR ranges matching the Python DRRanges (controllable for the fixed SE3 gains).
/// Base drone = hummingbird, overridable via DRONE_* env (same convention as
/// gen_dataset_rf / rotor_fly / jepa_fly).
fn sample_drone(r: &mut Rng) -> QuadParamsInput {
    let d = denv("DRONE_ARM", 0.17) * std::f64::consts::FRAC_1_SQRT_2;
    QuadParamsInput {
        mass: denv("DRONE_MASS", 0.5) * lerp(r, 0.5, 1.5),
        ixx: denv("DRONE_IXX", HUM_IXX) * lerp(r, 0.7, 1.3),
        iyy: denv("DRONE_IYY", HUM_IYY) * lerp(r, 0.7, 1.3),
        izz: denv("DRONE_IZZ", HUM_IZZ) * lerp(r, 0.7, 1.3),
        ixy: 0.0, iyz: 0.0, ixz: 0.0,
        rotor_pos: [[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]],
        rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: lerp(r, 0.05, 0.30), c_dy: lerp(r, 0.05, 0.30), c_dz: lerp(r, 0.05, 0.30),
        k_eta: denv("DRONE_K_ETA", HUM_KETA) * lerp(r, 0.5, 1.5),
        k_m: denv("DRONE_K_M", HUM_KM) * lerp(r, 0.5, 1.5),
        k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
        tau_m: lerp(r, 0.01, 0.03), // tamed so SE3 (which ignores motor lag) stays stable
        rotor_speed_min: 0.0, rotor_speed_max: 1500.0,
        k_w: 1.0,
    }
}

fn state_to_18(s: &State<f64>) -> [f32; 18] {
    let r = s.q.to_rotmat();
    [
        s.x.x as f32, s.x.y as f32, s.x.z as f32,
        s.v.x as f32, s.v.y as f32, s.v.z as f32,
        r.rows[0][0] as f32, r.rows[0][1] as f32, r.rows[0][2] as f32,
        r.rows[1][0] as f32, r.rows[1][1] as f32, r.rows[1][2] as f32,
        r.rows[2][0] as f32, r.rows[2][1] as f32, r.rows[2][2] as f32,
        s.w.x as f32, s.w.y as f32, s.w.z as f32,
    ]
}

fn small_tilt(r: &mut Rng, max_rad: f64) -> Quat<f64> {
    let roll = lerp(r, -max_rad, max_rad);
    let pitch = lerp(r, -max_rad, max_rad);
    let (sr, cr) = (0.5 * roll).sin_cos();
    let (sp, cp) = (0.5 * pitch).sin_cos();
    Quat::new(sr * cp, cr * sp, sr * sp, cr * cp).normalized()
}

fn one_traj(r: &mut Rng, steps: usize, ref_kind: RefKind, profile: Profile) -> Option<(Vec<f32>, Vec<f32>)> {
    let p = sample_drone(r);
    let dt_control = 0.005;
    let decim = 10; // -> 20 Hz logging
    let f_max = p.k_eta * p.rotor_speed_max * p.rotor_speed_max;
    let hover = (p.mass * 9.81 / (4.0 * p.k_eta)).sqrt();

    let veh: Multirotor<f64, RotorForce> = Multirotor::with_substeps(&p, 2);
    let se3 = Se3Control::new(&p);
    let center = Vec3::new(0.0, 0.0, 1.5);
    let traj = match ref_kind {
        RefKind::Gp => Ref::Gp(GpRef::sample(r, center, steps as f64 * 0.05)),
        RefKind::Fourier => Ref::Fourier(FourierRef::sample(r, center)),
    };
    let recovery = profile == Profile::Recovery;

    let mut s = State {
        x: center + if recovery {
            Vec3::new(lerp(r, -0.9, 0.9), lerp(r, -0.9, 0.9), lerp(r, -0.35, 0.65))
        } else {
            Vec3::new(lerp(r, -0.3, 0.3), lerp(r, -0.3, 0.3), lerp(r, -0.3, 0.3))
        },
        v: if recovery {
            Vec3::new(lerp(r, -1.1, 1.1), lerp(r, -1.1, 1.1), lerp(r, -0.9, 0.6))
        } else {
            Vec3::new(lerp(r, -0.3, 0.3), lerp(r, -0.3, 0.3), lerp(r, -0.3, 0.3))
        },
        q: if recovery { small_tilt(r, 0.30) } else { Quat::new(0.0, 0.0, 0.0, 1.0) },
        w: if recovery {
            Vec3::new(lerp(r, -0.7, 0.7), lerp(r, -0.7, 0.7), lerp(r, -0.4, 0.4))
        } else {
            Vec3::new(lerp(r, -0.2, 0.2), lerp(r, -0.2, 0.2), lerp(r, -0.2, 0.2))
        },
        wind: if recovery {
            Vec3::new(lerp(r, -0.8, 0.8), lerp(r, -0.8, 0.8), 0.0)
        } else {
            Vec3::zero()
        },
        rotor_speeds: [hover; 4],
    };
    if s.x.z < 0.8 {
        s.x.z = 0.8;
    }
    let wind = s.wind;

    // dual action label: [f0,f1,f2,f3 (rotor force), thrust, wx,wy,wz (CTBR)]
    let total_thrust_max = 4.0 * f_max;
    let rate_max = 12.0;
    let mut states = Vec::with_capacity(steps * 18);
    let mut actions = Vec::with_capacity(steps * 8);
    let mut win = [0.0f64; 8];
    for i in 0..steps * decim {
        if i % decim == 0 {
            states.extend_from_slice(&state_to_18(&s));
            win = [0.0; 8];
        }
        let flat = traj.at(i as f64 * dt_control);
        let (u, thrust, cmd_w) = se3.update_full(&s, &flat);
        let noise_scale = if recovery && i < steps * decim / 3 {
            0.10 * p.mass * 9.81 / 4.0
        } else {
            0.0
        };
        let force: [f64; 4] = core::array::from_fn(|j| (u[j] + noise_scale * r.normal()).clamp(0.0, f_max));
        // CTBR label (same flight): clamp like the Python pipeline
        let th = thrust.clamp(0.0, total_thrust_max);
        let cw = [
            cmd_w.x.clamp(-rate_max, rate_max),
            cmd_w.y.clamp(-rate_max, rate_max),
            cmd_w.z.clamp(-rate_max, rate_max),
        ];
        s = veh.step(&s, &force, dt_control); // drone is driven by the rotor forces
        s.wind = wind;
        let inv = 1.0 / decim as f64;
        for j in 0..4 {
            win[j] += force[j] * inv;
        }
        win[4] += th * inv;
        win[5] += cw[0] * inv;
        win[6] += cw[1] * inv;
        win[7] += cw[2] * inv;
        if (i + 1) % decim == 0 {
            for j in 0..8 {
                actions.push(win[j] as f32);
            }
        }
        let up_z = s.q.to_rotmat().rows[2][2];
        if !s.x.z.is_finite()
            || s.x.z < 0.35
            || (s.x - center).norm() > 8.0
            || s.v.norm() > 8.0
            || s.w.norm() > 8.0
            || up_z < 0.45
        {
            return None; // diverged / unstable
        }
    }
    Some((states, actions))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_traj: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let out = args.get(3).cloned().unwrap_or_else(|| "artifacts/se3_rotor.bin".into());
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let ref_kind = RefKind::parse(args.get(5));
    let profile = Profile::parse(args.get(6));
    println!(
        "references: {}, profile: {}",
        match ref_kind {
            RefKind::Gp => "GP (exp-sine-squared)",
            RefKind::Fourier => "sum-of-sines",
        },
        match profile {
            Profile::Smooth => "smooth",
            Profile::Recovery => "recovery",
        },
    );

    let mut r = Rng::new(seed);
    let mut all_states = Vec::with_capacity(n_traj * steps * 18);
    let mut all_actions = Vec::with_capacity(n_traj * steps * 8);
    let (mut kept, mut tries) = (0usize, 0usize);
    while kept < n_traj {
        tries += 1;
        if let Some((st, ac)) = one_traj(&mut r, steps, ref_kind, profile) {
            all_states.extend_from_slice(&st);
            all_actions.extend_from_slice(&ac);
            kept += 1;
            if kept % 200 == 0 {
                println!("  {kept}/{n_traj} ({tries} tries)");
            }
        }
    }

    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    f.write_all(&(n_traj as u32).to_le_bytes()).unwrap();
    f.write_all(&(steps as u32).to_le_bytes()).unwrap();
    f.write_all(&8u32.to_le_bytes()).unwrap(); // n_act columns (4 rotor-force + 4 CTBR)
    for v in &all_states {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    for v in &all_actions {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
    println!("wrote {out}: {n_traj} traj x {steps} steps, 8 action cols ({} discarded)", tries - n_traj);
}
