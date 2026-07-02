//! Generate ROTOR-FORCE MPPI training data (the broad rotor-force action
//! distribution the paper trains on, in rotor-force space). A rotor-force MPPI
//! races random gate courses on the true dynamics, logging state(18) + the four
//! rotor forces. This is what makes a rotor-force model in-distribution for
//! rotor-force planning at deployment.
//!
//!   cargo run --release --example gen_dataset_rf -- <n_traj> <steps> <out.bin> <seed>

use std::io::Write;

use racer::rng::Rng;
use racer::{Course, Gate, Multirotor, QuadParamsInput, Quat, RotorForce, State, Vec3};

fn lerp(r: &mut Rng, a: f64, b: f64) -> f64 {
    a + (b - a) * r.uniform()
}

fn denv(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
/// Table-I-style domain randomization AROUND a base drone. The base is the
/// hummingbird by default; override via DRONE_* env vars (same convention as
/// rotor_fly / jepa_fly, fed by train_recipe's drone-spec JSON).
fn sample_drone(r: &mut Rng) -> QuadParamsInput {
    let d = denv("DRONE_ARM", 0.17) * std::f64::consts::FRAC_1_SQRT_2;
    QuadParamsInput {
        mass: denv("DRONE_MASS", 0.5) * lerp(r, 0.6, 1.5),
        ixx: denv("DRONE_IXX", 3.65e-3) * lerp(r, 0.7, 1.3),
        iyy: denv("DRONE_IYY", 3.68e-3) * lerp(r, 0.7, 1.3),
        izz: denv("DRONE_IZZ", 7.03e-3) * lerp(r, 0.7, 1.3),
        ixy: 0.0, iyz: 0.0, ixz: 0.0,
        rotor_pos: [[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]],
        rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: lerp(r, 0.05, 0.30), c_dy: lerp(r, 0.05, 0.30), c_dz: lerp(r, 0.05, 0.30),
        k_eta: denv("DRONE_K_ETA", 5.57e-6) * lerp(r, 0.6, 1.4),
        k_m: denv("DRONE_K_M", 1.36e-7) * lerp(r, 0.6, 1.4),
        k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
        tau_m: denv("DRONE_TAU_M_LO", 0.01) + (denv("DRONE_TAU_M_HI", 0.03)
            - denv("DRONE_TAU_M_LO", 0.01)) * r.uniform(),
        rotor_speed_min: 0.0, rotor_speed_max: 1500.0,
        k_w: denv("DRONE_K_W", 1.0),
    }
}

fn random_course(r: &mut Rng) -> Course {
    let n = 5 + (r.uniform() * 3.0) as usize;
    let gates = (0..n).map(|_| {
        let c = Vec3::new(lerp(r, -5.0, 5.0), lerp(r, -5.0, 5.0), lerp(r, 1.0, 2.5));
        let ang = lerp(r, 0.0, std::f64::consts::TAU);
        Gate::new(c, Vec3::new(ang.cos(), ang.sin(), 0.0), lerp(r, 0.9, 1.3))
    }).collect();
    Course::new(gates, true)
}

fn state_to_18(s: &State<f64>) -> [f32; 18] {
    let r = s.q.to_rotmat();
    [
        s.x.x as f32, s.x.y as f32, s.x.z as f32, s.v.x as f32, s.v.y as f32, s.v.z as f32,
        r.rows[0][0] as f32, r.rows[0][1] as f32, r.rows[0][2] as f32,
        r.rows[1][0] as f32, r.rows[1][1] as f32, r.rows[1][2] as f32,
        r.rows[2][0] as f32, r.rows[2][1] as f32, r.rows[2][2] as f32,
        s.w.x as f32, s.w.y as f32, s.w.z as f32,
    ]
}

// rotor-force MPPI against the true dynamics
struct RfMppi {
    cfg_s: usize, cfg_t: usize, dt: f64, lambda: f64, beta: f64,
    sigma: f64, f_hover: f64, f_max: f64,
    w_vel: f64, w_alt: f64, w_terminal: f64, gate_reward: f64, v_max: f64, w_speed: f64,
    nominal: Vec<[f64; 4]>, rng: Rng,
}

impl RfMppi {
    fn cost(&self, init: Vec3<f64>, traj: &[Vec3<f64>], course: &Course) -> f64 {
        let gates = &course.gates;
        let n = gates.len();
        let mut idx = course.next;
        let mut prev = init;
        let mut c = 0.0;
        let inv_dt = 1.0 / self.dt;
        let tgt = |i: usize| -> Option<&Gate> {
            if i < n { Some(&gates[i]) } else if n > 0 { Some(&gates[i % n]) } else { None }
        };
        for &p in traj {
            if !p.z.is_finite() { return f64::INFINITY; }
            if let Some(g) = tgt(idx) {
                let to = g.center - prev;
                let dist = to.norm();
                if dist > 1e-6 {
                    let dir = to.scale(1.0 / dist);
                    let vel = (p - prev).scale(inv_dt);
                    c -= self.w_vel * vel.dot(dir);
                    let sp = vel.norm();
                    if sp > self.v_max { c += self.w_speed * (sp - self.v_max).powi(2); }
                }
                let dz = p.z - g.center.z;
                c += self.w_alt * dz * dz;
                if g.crossed(prev, p) { c -= self.gate_reward; idx += 1; }
            }
            prev = p;
        }
        if let (Some(end), Some(g)) = (traj.last(), tgt(idx)) {
            c += self.w_terminal * (*end - g.center).norm();
        }
        c
    }

    fn act(&mut self, veh: &Multirotor<f64, RotorForce>, state: &State<f64>, course: &Course) -> [f64; 4] {
        let (s, t) = (self.cfg_s, self.cfg_t);
        let smear = (1.0 - self.beta * self.beta).sqrt();
        let mut seqs: Vec<Vec<[f64; 4]>> = Vec::with_capacity(s);
        let mut costs = vec![0.0f64; s];
        for i in 0..s {
            let mut e = [0.0f64; 4];
            let mut seq = Vec::with_capacity(t);
            for k in 0..t {
                let mut f = [0.0; 4];
                for d in 0..4 {
                    e[d] = self.beta * e[d] + smear * self.rng.normal();
                    f[d] = (self.nominal[k][d] + self.sigma * e[d]).clamp(0.0, self.f_max);
                }
                seq.push(f);
            }
            // rollout true dynamics
            let mut sx = *state;
            let mut traj = Vec::with_capacity(t);
            for f in &seq {
                sx = veh.step(&sx, f, self.dt);
                traj.push(sx.x);
            }
            let c = self.cost(state.x, &traj, course);
            costs[i] = if c.is_finite() { c } else { 1e12 };
            seqs.push(seq);
        }
        let min_c = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut wsum = 0.0;
        let mut wts = vec![0.0; s];
        for i in 0..s { let w = (-(costs[i] - min_c) / self.lambda).exp(); wts[i] = w; wsum += w; }
        let inv = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };
        for k in 0..t {
            let mut f = [0.0f64; 4];
            for i in 0..s { let w = wts[i] * inv; for d in 0..4 { f[d] += w * seqs[i][k][d]; } }
            for d in 0..4 { f[d] = f[d].clamp(0.0, self.f_max); }
            self.nominal[k] = f;
        }
        let a = self.nominal[0];
        for k in 0..t - 1 { self.nominal[k] = self.nominal[k + 1]; }
        self.nominal[t - 1] = [self.f_hover; 4];
        a
    }
}

fn one_traj(r: &mut Rng, steps: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    let p = sample_drone(r);
    let dt = 0.05;
    let veh: Multirotor<f64, RotorForce> = Multirotor::with_substeps(&p, 2);
    let f_max = p.k_eta * p.rotor_speed_max * p.rotor_speed_max;
    let f_hover = p.mass * 9.81 / 4.0;
    let hov = (p.mass * 9.81 / (4.0 * p.k_eta)).sqrt();
    let mut mppi = RfMppi {
        cfg_s: 128, cfg_t: 20, dt, lambda: 0.4, beta: 0.85,
        sigma: 0.3 * f_hover, f_hover, f_max,
        w_vel: 1.0, w_alt: 3.0, w_terminal: 2.5, gate_reward: 80.0,
        v_max: lerp(r, 1.5, 5.0), w_speed: 8.0,
        nominal: vec![[f_hover; 4]; 20], rng: Rng::new((r.uniform() * 1e9) as u64 + 1),
    };
    let mut course = random_course(r);
    let mut s = State {
        x: Vec3::new(lerp(r, -1.0, 1.0), lerp(r, -1.0, 1.0), lerp(r, 1.0, 2.0)),
        v: Vec3::zero(), q: Quat::new(0.0, 0.0, 0.0, 1.0), w: Vec3::zero(),
        wind: Vec3::new(lerp(r, -1.5, 1.5), lerp(r, -1.5, 1.5), lerp(r, -0.5, 0.5)),
        rotor_speeds: [hov; 4],
    };
    let wind = s.wind;
    let mut states = Vec::with_capacity(steps * 18);
    let mut actions = Vec::with_capacity(steps * 4);
    for _ in 0..steps {
        let a = mppi.act(&veh, &s, &course);
        states.extend_from_slice(&state_to_18(&s));
        for d in 0..4 { actions.push(a[d] as f32); }
        let prev = s.x;
        s = veh.step(&s, &a, dt);
        s.wind = wind;
        course.advance(prev, s.x);
        if !s.x.z.is_finite() || s.x.z.abs() > 30.0 || s.v.norm() > 25.0 { return None; }
    }
    Some((states, actions))
}

fn gen_chunk(seed: u64, chunk: usize, steps: usize) -> (Vec<f32>, Vec<f32>) {
    let mut r = Rng::new(seed);
    let (mut st, mut ac) = (Vec::new(), Vec::new());
    let mut kept = 0;
    while kept < chunk {
        if let Some((s, a)) = one_traj(&mut r, steps) { st.extend_from_slice(&s); ac.extend_from_slice(&a); kept += 1; }
    }
    (st, ac)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_traj: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8000);
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let out = args.get(3).cloned().unwrap_or_else(|| "artifacts/rf_mppi.bin".into());
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);

    let nt = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let base = n_traj / nt;
    let rem = n_traj % nt;
    let chunks: Vec<usize> = (0..nt).map(|i| base + usize::from(i < rem)).collect();
    println!("collecting {n_traj} rotor-force MPPI traj across {nt} threads...");
    let res: Vec<(Vec<f32>, Vec<f32>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = chunks.iter().enumerate().filter(|(_, &c)| c > 0)
            .map(|(i, &c)| { let s = seed.wrapping_mul(1_000_003).wrapping_add(i as u64 + 1); sc.spawn(move || gen_chunk(s, c, steps)) })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut all_s = Vec::new();
    let mut all_a = Vec::new();
    for (s, a) in res { all_s.extend_from_slice(&s); all_a.extend_from_slice(&a); }

    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    f.write_all(&(n_traj as u32).to_le_bytes()).unwrap();
    f.write_all(&(steps as u32).to_le_bytes()).unwrap();
    f.write_all(&4u32.to_le_bytes()).unwrap(); // single action set (4 rotor forces)
    for v in &all_s { f.write_all(&v.to_le_bytes()).unwrap(); }
    for v in &all_a { f.write_all(&v.to_le_bytes()).unwrap(); }
    f.flush().unwrap();
    println!("wrote {out}: {n_traj} traj x {steps} steps, 4 rotor-force cols");
}
