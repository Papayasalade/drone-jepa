//! Rotor-force JEPA-MPPI racer: same gate course as race_jepa, but the action is
//! four individual rotor forces (the paper's action space) and the rollout model
//! is the rotor-force SkyJEPA. Self-contained MPPI (the shared MppiController is
//! CTBR-only). Lets us compare rotor-force vs CTBR racing head to head.
//!
//!   cargo run --release --example race_jepa_rotor

use std::collections::VecDeque;
use std::path::Path;

use jepa_rs::{SkyJepa, AD, SD};
use rotor_rs::rng::Rng;
use rotor_rs::{Course, Gate, Multirotor, QuadParamsInput, Quat, RotorForce, State, Vec3};

fn params() -> QuadParamsInput {
    let d = 0.17 * std::f64::consts::FRAC_1_SQRT_2;
    QuadParamsInput {
        mass: 0.5, ixx: 3.65e-3, iyy: 3.68e-3, izz: 7.03e-3, ixy: 0.0, iyz: 0.0, ixz: 0.0,
        rotor_pos: [[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]],
        rotor_directions: [1.0, -1.0, 1.0, -1.0],
        c_dx: 0.5e-2, c_dy: 0.5e-2, c_dz: 1e-2,
        k_eta: 5.57e-6, k_m: 1.36e-7, k_d: 1.19e-4, k_z: 2.32e-4, k_h: 3.39e-3, k_flap: 0.0,
        tau_m: 0.02, rotor_speed_min: 0.0, rotor_speed_max: 1500.0, k_w: 12.0,
    }
}

fn state_to_18(s: &State<f64>) -> [f64; SD] {
    let r = s.q.to_rotmat();
    [
        s.x.x, s.x.y, s.x.z, s.v.x, s.v.y, s.v.z,
        r.rows[0][0], r.rows[0][1], r.rows[0][2],
        r.rows[1][0], r.rows[1][1], r.rows[1][2],
        r.rows[2][0], r.rows[2][1], r.rows[2][2],
        s.w.x, s.w.y, s.w.z,
    ]
}

struct Cfg {
    s: usize, t: usize, dt: f64, lambda: f64, beta: f64,
    sigma: f64, f_hover: f64, f_max: f64,
    w_vel: f64, w_alt: f64, w_terminal: f64, gate_reward: f64, v_max: f64, w_speed: f64,
}

fn rollout_cost(init: Vec3<f64>, traj: &[Vec3<f64>], course: &Course, cfg: &Cfg) -> f64 {
    let gates = &course.gates;
    let n = gates.len();
    let mut idx = course.next;
    let mut prev = init;
    let mut cost = 0.0;
    let inv_dt = 1.0 / cfg.dt;
    let tgt = |i: usize| -> Option<&Gate> {
        if i < n { Some(&gates[i]) } else if course.loop_course && n > 0 { Some(&gates[i % n]) } else { None }
    };
    for &p in traj {
        if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
            return f64::INFINITY;
        }
        if let Some(g) = tgt(idx) {
            let to = g.center - prev;
            let dist = to.norm();
            if dist > 1e-6 {
                let dir = to.scale(1.0 / dist);
                let vel = (p - prev).scale(inv_dt);
                cost -= cfg.w_vel * vel.dot(dir);
                let sp = vel.norm();
                if sp > cfg.v_max { cost += cfg.w_speed * (sp - cfg.v_max) * (sp - cfg.v_max); }
            }
            let dz = p.z - g.center.z;
            cost += cfg.w_alt * dz * dz;
            if g.crossed(prev, p) { cost -= cfg.gate_reward; idx += 1; }
        }
        prev = p;
    }
    if let (Some(end), Some(g)) = (traj.last(), tgt(idx)) {
        cost += cfg.w_terminal * (*end - g.center).norm();
    }
    cost
}

struct RotorMppi {
    model: SkyJepa,
    h: usize,
    cfg: Cfg,
    nominal: Vec<[f64; AD]>,
    rng: Rng,
    states: VecDeque<[f64; SD]>,
    actions: VecDeque<[f64; AD]>,
}

impl RotorMppi {
    fn new(model: SkyJepa, cfg: Cfg, seed: u64) -> Self {
        let h = model.config().history;
        let nominal = vec![[cfg.f_hover; AD]; cfg.t];
        RotorMppi { model, h, cfg, nominal, rng: Rng::new(seed),
                    states: VecDeque::new(), actions: VecDeque::new() }
    }

    fn history(&self, init: &State<f64>) -> Vec<[f64; SD]> {
        let mut hist: Vec<[f64; SD]> = self.states.iter().copied().collect();
        hist.push(state_to_18(init));
        while hist.len() < self.h { hist.insert(0, hist[0]); }
        let s = hist.len() - self.h;
        hist[s..].to_vec()
    }
    fn past(&self) -> Vec<[f64; AD]> {
        let mut pa: Vec<[f64; AD]> = self.actions.iter().copied().collect();
        while pa.len() < self.h - 1 { pa.insert(0, *pa.first().unwrap_or(&[self.cfg.f_hover; AD])); }
        let s = pa.len() - (self.h - 1);
        pa[s..].to_vec()
    }

    fn act(&mut self, state: &State<f64>, course: &Course) -> [f64; AD] {
        let (s, t) = (self.cfg.s, self.cfg.t);
        let hist = self.history(state);
        let past = self.past();
        let smear = (1.0 - self.cfg.beta * self.cfg.beta).sqrt();

        // sample S rotor-force sequences (AR noise per rotor), build windows
        let mut seqs: Vec<Vec<[f64; AD]>> = Vec::with_capacity(s);
        let mut sh = Vec::with_capacity(s);
        let mut aw = Vec::with_capacity(s);
        for _ in 0..s {
            let mut e = [0.0f64; AD];
            let mut seq = Vec::with_capacity(t);
            let mut w: Vec<[f64; AD]> = Vec::with_capacity(self.h + t);
            w.extend_from_slice(&past);
            for k in 0..t {
                let mut f = [0.0; AD];
                for d in 0..AD {
                    e[d] = self.cfg.beta * e[d] + smear * self.rng.normal();
                    f[d] = (self.nominal[k][d] + self.cfg.sigma * e[d]).clamp(0.0, self.cfg.f_max);
                }
                seq.push(f);
                w.push(f);
            }
            let last = *w.last().unwrap();
            while w.len() < self.h + t { w.push(last); }
            seqs.push(seq);
            sh.push(hist.clone());
            aw.push(w);
        }

        let trajs = match self.model.predict_batch(&sh, &aw) {
            Ok(p) => p,
            Err(_) => return self.nominal[0],
        };
        let init = state.x;
        let mut costs = vec![0.0f64; s];
        for i in 0..s {
            let traj: Vec<Vec3<f64>> = trajs[i].iter().map(|x| Vec3::new(x[0], x[1], x[2])).collect();
            let c = rollout_cost(init, &traj, course, &self.cfg);
            costs[i] = if c.is_finite() { c } else { 1e12 };
        }
        let min_c = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut wsum = 0.0;
        let mut weights = vec![0.0; s];
        for i in 0..s {
            let wv = (-(costs[i] - min_c) / self.cfg.lambda).exp();
            weights[i] = wv; wsum += wv;
        }
        let inv = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };
        for k in 0..t {
            let mut f = [0.0f64; AD];
            for i in 0..s {
                let wv = weights[i] * inv;
                for d in 0..AD { f[d] += wv * seqs[i][k][d]; }
            }
            for d in 0..AD { f[d] = f[d].clamp(0.0, self.cfg.f_max); }
            self.nominal[k] = f;
        }
        let action = self.nominal[0];
        for k in 0..t - 1 { self.nominal[k] = self.nominal[k + 1]; }
        self.nominal[t - 1] = [self.cfg.f_hover; AD];
        action
    }

    fn observe(&mut self, state: &State<f64>, action: &[f64; AD]) {
        self.states.push_back(state_to_18(state));
        while self.states.len() > self.h - 1 { self.states.pop_front(); }
        self.actions.push_back(*action);
        while self.actions.len() > self.h - 1 { self.actions.pop_front(); }
    }
}

fn main() {
    let p = params();
    let dt = 0.05;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = SkyJepa::load(
        root.join("weights/skyjepa_rotor.safetensors").to_str().unwrap(),
        root.join("weights/skyjepa_rotor.json").to_str().unwrap(),
    ).expect("load rotor model (export_jepa.py skyjepa_rotor_paper.pt skyjepa_rotor)");
    let t = model.config().horizon;
    let f_max = p.k_eta * p.rotor_speed_max * p.rotor_speed_max;
    let f_hover = p.mass * 9.81 / 4.0;
    println!("rotor-force JEPA-MPPI (H={}, T={t}), f_hover={f_hover:.2} f_max={f_max:.1}", model.config().history);

    let gates = vec![
        Gate::new(Vec3::new(2.0, 0.0, 1.5), Vec3::new(1.0, 0.0, 0.0), 0.9),
        Gate::new(Vec3::new(4.0, 0.8, 1.6), Vec3::new(1.0, 0.0, 0.0), 0.9),
        Gate::new(Vec3::new(6.0, -0.8, 1.4), Vec3::new(1.0, 0.0, 0.0), 0.9),
        Gate::new(Vec3::new(8.0, 0.0, 1.5), Vec3::new(1.0, 0.0, 0.0), 0.9),
    ];
    let n_gates = gates.len();
    let mut course = Course::new(gates, false);

    let cfg = Cfg {
        s: 128, t, dt, lambda: 0.4, beta: 0.85,
        sigma: 0.25 * f_hover, f_hover, f_max,
        w_vel: 1.0, w_alt: 3.0, w_terminal: 2.5, gate_reward: 80.0, v_max: 3.0, w_speed: 8.0,
    };
    let reality: Multirotor<f64, RotorForce> = Multirotor::with_substeps(&p, 8);
    let mut ctrl = RotorMppi::new(model, cfg, 42);

    let hov = (p.mass * 9.81 / (4.0 * p.k_eta)).sqrt();
    let mut s = State {
        x: Vec3::new(0.0, 0.0, 1.5), v: Vec3::zero(), q: Quat::new(0.0, 0.0, 0.0, 1.0),
        w: Vec3::zero(), wind: Vec3::zero(), rotor_speeds: [hov; 4],
    };
    for step in 0..600 {
        let a = ctrl.act(&s, &course);
        let prev_state = s;
        let prev = s.x;
        s = reality.step(&s, &a, dt);
        ctrl.observe(&prev_state, &a);
        if course.advance(prev, s.x) {
            println!("  t={:5.2}s  gate {}/{n_gates}  pos=[{:.2},{:.2},{:.2}]  |v|={:.2}",
                step as f64 * dt, course.gates_passed(), s.x.x, s.x.y, s.x.z, s.v.norm());
        }
        if !s.x.z.is_finite() || s.x.z < -1.0 { println!("  crashed at t={:.2}s", step as f64 * dt); break; }
        if course.finished() { println!("\nFINISHED {n_gates} gates in {:.2}s", (step + 1) as f64 * dt); return; }
    }
    println!("\npassed {}/{n_gates} gates (rotor-force JEPA-MPPI)", course.gates_passed());
}
