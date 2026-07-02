//! Differentiable Kinematic Integrator (CTBR), ported from drone_jepa/model/dki.py.
//! Parameter-free physics: semi-implicit Euler + SO(3) exp (Rodrigues). f64.
//! State layout: [pos3, vel3, R(row-major 9), omega3].

type M3 = [[f64; 3]; 3];

#[inline]
fn matmul(a: &M3, b: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}

/// Rodrigues exp of an axis-angle vector phi -> rotation matrix.
fn exp_so3(phi: [f64; 3]) -> M3 {
    let theta = (phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]).sqrt();
    let theta = theta.max(1e-8);
    let k = [phi[0] / theta, phi[1] / theta, phi[2] / theta];
    // K = hat(k)
    let kk: M3 = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
    let kk2 = matmul(&kk, &kk);
    let s = theta.sin();
    let c = 1.0 - theta.cos();
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let eye = if i == j { 1.0 } else { 0.0 };
            r[i][j] = eye + s * kk[i][j] + c * kk2[i][j];
        }
    }
    r
}

/// One rotor-force DKI step. thrust = sum of 4 rotor forces; omega_dot = K @ a,
/// where K is the prober's 3x4 residual angular-accel map (row-major [f64;12]).
pub fn step_rotor_force(
    x: &[f64; 18],
    a: &[f64; 4],
    dvdot: &[f64; 3],
    k: &[f64; 12],
    dt: f64,
    m_nominal: f64,
    g: f64,
) -> [f64; 18] {
    let thrust = a[0] + a[1] + a[2] + a[3];
    let omega_dot = [
        k[0] * a[0] + k[1] * a[1] + k[2] * a[2] + k[3] * a[3],
        k[4] * a[0] + k[5] * a[1] + k[6] * a[2] + k[7] * a[3],
        k[8] * a[0] + k[9] * a[1] + k[10] * a[2] + k[11] * a[3],
    ];
    step_collective(x, thrust, dvdot, &omega_dot, dt, m_nominal, g)
}

/// One CTBR DKI step. dvdot = residual lin-accel, omega_dot = angular accel.
pub fn step_ctbr(
    x: &[f64; 18],
    a: &[f64; 4],
    dvdot: &[f64; 3],
    omega_dot: &[f64; 3],
    dt: f64,
    m_nominal: f64,
    g: f64,
) -> [f64; 18] {
    step_collective(x, a[0], dvdot, omega_dot, dt, m_nominal, g) // ctbr: thrust = a[0]
}

/// Shared collective-thrust DKI integration (semi-implicit Euler + SO(3) exp).
fn step_collective(
    x: &[f64; 18],
    thrust: f64,
    dvdot: &[f64; 3],
    omega_dot: &[f64; 3],
    dt: f64,
    m_nominal: f64,
    g: f64,
) -> [f64; 18] {
    let p = [x[0], x[1], x[2]];
    let v = [x[3], x[4], x[5]];
    let r: M3 = [
        [x[6], x[7], x[8]],
        [x[9], x[10], x[11]],
        [x[12], x[13], x[14]],
    ];
    let omega = [x[15], x[16], x[17]];

    // translational: gravity + nominal thrust along body-z (world) + residual
    let inv_mass = 1.0 / m_nominal;
    let body_z_world = [r[0][2], r[1][2], r[2][2]]; // 3rd column
    let vdot = [
        thrust * inv_mass * body_z_world[0] + dvdot[0],
        thrust * inv_mass * body_z_world[1] + dvdot[1],
        -g + thrust * inv_mass * body_z_world[2] + dvdot[2],
    ];
    let v_next = [v[0] + vdot[0] * dt, v[1] + vdot[1] * dt, v[2] + vdot[2] * dt];
    let p_next = [p[0] + v_next[0] * dt, p[1] + v_next[1] * dt, p[2] + v_next[2] * dt];

    // rotational
    let omega_next = [
        omega[0] + omega_dot[0] * dt,
        omega[1] + omega_dot[1] * dt,
        omega[2] + omega_dot[2] * dt,
    ];
    // R_next = R @ exp_so3(omega * dt)  (current omega, body frame)
    let exp = exp_so3([omega[0] * dt, omega[1] * dt, omega[2] * dt]);
    let r_next = matmul(&r, &exp);

    [
        p_next[0], p_next[1], p_next[2],
        v_next[0], v_next[1], v_next[2],
        r_next[0][0], r_next[0][1], r_next[0][2],
        r_next[1][0], r_next[1][1], r_next[1][2],
        r_next[2][0], r_next[2][1], r_next[2][2],
        omega_next[0], omega_next[1], omega_next[2],
    ]
}
