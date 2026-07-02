//! Gate-racing course on top of the dynamics: gates the drone must fly through
//! in sequence, with pass/miss detection. Independent of the controller — drives
//! classical MPPI, JEPA-MPPI, or an RL policy alike.

use crate::linalg::Vec3;

/// A racing gate: a disc of radius `radius` centered at `center`, whose plane has
/// unit `normal` (the direction you fly through). "Passing" = crossing the plane
/// near the center, traveling roughly along `normal`.
#[derive(Clone, Copy, Debug)]
pub struct Gate {
    pub center: Vec3<f64>,
    pub normal: Vec3<f64>,
    pub radius: f64,
}

impl Gate {
    pub fn new(center: Vec3<f64>, normal: Vec3<f64>, radius: f64) -> Self {
        let n = normal.norm();
        let normal = if n > 1e-9 { normal.scale(1.0 / n) } else { Vec3::new(1.0, 0.0, 0.0) };
        Gate { center, normal, radius }
    }

    /// Signed distance from `p` to the gate plane (along `normal`).
    #[inline]
    pub fn signed_plane_dist(&self, p: Vec3<f64>) -> f64 {
        (p - self.center).dot(self.normal)
    }

    /// Perpendicular (in-plane) distance from the gate center, i.e. how far off
    /// the bullseye the point projects.
    #[inline]
    pub fn radial_offset(&self, p: Vec3<f64>) -> f64 {
        let d = p - self.center;
        let along = d.dot(self.normal);
        let perp = d - self.normal.scale(along);
        perp.norm()
    }

    /// Did the segment `prev -> cur` cross the gate plane within the radius (from
    /// either side)? Two-way so randomly-oriented gates are passable regardless of
    /// approach direction.
    pub fn crossed(&self, prev: Vec3<f64>, cur: Vec3<f64>) -> bool {
        let dp = self.signed_plane_dist(prev);
        let dc = self.signed_plane_dist(cur);
        // a sign change means the segment crossed the plane
        if dp * dc > 0.0 || (dp - dc).abs() < 1e-12 {
            return false;
        }
        // interpolate crossing point, check it's inside the ring
        let t = dp / (dp - dc); // in [0,1]
        let hit = prev + (cur - prev).scale(t);
        self.radial_offset(hit) <= self.radius
    }
}

/// An ordered sequence of gates plus the index of the next one to pass.
#[derive(Clone, Debug)]
pub struct Course {
    pub gates: Vec<Gate>,
    pub next: usize,
    pub laps_completed: usize,
    pub loop_course: bool,
}

impl Course {
    pub fn new(gates: Vec<Gate>, loop_course: bool) -> Self {
        Course { gates, next: 0, laps_completed: 0, loop_course }
    }

    /// The gate currently being targeted (None if the course is finished).
    pub fn target(&self) -> Option<Gate> {
        self.gates.get(self.next).copied()
    }

    pub fn finished(&self) -> bool {
        !self.loop_course && self.next >= self.gates.len()
    }

    /// Update progress given the drone moved `prev -> cur`. Returns true if a gate
    /// was passed on this segment.
    pub fn advance(&mut self, prev: Vec3<f64>, cur: Vec3<f64>) -> bool {
        if self.finished() {
            return false;
        }
        let g = self.gates[self.next];
        if g.crossed(prev, cur) {
            self.next += 1;
            if self.loop_course && self.next >= self.gates.len() {
                self.next = 0;
                self.laps_completed += 1;
            }
            return true;
        }
        false
    }

    pub fn gates_passed(&self) -> usize {
        self.laps_completed * self.gates.len() + self.next
    }
}
