// Autonomous MPPI gate racer rendered in Three.js, driven by the rotor-rs WASM
// sim. Sim is z-up; we render directly in sim coords (camera up = +z). The sim
// quaternion is [x,y,z,w] = Three's convention, so orientation maps 1:1.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import init, { WasmRacer } from "./pkg/rotor_rs.js";

await init();
// Reproducible tracks: a `?seed=N` in the URL replays the exact same course
// sequence; otherwise pick a random one.
const urlSeed = new URLSearchParams(location.search).get("seed");
const seed = urlSeed !== null && urlSeed !== "" ? Number(urlSeed) >>> 0 : Math.floor(Math.random() * 1e9);
const racer = new WasmRacer(seed);
racer.set_randomize(true);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f14);
scene.fog = new THREE.Fog(0x0b0f14, 24, 70);

const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.1, 300);
camera.up.set(0, 0, 1); // z-up world
// 3/4 view framing the full 0-15 m sky column (gates up high, floor below)
camera.position.set(-7.0, -19.0, 11.0);
camera.lookAt(6.0, 0, 8.0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

// mouse-controlled camera: drag to orbit, scroll to zoom, right-drag to pan
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(6.0, 0, 8.0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();
addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

scene.add(new THREE.AmbientLight(0x668, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(5, -8, 12);
scene.add(sun);

// solid ground/floor at z=0 — the thing a "crash" hits (the sim has no collision,
// but showing it makes altitude and ground-dives legible). Grid sits just above it.
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(120, 120),
  new THREE.MeshStandardMaterial({ color: 0x1b2c47, emissive: 0x0a1424, metalness: 0.1, roughness: 0.9 })
);
floor.position.z = -0.02; // just below the grid lines
scene.add(floor);
const grid = new THREE.GridHelper(120, 120, 0x3d68a0, 0x24405f);
grid.rotation.x = Math.PI / 2; // GridHelper is xz by default -> make it xy
scene.add(grid);

// --- gates (torus rings oriented by normal); rebuilt when the course changes ---
const gateGroup = new THREE.Group();
scene.add(gateGroup);
let gateMeshes = [];
let gateSig = "";
function syncGates() {
  const d = racer.gates(); // [cx,cy,cz, nx,ny,nz, r] * N
  const sig = d.length + ":" + (d[0] ?? 0).toFixed(2) + "," + (d[1] ?? 0).toFixed(2);
  if (sig === gateSig) return;
  gateSig = sig;
  gateGroup.clear();
  gateMeshes = [];
  for (let i = 0; i < d.length; i += 7) {
    const [cx, cy, cz, nx, ny, nz, r] = d.slice(i, i + 7);
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(r, 0.07, 14, 44),
      new THREE.MeshStandardMaterial({ color: 0x3df, emissive: 0x0a3a55, metalness: 0.3, roughness: 0.5 })
    );
    ring.position.set(cx, cy, cz);
    ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(nx, ny, nz).normalize());
    gateGroup.add(ring);
    gateMeshes.push(ring);
  }
}
syncGates();

// --- drones: arms + rotors are repositioned to the ACTUAL (possibly asymmetric)
// frame each time the drone changes (random-drone mode). ---
function makeDrone(bodyColor, emissive, rotorColor, glowColor) {
  const drone = new THREE.Group();
  drone.add(new THREE.Mesh(
    new THREE.BoxGeometry(0.12, 0.12, 0.05),
    new THREE.MeshStandardMaterial({ color: bodyColor, emissive, emissiveIntensity: 0.6, metalness: 0.3, roughness: 0.5 })
  ));
  drone.add(new THREE.PointLight(glowColor, 1.0, 6));
  const rotors = [], arms = [];
  for (let i = 0; i < 4; i++) {
    const rot = new THREE.Mesh(
      new THREE.CylinderGeometry(0.07, 0.07, 0.02, 16),
      new THREE.MeshStandardMaterial({ color: rotorColor, transparent: true, opacity: 0.75 })
    );
    rot.rotation.x = Math.PI / 2;
    drone.add(rot); rotors.push(rot);
    const arm = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color: bodyColor })
    );
    drone.add(arm); arms.push(arm);
  }
  drone.scale.setScalar(2.2);
  drone.userData = { rotors, arms };
  scene.add(drone);
  return drone;
}
// reposition a drone's 4 rotors + arms to `pos` = [x,y,z]×4 (the real rotor offsets)
function syncDroneShape(drone, pos) {
  const { rotors, arms } = drone.userData;
  for (let i = 0; i < 4; i++) {
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2] + 0.03;
    rotors[i].position.set(x, y, z);
    arms[i].geometry.setFromPoints([new THREE.Vector3(0, 0, 0.02), new THREE.Vector3(x, y, z)]);
    arms[i].geometry.attributes.position.needsUpdate = true;
  }
}
const droneRl = makeDrone(0xb38aff, 0x5a2da8, 0xc9f, 0x9040ff); // RL policy
const droneRf = makeDrone(0x4dd0e1, 0x0a6a7a, 0x9ef, 0x40e0ff); // JEPA (rotor-force)
const allDrones = [droneRl, droneRf];
let shapeSig = "";
function syncShapes() { // all share the one (current) drone
  const rp = racer.drone_rotor_pos();
  const sig = rp.map((v) => v.toFixed(3)).join(",");
  if (sig === shapeSig) return;
  shapeSig = sig;
  for (const d of allDrones) syncDroneShape(d, rp);
}
syncShapes();

// trails (one per drone)
const trailN = 120;
function makeTrail(color) {
  const pos = new Float32Array(trailN * 3);
  const line = new THREE.Line(
    new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(pos, 3)),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 })
  );
  scene.add(line);
  return { pos, line, head: 0 };
}
const trailRl = makeTrail(0x9040ff);
const trailRf = makeTrail(0x40e0ff);
function pushTrail(tr, s) {
  tr.pos[tr.head * 3] = s[0];
  tr.pos[tr.head * 3 + 1] = s[1];
  tr.pos[tr.head * 3 + 2] = s[2];
  tr.head = (tr.head + 1) % trailN;
  tr.line.geometry.attributes.position.needsUpdate = true;
}

const stat = document.getElementById("stat");


// real-time pacing: advance the 20 Hz sim by wall-clock, capped so it never
// runs faster than real-time (and never spirals if a frame is slow).
let acc = 0;
let last = performance.now();
const DT = 0.05;
function frame(now) {
  requestAnimationFrame(frame);
  now = now || performance.now();
  acc += Math.min((now - last) / 1000, 0.1);
  last = now;
  let n = 0;
  while (acc >= DT && n < 3) { racer.step(); acc -= DT; n++; }
  if (acc > 0.25) acc = 0; // drop backlog
  syncGates(); // course may have changed (new race)
  syncShapes(); // drone shape may have changed (random-drone mode)
  const rl = racer.rl_state(); // RL drone, 17
  const rf = racer.rf_state(); // JEPA (rotor-force) drone, 17

  droneRl.position.set(rl[0], rl[1], rl[2]);
  droneRl.quaternion.set(rl[6], rl[7], rl[8], rl[9]);
  droneRf.position.set(rf[0], rf[1], rf[2]);
  droneRf.quaternion.set(rf[6], rf[7], rf[8], rf[9]);
  pushTrail(trailRl, rl);
  pushTrail(trailRf, rf);

  // Highlight the next visible target. Prefer the RL policy while it races; if
  // it has finished, follow the JEPA drone still trying the course.
  const ng = !racer.rl_hovering() ? racer.rl_next_gate() : racer.rf_next_gate();
  gateMeshes.forEach((m, i) => m.material.emissive.setHex(i === ng % gateMeshes.length ? 0x2a8 : 0x0a3a55));

  const spR = Math.hypot(rl[3], rl[4], rl[5]);
  const spF = Math.hypot(rf[3], rf[4], rf[5]);
  const won = `<span style="color:#9f9">WON ✓</span>`;
  const lost = (n) => (n ? ` <span style="color:#f88">(${n}× lost)</span>` : "");
  const rlStatus = racer.rl_hovering() ? won : `gate ${racer.rl_next_gate() + 1}${lost(racer.rl_respawns())}`;
  const rfStatus = racer.rf_hovering() ? won : `gate ${racer.rf_next_gate() + 1}${lost(racer.rf_respawns())}`;
  stat.innerHTML =
    `<span style="color:#b38aff">●</span> RL policy&nbsp;&nbsp; ${rlStatus} &nbsp; ${spR.toFixed(1)} m/s &nbsp; wins:<b>${racer.rl_wins()}</b><br>` +
    `<span style="color:#4dd0e1">●</span> JEPA rotor&nbsp; ${rfStatus} &nbsp; ${spF.toFixed(1)} m/s &nbsp; wins:<b>${racer.rf_wins()}</b>`;

  const m = racer.drone_mass();
  dronev.textContent = `-> ${m.toFixed(2)} kg · arm ${(racer.drone_arm() * 100).toFixed(0)} cm`;

  controls.update();
  renderer.render(scene, camera);
}

requestAnimationFrame(frame);

// --- controls ---
const dronev = document.getElementById("dronev");
document.getElementById("reset").onclick = () => racer.reset();

// --- JEPA planner sample counts ---
function wireSamples(sliderId, labelId, getter, setter) {
  const slider = document.getElementById(sliderId);
  const label = document.getElementById(labelId);
  slider.value = getter();
  label.textContent = getter();
  slider.addEventListener("input", () => {
    label.textContent = slider.value;
    setter(+slider.value);
  });
}
wireSamples("rfSamp", "rfSampV", () => racer.rf_samples(), (n) => racer.set_rf_samples(n));

// --- reproducible seed / debug snapshot ---
const seedEl = document.getElementById("seed");
seedEl.value = String(racer.seed() >>> 0);
document.getElementById("loadseed").onclick = () => {
  const v = (Number(seedEl.value) >>> 0) || 0;
  location.search = "?seed=" + v;
};
document.getElementById("rndseed").onclick = () => {
  location.search = "?seed=" + (Math.floor(Math.random() * 1e9) >>> 0);
};
document.getElementById("copydbg").onclick = async () => {
  const json = racer.debug_snapshot();
  try { await navigator.clipboard.writeText(json); } catch (_) {}
  const b = document.getElementById("copydbg");
  const old = b.textContent;
  b.textContent = "copied!";
  console.log("debug snapshot:", json);
  setTimeout(() => (b.textContent = old), 1200);
};

window.racer = racer;
