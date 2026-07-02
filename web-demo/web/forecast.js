import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import init, { WasmForecast } from "./pkg/rotor_rs.js";

await init();

const seed = Math.floor(Math.random() * 1e9);
const sim = new WasmForecast(seed);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9fc7e8);
scene.fog = new THREE.Fog(0x9fc7e8, 28, 95);

const camera = new THREE.PerspectiveCamera(56, innerWidth / innerHeight, 0.1, 220);
camera.up.set(0, 0, 1);
camera.position.set(-8, -13, 7);
camera.lookAt(0, 0, 2.0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 2);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();
addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

scene.add(new THREE.HemisphereLight(0xd9ecff, 0x6d7a45, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(-7, -8, 14);
sun.castShadow = true;
scene.add(sun);

// MUST stay identical to forecast_terrain_height in rotor-rs/src/wasm.rs
// (the sim enforces clearance against the Rust copy; this one only renders).
function terrainHeight(x, y) {
  const ridge = y - 0.35 * x - 3.5; // diagonal ridge line
  return -0.4
    + 1.3 * Math.sin(0.10 * x + 0.4) * Math.cos(0.085 * y - 0.8)
    + 0.7 * Math.sin(0.23 * x) * Math.sin(0.19 * y + 1.1)
    + 1.2 * Math.exp(-(ridge * ridge) / 22.0)
    + 0.25 * Math.sin(0.55 * x + 0.3 * y);
}

function makeTerrain() {
  const geom = new THREE.PlaneGeometry(90, 90, 120, 120);
  const pos = geom.attributes.position;
  const colors = [];
  const color = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = terrainHeight(x, y);
    pos.setZ(i, z);
    const mix = THREE.MathUtils.clamp((z + 2.6) / 5.7, 0, 1);
    color.setRGB(0.18 + 0.22 * mix, 0.35 + 0.36 * mix, 0.22 + 0.08 * mix);
    colors.push(color.r, color.g, color.b);
  }
  geom.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geom.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0.0 });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.receiveShadow = true;
  scene.add(mesh);

  const grid = new THREE.GridHelper(90, 45, 0xffffff, 0xffffff);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = 0.02;
  grid.material.transparent = true;
  grid.material.opacity = 0.12;
  scene.add(grid);
}
makeTerrain();

function scatterRocks() {
  const mat = new THREE.MeshStandardMaterial({ color: 0x6f725f, roughness: 1 });
  for (let i = 0; i < 42; i++) {
    const x = (Math.random() - 0.5) * 60;
    const y = (Math.random() - 0.5) * 60;
    if (Math.hypot(x, y) < 5) continue;
    const z = terrainHeight(x, y) + 0.03;
    const rock = new THREE.Mesh(new THREE.DodecahedronGeometry(0.12 + Math.random() * 0.18, 0), mat);
    rock.position.set(x, y, z);
    rock.scale.z = 0.35 + Math.random() * 0.7;
    scene.add(rock);
  }
}
scatterRocks();

function makeDrone() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.22, 0.16, 0.07),
    new THREE.MeshStandardMaterial({ color: 0xb38aff, emissive: 0x2d135c, emissiveIntensity: 0.6, roughness: 0.45 })
  );
  body.castShadow = true;
  g.add(body);
  const armMat = new THREE.LineBasicMaterial({ color: 0xe4d8ff });
  const rotorMat = new THREE.MeshStandardMaterial({ color: 0xf2edff, transparent: true, opacity: 0.82 });
  for (let i = 0; i < 4; i++) {
    const sx = i < 2 ? 1 : -1;
    const sy = i === 0 || i === 3 ? 1 : -1;
    const p = new THREE.Vector3(sx * 0.34, sy * 0.34, 0.03);
    const arm = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), p]), armMat);
    g.add(arm);
    const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.018, 20), rotorMat);
    rotor.position.copy(p);
    rotor.rotation.x = Math.PI / 2;
    g.add(rotor);
  }
  const glow = new THREE.PointLight(0xb38aff, 1.0, 5);
  g.add(glow);
  scene.add(g);
  return g;
}
const drone = makeDrone();

// payload: a dark crate slung ~0.5 m under the body, shown when attached
const payloadMesh = (() => {
  const g = new THREE.Group();
  const box = new THREE.Mesh(
    new THREE.BoxGeometry(0.22, 0.22, 0.18),
    new THREE.MeshStandardMaterial({ color: 0x8a5a2b, roughness: 0.9 })
  );
  box.position.z = -0.52;
  box.castShadow = true;
  g.add(box);
  const rope = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, -0.05), new THREE.Vector3(0, 0, -0.43)]),
    new THREE.LineBasicMaterial({ color: 0xcccccc })
  );
  g.add(rope);
  g.visible = false;
  drone.add(g);
  return g;
})();

// drone mesh scales with the sampled frame (arm 0.25 m = the nominal bigquad)
function fitDroneScale() {
  drone.scale.setScalar(sim.drone_arm() / 0.25);
}

// wind streaks: advected line segments in a box around the drone, visible
// whenever the sim reports wind
const WIND_N = 160;
const windHeads = new Float32Array(WIND_N * 3);
const windGeom = new THREE.BufferGeometry();
windGeom.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(WIND_N * 6), 3));
const windLines = new THREE.LineSegments(
  windGeom,
  new THREE.LineBasicMaterial({ color: 0xbfe3ff, transparent: true, opacity: 0.5 })
);
windLines.frustumCulled = false;
windLines.visible = false;
scene.add(windLines);
const WIND_BOX = { x: 34, y: 34, z: 10 };
function seedWind(center) {
  for (let i = 0; i < WIND_N; i++) {
    windHeads[i * 3] = center.x + (Math.random() - 0.5) * WIND_BOX.x;
    windHeads[i * 3 + 1] = center.y + (Math.random() - 0.5) * WIND_BOX.y;
    windHeads[i * 3 + 2] = center.z + (Math.random() - 0.5) * WIND_BOX.z;
  }
}
function updateWind(center, dtSec) {
  const w = sim.wind();
  const mag = Math.hypot(w[0], w[1], w[2]);
  const on = mag > 0.05;
  if (on && !windLines.visible) seedWind(center);
  windLines.visible = on;
  if (!on) return;
  const len = Math.min(0.25 + 0.22 * mag, 1.6);
  const ux = w[0] / mag, uy = w[1] / mag, uz = w[2] / mag;
  const pos = windGeom.attributes.position.array;
  for (let i = 0; i < WIND_N; i++) {
    let x = windHeads[i * 3] + w[0] * dtSec * 2.2;
    let y = windHeads[i * 3 + 1] + w[1] * dtSec * 2.2;
    let z = windHeads[i * 3 + 2] + w[2] * dtSec * 2.2;
    // wrap into the box around the drone
    if (x - center.x > WIND_BOX.x / 2) x -= WIND_BOX.x; else if (center.x - x > WIND_BOX.x / 2) x += WIND_BOX.x;
    if (y - center.y > WIND_BOX.y / 2) y -= WIND_BOX.y; else if (center.y - y > WIND_BOX.y / 2) y += WIND_BOX.y;
    if (z - center.z > WIND_BOX.z / 2) z -= WIND_BOX.z; else if (center.z - z > WIND_BOX.z / 2) z += WIND_BOX.z;
    windHeads[i * 3] = x; windHeads[i * 3 + 1] = y; windHeads[i * 3 + 2] = z;
    pos[i * 6] = x; pos[i * 6 + 1] = y; pos[i * 6 + 2] = z;
    pos[i * 6 + 3] = x - ux * len; pos[i * 6 + 4] = y - uy * len; pos[i * 6 + 5] = z - uz * len;
  }
  windGeom.attributes.position.needsUpdate = true;
}
let followBase = new THREE.Vector3(0, 0, 2.4);
const desiredBase = new THREE.Vector3();
const followDelta = new THREE.Vector3();
function trackCamera(s) {
  // Preserve user panning: OrbitControls pans by moving both camera.position and
  // controls.target. Follow the drone with lag so its motion stays visible.
  const panOffset = controls.target.clone().sub(followBase);
  desiredBase.set(s[0], s[1], s[2]);
  followBase.lerp(desiredBase, 0.045);
  followDelta.copy(followBase).add(panOffset).sub(controls.target);
  controls.target.add(followDelta);
  camera.position.add(followDelta);
}

function makeTrail(color, n, opacity, width = 1) {
  const pos = new Float32Array(n * 3);
  const geom = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity, linewidth: width });
  const line = new THREE.Line(geom, mat);
  scene.add(line);
  return { pos, geom, line, n, head: 0 };
}

const realTrail = makeTrail(0xb38aff, 180, 0.62);
function makeTube(color, radius, opacity) {
  const mat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), mat);
  scene.add(mesh);
  return mesh;
}

const trueTube = makeTube(0xffffff, 0.052, 0.54);
trueTube.userData.radius = 0.052;
const jepaTube = makeTube(0x5dff9a, 0.075, 0.86);
jepaTube.userData.radius = 0.075;

function pushTrail(tr, s) {
  tr.pos[tr.head * 3] = s[0];
  tr.pos[tr.head * 3 + 1] = s[1];
  tr.pos[tr.head * 3 + 2] = s[2];
  tr.head = (tr.head + 1) % tr.n;
  tr.geom.attributes.position.needsUpdate = true;
}

function setTube(mesh, pts) {
  const count = pts.length / 3;
  if (count < 2) return;
  const p = [];
  for (let i = 0; i < count; i++) {
    p.push(new THREE.Vector3(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]));
  }
  const curve = new THREE.CatmullRomCurve3(p);
  const geom = new THREE.TubeGeometry(curve, Math.max(8, count * 3), mesh.userData.radius, 9, false);
  mesh.geometry.dispose();
  mesh.geometry = geom;
}

function meanError(a, b) {
  const count = Math.min(a.length, b.length) / 3;
  if (count < 1) return 0;
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const dx = a[i * 3] - b[i * 3];
    const dy = a[i * 3 + 1] - b[i * 3 + 1];
    const dz = a[i * 3 + 2] - b[i * 3 + 2];
    sum += Math.hypot(dx, dy, dz);
  }
  return sum / count;
}

const stat = document.getElementById("stat");
const HORIZON_STEPS = 10; // 0.5 s ghost
const showJepaEl = document.getElementById("showJepa");
document.getElementById("payload").onclick = () => sim.toggle_payload();
document.getElementById("gust").onclick = () => sim.gust();
document.getElementById("calm").onclick = () => sim.calm();
document.getElementById("newDrone").onclick = () => { sim.new_drone(); fitDroneScale(); };
fitDroneScale();
let windLast = performance.now();
const speedEl = document.getElementById("speed");
speedEl.oninput = () => {
  const k = parseFloat(speedEl.value);
  sim.set_speed(k);
  document.getElementById("speedV").textContent = k.toFixed(2) + "x";
};

let acc = 0;
let last = performance.now();
const DT = 0.05;
const TIME_SCALE = 1.0;
let forecastDirty = true;
let prevState = sim.state();
let currState = prevState.slice();
showJepaEl.addEventListener("change", () => { jepaTube.visible = showJepaEl.checked; });

function interpState(a, b, t) {
  const out = b.slice();
  for (let i = 0; i < 6; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  for (let i = 10; i < 17; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  // Quaternion nlerp is enough for this small 20 Hz render interpolation.
  let dot = a[6] * b[6] + a[7] * b[7] + a[8] * b[8] + a[9] * b[9];
  const sign = dot < 0 ? -1 : 1;
  out[6] = a[6] + (sign * b[6] - a[6]) * t;
  out[7] = a[7] + (sign * b[7] - a[7]) * t;
  out[8] = a[8] + (sign * b[8] - a[8]) * t;
  out[9] = a[9] + (sign * b[9] - a[9]) * t;
  const qn = Math.hypot(out[6], out[7], out[8], out[9]) || 1;
  out[6] /= qn; out[7] /= qn; out[8] /= qn; out[9] /= qn;
  return out;
}

function frame(now) {
  requestAnimationFrame(frame);
  now = now || performance.now();
  acc += Math.min((now - last) / 1000, 0.1) * TIME_SCALE;
  last = now;
  let n = 0;
  while (acc >= DT && n < 3) {
    prevState = currState;
    currState = sim.step();
    acc -= DT;
    n++;
  }
  if (n > 0) forecastDirty = true;
  const s = interpState(prevState, currState, Math.min(acc / DT, 1));

  drone.position.set(s[0], s[1], s[2]);
  drone.quaternion.set(s[6], s[7], s[8], s[9]);
  payloadMesh.visible = sim.payload_attached();
  updateWind(drone.position, Math.min((now - windLast) / 1000, 0.1));
  windLast = now;
  trackCamera(s);
  pushTrail(realTrail, s);
  if (forecastDirty) {
    forecastDirty = false;
    const steps = HORIZON_STEPS;
    const f = sim.forecast(steps);
    const truth = f.slice(0, steps * 3);
    const jepa = f.slice(steps * 3, steps * 6);
    setTube(trueTube, truth);
    setTube(jepaTube, jepa);
    jepaTube.visible = showJepaEl.checked;
    const jepaErr = meanError(jepa, truth);

    stat.innerHTML =
      `drone ${sim.drone_mass().toFixed(2)} kg · arm ${(sim.drone_arm() * 100).toFixed(0)} cm<br>` +
      `ghost horizon ${(steps * DT).toFixed(2)}s: ` +
      `<span style="color:#5dff9a">JEPA ${jepaErr.toFixed(2)}m</span>`;
  }

  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(frame);

window.forecast = sim;
