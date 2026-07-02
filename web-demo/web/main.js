// Minimal harness: load the WASM dynamics, fly a hover, expose live "break it" knobs.
import init, { WasmDrone } from "./pkg/racer.js";

async function main() {
  await init();

  // {} -> stock hummingbird; pass overrides like { mass: 0.8, kEta: 7e-6 }.
  const drone = new WasmDrone({}, 8);
  drone.reset_hover(0, 0, 1.5);
  window.drone = drone; // poke from the console

  const dt = 0.02; // 50 Hz sim tick (8 RK4 substeps each)
  const log = document.getElementById("log");
  let t = 0;

  document.getElementById("reset").onclick = () => {
    drone.set_mass(0.5);
    drone.set_wind(0, 0, 0);
    drone.reset_hover(0, 0, 1.5);
    t = 0;
  };
  document.getElementById("heavy").onclick = () => drone.set_mass(1.0);
  document.getElementById("wind").onclick = () => drone.set_wind(6, 0, 0);
  document.getElementById("calm").onclick = () => drone.set_wind(0, 0, 0);

  function frame() {
    // Hold a constant hover thrust. With stock mass it floats; double the mass
    // (button) and it sags — the dynamics respond live.
    const hf = drone.hover_force(); // mass*g/4 for CURRENT mass
    // Use a slightly-too-weak thrust so "double mass" visibly drops it.
    const f = 0.5 * 9.81 / 4; // fixed nominal (NOT recomputed for new mass)
    drone.step_rotor_force([f, f, f, f], dt);
    t += dt;

    const s = drone.state();
    log.textContent =
      `t=${t.toFixed(2)} s\n` +
      `pos   = [${s[0].toFixed(3)}, ${s[1].toFixed(3)}, ${s[2].toFixed(3)}] m\n` +
      `vel   = [${s[3].toFixed(3)}, ${s[4].toFixed(3)}, ${s[5].toFixed(3)}] m/s\n` +
      `quat  = [${s[6].toFixed(3)}, ${s[7].toFixed(3)}, ${s[8].toFixed(3)}, ${s[9].toFixed(3)}]\n` +
      `rotor = ${s[13].toFixed(0)} rad/s   (hover≈${drone.hover_rpm().toFixed(0)})`;
    requestAnimationFrame(frame);
  }
  frame();
}

main();
