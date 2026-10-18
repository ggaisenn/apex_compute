import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { getNaiveAccess } from "../executionTrace";
import type { Kernel, MatrixSize, RunMode, Theme } from "../types";

interface ThreeSceneProps {
  size: MatrixSize;
  theme: Theme;
  kernel: Kernel;
  mode: RunMode;
  step: number;
  playing: boolean;
  speed: number;
  compact?: boolean;
}

const palette = {
  cyan: 0x52e5d2,
  amber: 0xffbd66,
  violet: 0xb491ff,
  red: 0xff7c72,
  ink: 0x0b0f0e,
};

interface MatrixLayout {
  dimension: MatrixSize;
  columns: number;
  rows: number;
  cellSize: number;
  gap: number;
  rowLift: number;
  cellHeight: number;
  outerPad: number;
  outerWidth: number;
  outerDepth: number;
}

interface MatrixBounds {
  left: number;
  right: number;
  front: number;
  back: number;
  bottom: number;
  top: number;
  centerX: number;
  centerY: number;
  centerZ: number;
}

interface NaivePulse {
  mesh: THREE.Mesh;
  row: number;
}

const getMatrixLayout = (dimension: MatrixSize, compact: boolean): MatrixLayout => {
  const visualGrid: Record<MatrixSize, { columns: number; rows: number }> = {
    64: { columns: compact ? 4 : 5, rows: compact ? 3 : 4 },
    128: { columns: compact ? 5 : 6, rows: compact ? 4 : 5 },
    256: { columns: compact ? 6 : 8, rows: compact ? 4 : 6 },
    512: { columns: compact ? 8 : 10, rows: compact ? 5 : 7 },
  };
  const cellSize = compact ? 0.16 : 0.2;
  const gap = compact ? 0.028 : 0.035;
  const { columns, rows } = visualGrid[dimension];
  const rowLift = cellSize * 0.22;
  const cellHeight = cellSize * 0.55;
  const outerPad = cellSize + 0.12;

  return {
    dimension,
    columns,
    rows,
    cellSize,
    gap,
    rowLift,
    cellHeight,
    outerPad,
    outerWidth: (columns - 1) * (cellSize + gap) + outerPad,
    outerDepth: (rows - 1) * (cellSize + gap) + outerPad,
  };
};

const getCellPosition = (origin: THREE.Vector3, layout: MatrixLayout, row: number, column: number) =>
  new THREE.Vector3(
    origin.x + column * (layout.cellSize + layout.gap),
    origin.y + (layout.rows - row) * layout.rowLift,
    origin.z + row * (layout.cellSize + layout.gap),
  );

const addMatrix = (
  group: THREE.Group,
  label: string,
  matrixName: "A" | "B" | "C",
  origin: THREE.Vector3,
  color: number,
  tileOffset: number,
  layout: MatrixLayout,
): MatrixBounds => {
  const { cellHeight, cellSize, columns, gap, outerPad, rowLift, rows } = layout;
  const cellGeometry = new THREE.BoxGeometry(cellSize, cellHeight, cellSize);

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const tileIndex = (row * columns + column) % 12;
      const material = new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.12,
        metalness: 0.2,
        roughness: 0.62,
        transparent: true,
        opacity: 0.76,
      });
      const mesh = new THREE.Mesh(cellGeometry, material);
      mesh.userData.tileIndex = tileIndex;
      mesh.userData.tileOffset = tileOffset;
      mesh.userData.matrixColor = color;
      mesh.userData.matrixName = matrixName;
      mesh.userData.visualRow = row;
      mesh.userData.visualColumn = column;
      mesh.userData.baseOpacity = 0.76;
      mesh.position.copy(getCellPosition(origin, layout, row, column));
      mesh.castShadow = true;
      group.add(mesh);
    }
  }

  const left = origin.x - outerPad / 2;
  const right = origin.x + (columns - 1) * (cellSize + gap) + outerPad / 2;
  const front = origin.z - outerPad / 2;
  const back = origin.z + (rows - 1) * (cellSize + gap) + outerPad / 2;
  const bottom = origin.y + rowLift - cellHeight / 2 - 0.04;
  const top = origin.y + rows * rowLift + cellHeight / 2 + 0.04;
  const outline = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(right - left, top - bottom, back - front)),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.32 }),
  );
  outline.position.set((left + right) / 2, (bottom + top) / 2, (front + back) / 2);
  group.add(outline);
  const text = document.createElement("canvas");
  text.width = 256;
  text.height = 72;
  const context = text.getContext("2d");
  if (context) {
    context.fillStyle = group.userData.theme === "light" ? "#42534b" : "#9eaaa4";
    context.font = "600 22px monospace";
    context.fillText(label, 12, 34);
  }
  const labelSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(text), transparent: true }));
  labelSprite.scale.set(layout.columns > 6 ? 1.03 : 0.86, 0.25, 1);
  labelSprite.position.set(left + 0.05, top + 0.13, front);
  group.add(labelSprite);

  return {
    left,
    right,
    front,
    back,
    bottom,
    top,
    centerX: (left + right) / 2,
    centerY: (bottom + top) / 2,
    centerZ: (front + back) / 2,
  };
};

const addFlowLine = (group: THREE.Group, start: THREE.Vector3, end: THREE.Vector3, color: number) => {
  const control = start.clone().lerp(end, 0.5);
  control.y += 0.18;
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(new THREE.CatmullRomCurve3([start, control, end]).getPoints(18)),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 }),
  );
  line.userData.executionFlow = true;
  group.add(line);
  const orb = new THREE.Mesh(
    new THREE.SphereGeometry(0.065, 12, 12),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.95 }),
  );
  orb.userData.flowStart = start.clone();
  orb.userData.flowEnd = end.clone();
  orb.userData.executionFlow = true;
  group.add(orb);
};

const addNaiveCachePulses = (
  group: THREE.Group,
  origin: THREE.Vector3,
  layout: MatrixLayout,
): NaivePulse[] => {
  const pulseGeometry = new THREE.SphereGeometry(layout.cellSize * 0.3, 12, 12);
  const pulses: NaivePulse[] = [];

  for (let row = 0; row < layout.rows; row += 1) {
    const pulse = new THREE.Mesh(
      pulseGeometry,
      new THREE.MeshBasicMaterial({
        color: palette.red,
        transparent: true,
        opacity: 0.08,
        depthWrite: false,
      }),
    );
    pulse.userData.naiveCachePulse = true;
    pulse.position.copy(getCellPosition(origin, layout, row, 0));
    pulse.position.y += layout.cellHeight * 0.72;
    group.add(pulse);
    pulses.push({ mesh: pulse, row });
  }

  return pulses;
};

export const ThreeScene = ({ size, theme, kernel, mode, step, playing, speed, compact = false }: ThreeSceneProps) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef({ step, playing, kernel, mode, speed });

  useEffect(() => {
    stateRef.current = { step, playing, kernel, mode, speed };
  }, [step, playing, kernel, mode, speed]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;

    const scene = new THREE.Scene();
    const sceneBackground = theme === "light" ? 0xf1f6f3 : palette.ink;
    scene.background = new THREE.Color(sceneBackground);
    scene.fog = new THREE.Fog(sceneBackground, 5.5, 14);
    const camera = new THREE.PerspectiveCamera(33, 1, 0.1, 100);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 2;
    controls.maxDistance = 14;

    scene.add(new THREE.HemisphereLight(theme === "light" ? 0xffffff : 0xb7fff4, theme === "light" ? 0xd3e0da : 0x131a18, theme === "light" ? 2.2 : 1.5));
    const keyLight = new THREE.DirectionalLight(theme === "light" ? 0xffffff : 0xbefcf1, theme === "light" ? 2.5 : 3.2);
    keyLight.position.set(3, 5, 4);
    keyLight.castShadow = true;
    scene.add(keyLight);
    const fillLight = new THREE.PointLight(0x735df5, theme === "light" ? 2.2 : 5, 8);
    fillLight.position.set(-2, 2, 1);
    scene.add(fillLight);

    const stage = new THREE.Group();
    stage.userData.theme = theme;
    stage.rotation.y = -0.08;
    scene.add(stage);
    const layout = getMatrixLayout(size, compact);
    const rowGap = compact ? 0.42 : 0.58;
    const rowWidth = layout.outerWidth * 2 + rowGap;
    const rowLeft = -rowWidth / 2;
    const matrixOriginZ = -layout.outerDepth / 2 + layout.outerPad / 2;
    const matrixOriginOffsetX = layout.outerPad / 2;
    const aOrigin = new THREE.Vector3(rowLeft + matrixOriginOffsetX, 0, matrixOriginZ);
    const bOrigin = new THREE.Vector3(rowLeft + layout.outerWidth + rowGap + matrixOriginOffsetX, 0, matrixOriginZ);
    const cOrigin = new THREE.Vector3(-layout.outerWidth / 2 + matrixOriginOffsetX, compact ? -0.52 : -0.68, matrixOriginZ);
    const a = addMatrix(
      stage,
      `A · ${size}×${size}`,
      "A",
      aOrigin,
      palette.cyan,
      0,
      layout,
    );
    const b = addMatrix(
      stage,
      `B · ${size}×${size}`,
      "B",
      bOrigin,
      palette.violet,
      4,
      layout,
    );
    const c = addMatrix(
      stage,
      `C · ${size}×${size}`,
      "C",
      cOrigin,
      palette.amber,
      8,
      layout,
    );
    const naivePulses = addNaiveCachePulses(stage, bOrigin, layout);
    addFlowLine(stage, new THREE.Vector3(a.right + 0.04, a.centerY, a.centerZ), new THREE.Vector3(c.left + 0.16, c.top, c.centerZ), palette.cyan);
    addFlowLine(stage, new THREE.Vector3(b.left - 0.04, b.centerY, b.centerZ), new THREE.Vector3(c.right - 0.16, c.top, c.centerZ), palette.violet);

    const fitCameraToStage = () => {
      const bounds = new THREE.Box3().setFromObject(stage);
      const center = bounds.getCenter(new THREE.Vector3());
      const size = bounds.getSize(new THREE.Vector3());
      const aspect = Math.max(camera.aspect, 0.1);
      const verticalFov = THREE.MathUtils.degToRad(camera.fov);
      const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * aspect);
      const verticalDistance = size.y / (2 * Math.tan(verticalFov / 2));
      const horizontalDistance = size.x / (2 * Math.tan(horizontalFov / 2));
      const distance = Math.max(verticalDistance, horizontalDistance, size.z * 0.72) * 1.7;
      const direction = new THREE.Vector3(1.12, 0.82, 1.32).normalize();

      camera.position.copy(center).add(direction.multiplyScalar(Math.max(distance, 3.5)));
      camera.near = 0.1;
      camera.far = Math.max(100, distance * 6);
      camera.updateProjectionMatrix();
      controls.target.copy(center);
      controls.minDistance = Math.max(2, distance * 0.55);
      controls.maxDistance = Math.max(14, distance * 3.5);
      controls.update();
    };

    const resize = () => {
      const width = mount.clientWidth;
      const height = mount.clientHeight;
      renderer.setSize(width, height, false);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
      fitCameraToStage();
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    const clock = new THREE.Clock();
    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();
      const current = stateRef.current;
      if (current.playing) {
        stage.rotation.y = -0.08 + Math.sin(elapsed * 0.22) * 0.025;
        stage.position.y = Math.sin(elapsed * 0.8) * 0.025;
      }
      const activeColor = current.mode === "naive" ? palette.red : current.kernel === "avx512" ? palette.amber : palette.cyan;
      const pulseRate = 0.35 + (current.speed / 100) * 1.2;
      const naiveAccess = getNaiveAccess(current.step, layout.rows, layout.columns);
      const naivePulsePhase = current.playing
        ? (elapsed * pulseRate + current.step / 12) % 1
        : naiveAccess.innerRow / layout.rows;
      stage.children.forEach((object) => {
        if (object.userData.executionFlow && !(object instanceof THREE.Mesh)) {
          object.visible = current.mode !== "naive";
          return;
        }
        if (object instanceof THREE.Mesh && object.userData.flowStart && object.userData.flowEnd) {
          object.visible = current.mode !== "naive";
          const pulse = current.playing ? (elapsed * pulseRate + current.step * 0.08) % 1 : (current.step % 12) / 12;
          object.position.lerpVectors(object.userData.flowStart, object.userData.flowEnd, pulse);
          const material = object.material as THREE.MeshBasicMaterial;
          material.color.set(activeColor);
        } else if (object instanceof THREE.Mesh && object.userData.tileIndex !== undefined) {
          const material = object.material as THREE.MeshStandardMaterial;
          if (current.mode === "naive") {
            const matrixName = object.userData.matrixName as "A" | "B" | "C";
            const row = Number(object.userData.visualRow);
            const column = Number(object.userData.visualColumn);
            const isActive = matrixName === "A"
              ? row === naiveAccess.outputRow && column === naiveAccess.innerColumn
              : matrixName === "B"
                ? row === naiveAccess.innerRow && column === naiveAccess.outputColumn
                : row === naiveAccess.outputRow && column === naiveAccess.outputColumn;
            const isStridedBAccess = matrixName === "B" && column === naiveAccess.outputColumn;
            const matrixColor = Number(object.userData.matrixColor);
            const cellColor = isActive ? (matrixName === "B" ? palette.red : activeColor) : matrixColor;
            material.color.set(cellColor);
            material.emissive.set(cellColor);
            material.emissiveIntensity = isActive ? (current.playing ? 0.95 : 0.65) : isStridedBAccess ? 0.24 : 0.12;
            material.opacity = isActive ? 1 : isStridedBAccess ? 0.58 : Number(object.userData.baseOpacity);
          } else {
            const activeTile = (current.step + Number(object.userData.tileOffset ?? 0)) % 12;
            const isActive = Number(object.userData.tileIndex) === activeTile;
            material.color.set(isActive ? activeColor : Number(object.userData.matrixColor));
            material.emissive.set(isActive ? activeColor : Number(object.userData.matrixColor));
            material.emissiveIntensity = isActive ? (current.playing ? 0.95 : 0.65) : 0.12;
            material.opacity = isActive ? 1 : Number(object.userData.baseOpacity);
          }
        }
      });
      naivePulses.forEach(({ mesh, row }) => {
        const material = mesh.material as THREE.MeshBasicMaterial;
        const pulsePosition = getCellPosition(bOrigin, layout, row, naiveAccess.outputColumn);
        mesh.position.copy(pulsePosition);
        mesh.position.y += layout.cellHeight * 0.72;
        if (current.mode !== "naive") {
          material.opacity = 0;
          return;
        }

        const phase = row / layout.rows;
        const distance = Math.min(Math.abs(naivePulsePhase - phase), 1 - Math.abs(naivePulsePhase - phase));
        const glow = current.playing ? Math.max(0, 1 - distance * layout.rows * 1.7) : row === naiveAccess.innerRow ? 1 : 0;
        material.opacity = current.playing ? 0.08 + glow * 0.92 : glow;
        const scale = 0.65 + glow * 0.8;
        mesh.scale.setScalar(scale);
      });
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      stage.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.LineSegments || object instanceof THREE.Line) {
          object.geometry.dispose();
          const material = object.material;
          if (Array.isArray(material)) material.forEach((entry) => entry.dispose());
          else material.dispose();
        }
      });
    };
  }, [compact, size, theme]);

  return <div aria-label="3D matrix execution visualizer" className="three-stage" ref={mountRef} />;
};

export default ThreeScene;
