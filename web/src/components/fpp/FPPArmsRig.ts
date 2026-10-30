import * as THREE from 'three';

export interface FPPArmsRig {
  armsGroup: THREE.Group;
  leftArm: THREE.Group;
  rightArm: THREE.Group;
  smartwatchScreen: THREE.Mesh;
  smartwatchScreenMat: THREE.MeshBasicMaterial;
  update: (
    delta: number,
    isWalking: boolean,
    isSprinting: boolean,
    isSpeaking: boolean,
    roomLabel?: string,
  ) => void;
}

/**
 * Creates the dynamic smartwatch OLED canvas: real clock, active room status,
 * and a glowing voice waveform during push-to-talk.
 */
function createSmartwatchScreenCanvas(): {
  texture: THREE.CanvasTexture;
  updateScreen: (isSpeaking: boolean, timeSec: number, roomLabel: string) => void;
} {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d')!;

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;

  const updateScreen = (isSpeaking: boolean, timeSec: number, roomLabel: string) => {
    // Matte black OLED glass background
    ctx.fillStyle = '#07090C';
    ctx.fillRect(0, 0, 256, 256);

    // Titanium bezel ring
    ctx.strokeStyle = '#3A3F46';
    ctx.lineWidth = 6;
    ctx.strokeRect(3, 3, 250, 250);
    ctx.strokeStyle = '#C9CDD4';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(11, 11, 234, 234);

    // Header: SVARA OS
    ctx.fillStyle = '#F5F1E8';
    ctx.font = 'bold 17px monospace';
    ctx.fillText('SVARA OS', 22, 38);

    // Status dot
    ctx.beginPath();
    ctx.arc(226, 32, 7, 0, Math.PI * 2);
    ctx.fillStyle = isSpeaking ? '#FF7A5C' : '#5EE6A0';
    ctx.fill();
    ctx.fillStyle = '#9AA5B1';
    ctx.font = '12px monospace';
    ctx.fillText(isSpeaking ? 'REC' : 'LIVE', 200, 35);

    const now = new Date();
    const hrs = String(now.getHours()).padStart(2, '0');
    const mins = String(now.getMinutes()).padStart(2, '0');
    const secs = String(now.getSeconds()).padStart(2, '0');

    if (isSpeaking) {
      // Big REC state + animated waveform bars
      ctx.fillStyle = '#FF8A6B';
      ctx.font = 'bold 26px monospace';
      ctx.fillText('LISTENING', 22, 96);

      const bars = 12;
      const barW = 12;
      const gap = 7;
      const startX = 24;
      for (let i = 0; i < bars; i++) {
        const h = Math.abs(Math.sin(timeSec * 9 + i * 0.85)) * 52 + 10;
        ctx.fillStyle = `rgba(255, 214, 130, ${0.55 + 0.45 * Math.abs(Math.sin(timeSec * 7 + i))})`;
        ctx.fillRect(startX + i * (barW + gap), 128 - h / 2, barW, h);
      }
      ctx.fillStyle = '#FFD9BC';
      ctx.font = '13px monospace';
      ctx.fillText('TAHAN SPASI...', 22, 176);
    } else {
      // Real digital clock
      ctx.fillStyle = '#FFFFFF';
      ctx.font = 'bold 46px monospace';
      ctx.fillText(`${hrs}:${mins}`, 22, 96);
      ctx.fillStyle = '#8B96A3';
      ctx.font = '16px monospace';
      ctx.fillText(`:${secs}`, 168, 96);

      ctx.fillStyle = '#F3C06B';
      ctx.font = 'bold 18px monospace';
      ctx.fillText('● HOME', 22, 136);

      // Active room status
      ctx.fillStyle = '#CFD6DD';
      ctx.font = '15px monospace';
      ctx.fillText('R: ' + (roomLabel || 'GARDEN'), 22, 162);

      // Small status rows
      ctx.fillStyle = '#5E6A76';
      ctx.font = '13px monospace';
      ctx.fillText('MIC: READY', 22, 190);
      ctx.fillText('NET: STABLE', 22, 210);

      // Battery glyph
      ctx.fillStyle = '#5EE6A0';
      ctx.fillRect(196, 196, 40, 16);
      ctx.strokeStyle = '#5EE6A0';
      ctx.lineWidth = 2;
      ctx.strokeRect(196, 196, 40, 16);
      ctx.fillRect(238, 201, 4, 7);
      ctx.fillStyle = '#07090C';
      ctx.fillRect(198, 198, 30, 12);
      ctx.fillStyle = '#5EE6A0';
      ctx.fillRect(198, 198, 24, 12);
    }

    texture.needsUpdate = true;
  };

  return { texture, updateScreen };
}

/**
 * Builds premium FPP arms: charcoal knit sweater sleeves with rolled cuff,
 * natural skin hands and a titanium smartwatch on the left wrist.
 * Attach to the camera for a weighted, organic first-person presence.
 */
export function createFPPArms(): FPPArmsRig {
  const armsGroup = new THREE.Group();

  // --- Materials ---
  const sleeveMat = new THREE.MeshStandardMaterial({
    color: 0x2E2A28, // dark charcoal knit
    roughness: 0.92,
    metalness: 0.02,
  });

  const cuffMat = new THREE.MeshStandardMaterial({
    color: 0x4A433C, // slightly lighter ribbed cuff
    roughness: 0.85,
    metalness: 0.02,
  });

  const skinMat = new THREE.MeshStandardMaterial({
    color: 0xE2B18E, // warm natural skin
    roughness: 0.75,
    metalness: 0.02,
  });

  const titaniumMat = new THREE.MeshStandardMaterial({
    color: 0xC8CBD1,
    roughness: 0.28,
    metalness: 0.9,
  });

  const bandMat = new THREE.MeshStandardMaterial({
    color: 0x1C1F24,
    roughness: 0.5,
    metalness: 0.35,
  });

  const { texture: watchScreenTex, updateScreen: updateWatchScreen } =
    createSmartwatchScreenCanvas();
  const smartwatchScreenMat = new THREE.MeshBasicMaterial({ map: watchScreenTex });

  const makeForearm = (side: 1 | -1): THREE.Group => {
    const arm = new THREE.Group();

    // Knit sleeve forearm (slightly tapered)
    const sleeve = new THREE.Mesh(new THREE.CylinderGeometry(0.052, 0.043, 0.4, 18), sleeveMat);
    sleeve.rotation.x = Math.PI / 2.4;
    sleeve.rotation.z = 0.22 * side;
    sleeve.castShadow = true;
    arm.add(sleeve);

    // Ribbed knit cuff at the wrist
    const cuff = new THREE.Mesh(new THREE.CylinderGeometry(0.046, 0.05, 0.07, 18), cuffMat);
    cuff.rotation.x = Math.PI / 2.4;
    cuff.rotation.z = 0.24 * side;
    cuff.position.set(0.008, 0.085, -0.17);
    arm.add(cuff);

    // Palm
    const palm = new THREE.Mesh(new THREE.BoxGeometry(0.068, 0.03, 0.085), skinMat);
    palm.position.set(0.035 * side, 0.115, -0.24);
    palm.rotation.x = 0.22;
    palm.castShadow = true;
    arm.add(palm);

    // Four curled fingers
    const fingersMat = skinMat;
    for (let f = 0; f < 4; f++) {
      const finger = new THREE.Mesh(new THREE.CapsuleGeometry(0.011, 0.028, 3, 8), fingersMat);
      finger.position.set(0.014 + f * 0.016 * side, 0.098, -0.31);
      finger.rotation.x = 0.55;
      arm.add(finger);
    }

    // Thumb
    const thumb = new THREE.Mesh(new THREE.CapsuleGeometry(0.012, 0.02, 3, 8), skinMat);
    thumb.position.set(0.075 * side, 0.11, -0.27);
    thumb.rotation.z = -0.4 * side;
    thumb.rotation.x = 0.3;
    arm.add(thumb);

    return arm;
  };

  // --- LEFT ARM (with titanium smartwatch) ---
  const leftArm = new THREE.Group();

  const leftForearm = makeForearm(1);
  leftForearm.position.set(0.012, 0.015, 0.02);
  leftArm.add(leftForearm);

  // Watch band (strap around wrist)
  const band = new THREE.Mesh(new THREE.TorusGeometry(0.05, 0.0085, 10, 20), bandMat);
  band.position.set(0.008, 0.085, -0.16);
  band.rotation.x = Math.PI / 2 + 0.15;
  band.rotation.y = 0.2;
  leftArm.add(band);

  // Titanium case
  const watchCase = new THREE.Mesh(new THREE.BoxGeometry(0.058, 0.064, 0.012), titaniumMat);
  watchCase.position.set(0.008, 0.112, -0.205);
  watchCase.rotation.x = -0.12;
  watchCase.rotation.z = -0.08;
  leftArm.add(watchCase);

  // OLED screen (on top of case)
  const smartwatchScreen = new THREE.Mesh(
    new THREE.PlaneGeometry(0.05, 0.056),
    smartwatchScreenMat,
  );
  smartwatchScreen.position.set(0.008, 0.115, -0.199);
  smartwatchScreen.rotation.x = -Math.PI / 2 - 0.1;
  smartwatchScreen.rotation.z = -0.08;
  leftArm.add(smartwatchScreen);

  leftArm.position.set(-0.3, -0.26, -0.46);
  armsGroup.add(leftArm);

  // --- RIGHT ARM (no watch) ---
  const rightArm = new THREE.Group();
  const rightForearm = makeForearm(-1);
  rightForearm.position.set(-0.012, 0.015, 0.02);
  rightArm.add(rightForearm);

  // Knit cuff ring detail
  const rightCuff = new THREE.Mesh(new THREE.CylinderGeometry(0.047, 0.05, 0.06, 18), cuffMat);
  rightCuff.rotation.x = Math.PI / 2.4;
  rightCuff.rotation.z = -0.24;
  rightCuff.position.set(-0.008, 0.085, -0.17);
  rightArm.add(rightCuff);

  rightArm.position.set(0.3, -0.26, -0.46);
  armsGroup.add(rightArm);

  // Base rest poses
  const baseLeftPos = new THREE.Vector3(-0.3, -0.26, -0.46);
  const baseRightPos = new THREE.Vector3(0.3, -0.26, -0.46);

  let walkSwayTimer = 0;
  let breathePhase = 0;
  let speakLerp = 0;
  let lastRoom = '';

  const update = (
    delta: number,
    isWalking: boolean,
    isSprinting: boolean,
    isSpeaking: boolean,
    roomLabel?: string,
  ) => {
    const timeSec = performance.now() / 1000;
    updateWatchScreen(isSpeaking, timeSec, roomLabel || lastRoom);
    if (roomLabel) lastRoom = roomLabel;

    // Walking / sprinting / idle-breathing sway rhythm
    if (isWalking) {
      walkSwayTimer += delta * (isSprinting ? 12.5 : 7.5);
    } else {
      walkSwayTimer += delta * 1.6;
      breathePhase += delta * 0.8;
    }

    const amp = isWalking ? (isSprinting ? 0.042 : 0.024) : 0.006;
    const breath = Math.sin(breathePhase) * 0.004;
    const swayX = Math.cos(walkSwayTimer) * amp;
    const swayY = Math.sin(walkSwayTimer * 2) * amp + breath;

    // Smooth push-to-talk arm raise towards mouth
    const targetSpeak = isSpeaking ? 1 : 0;
    speakLerp = THREE.MathUtils.lerp(speakLerp, targetSpeak, delta * 9);

    leftArm.position.x = baseLeftPos.x + swayX;
    leftArm.position.y = baseLeftPos.y + swayY;
    leftArm.position.z = baseLeftPos.z;

    rightArm.position.x = baseRightPos.x - swayX;
    rightArm.position.y = baseRightPos.y - swayY;
    rightArm.position.z = baseRightPos.z;

    if (speakLerp > 0.001) {
      leftArm.position.x = THREE.MathUtils.lerp(leftArm.position.x, -0.05, speakLerp);
      leftArm.position.y = THREE.MathUtils.lerp(leftArm.position.y, -0.12, speakLerp);
      leftArm.position.z = THREE.MathUtils.lerp(leftArm.position.z, -0.34, speakLerp);
      leftArm.rotation.x = THREE.MathUtils.lerp(leftArm.rotation.x, 0.5, speakLerp);
      leftArm.rotation.y = THREE.MathUtils.lerp(leftArm.rotation.y, 0.38, speakLerp);
      leftArm.rotation.z = THREE.MathUtils.lerp(leftArm.rotation.z, -0.45, speakLerp);
    } else {
      leftArm.rotation.set(0, 0, 0);
    }
  };

  return {
    armsGroup,
    leftArm,
    rightArm,
    smartwatchScreen,
    smartwatchScreenMat,
    update,
  };
}