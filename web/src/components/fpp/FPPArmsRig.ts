import * as THREE from 'three';

export interface FPPArmsRig {
  armsGroup: THREE.Group;
  leftArm: THREE.Group;
  rightArm: THREE.Group;
  smartwatchScreen: THREE.Mesh;
  smartwatchScreenMat: THREE.MeshBasicMaterial;
  update: (delta: number, isWalking: boolean, isSprinting: boolean, isSpeaking: boolean) => void;
}

/**
 * Creates dynamic canvas texture for the smart-home wristwatch.
 * Displays real-time status: time, connection, and glowing waveform during push-to-talk.
 */
function createSmartwatchScreenCanvas(): { texture: THREE.CanvasTexture; updateScreen: (isSpeaking: boolean, timeSec: number) => void } {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d')!;

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;

  const updateScreen = (isSpeaking: boolean, timeSec: number) => {
    // Dark cyber/military background
    ctx.fillStyle = '#081114';
    ctx.fillRect(0, 0, 128, 128);

    // Glowing border bezel
    ctx.strokeStyle = isSpeaking ? '#C4492F' : '#2F6B5E';
    ctx.lineWidth = 4;
    ctx.strokeRect(4, 4, 120, 120);

    // Header: SVARA AI OS
    ctx.fillStyle = '#A9B8B2';
    ctx.font = 'bold 11px monospace';
    ctx.fillText('SVARA OS 2.0', 14, 22);

    // Status indicator dot
    ctx.beginPath();
    ctx.arc(112, 18, 4, 0, Math.PI * 2);
    ctx.fillStyle = isSpeaking ? '#E3593D' : '#3EE08F';
    ctx.fill();

    // Center display
    if (isSpeaking) {
      ctx.fillStyle = '#FF7D60';
      ctx.font = 'bold 12px monospace';
      ctx.fillText('REC VOICE...', 14, 44);

      // Dynamic animated audio waveform bars
      const bars = 8;
      const barWidth = 8;
      const spacing = 4;
      const startX = 18;
      for (let i = 0; i < bars; i++) {
        const height = Math.abs(Math.sin(timeSec * 8 + i * 0.9)) * 32 + 6;
        ctx.fillStyle = '#F2B33D';
        ctx.fillRect(startX + i * (barWidth + spacing), 100 - height, barWidth, height);
      }
    } else {
      // Clock / Ready state
      const now = new Date();
      const hours = String(now.getHours()).padStart(2, '0');
      const mins = String(now.getMinutes()).padStart(2, '0');
      const secs = String(now.getSeconds()).padStart(2, '0');
      ctx.fillStyle = '#E8F1EE';
      ctx.font = 'bold 22px monospace';
      ctx.fillText(`${hours}:${mins}:${secs}`, 12, 62);

      ctx.fillStyle = '#2F6B5E';
      ctx.font = '10px monospace';
      ctx.fillText('STATUS: ONLINE', 14, 88);
      ctx.fillText('MIC: PTT READY', 14, 102);
    }

    texture.needsUpdate = true;
  };

  return { texture, updateScreen };
}

/**
 * Builds realistic first-person arms (wearing sleek tactical jacket sleeves & skin hands)
 * attached to the camera viewport with procedural walking sway and voice-activation lift.
 */
export function createFPPArms(): FPPArmsRig {
  const armsGroup = new THREE.Group();

  // Materials
  const sleeveMat = new THREE.MeshStandardMaterial({
    color: 0x1A252A, // Sleek dark tactical navy sleeve
    roughness: 0.65,
    metalness: 0.1,
  });

  const skinMat = new THREE.MeshStandardMaterial({
    color: 0xC89578, // Realistic human skin tone
    roughness: 0.8,
    metalness: 0.05,
  });

  const metalMat = new THREE.MeshStandardMaterial({
    color: 0x222629, // Watch case gunmetal
    roughness: 0.3,
    metalness: 0.85,
  });

  const { texture: watchScreenTex, updateScreen: updateWatchScreen } = createSmartwatchScreenCanvas();
  const smartwatchScreenMat = new THREE.MeshBasicMaterial({ map: watchScreenTex });

  // --- Left Arm & Smartwatch ---
  const leftArm = new THREE.Group();

  // Left Forearm (sleeve)
  const leftForearm = new THREE.Mesh(new THREE.CylinderGeometry(0.048, 0.042, 0.36, 16), sleeveMat);
  leftForearm.rotation.x = Math.PI / 2.3;
  leftForearm.rotation.z = 0.25;
  leftForearm.position.set(0, 0, 0);
  leftForearm.castShadow = true;
  leftArm.add(leftForearm);

  // Left Wrist & Smartwatch
  const watchBand = new THREE.Mesh(new THREE.CylinderGeometry(0.044, 0.044, 0.045, 16), metalMat);
  watchBand.position.set(0.035, 0.07, -0.16);
  watchBand.rotation.x = Math.PI / 2.3;
  leftArm.add(watchBand);

  // Watch Face Bezel (Square casing)
  const watchBezel = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.012, 0.05), metalMat);
  watchBezel.position.set(0.035, 0.108, -0.16);
  watchBezel.rotation.x = 0.15;
  watchBezel.rotation.z = -0.2;
  leftArm.add(watchBezel);

  // Watch OLED Screen
  const smartwatchScreen = new THREE.Mesh(new THREE.PlaneGeometry(0.042, 0.042), smartwatchScreenMat);
  smartwatchScreen.position.set(0.035, 0.115, -0.16);
  smartwatchScreen.rotation.x = -Math.PI / 2 + 0.15;
  smartwatchScreen.rotation.z = -0.2;
  leftArm.add(smartwatchScreen);

  // Left Hand Palm
  const leftHand = new THREE.Mesh(new THREE.BoxGeometry(0.065, 0.028, 0.08), skinMat);
  leftHand.position.set(0.04, 0.10, -0.23);
  leftHand.rotation.x = 0.25;
  leftHand.rotation.z = 0.1;
  leftArm.add(leftHand);

  // Left Thumb & Fingers curl
  const leftFingers = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.024, 0.05), skinMat);
  leftFingers.position.set(0.04, 0.085, -0.29);
  leftFingers.rotation.x = 0.6;
  leftArm.add(leftFingers);

  // Position left arm in viewport (lower left)
  leftArm.position.set(-0.28, -0.24, -0.45);
  armsGroup.add(leftArm);

  // --- Right Arm ---
  const rightArm = new THREE.Group();

  // Right Forearm (sleeve)
  const rightForearm = new THREE.Mesh(new THREE.CylinderGeometry(0.048, 0.042, 0.36, 16), sleeveMat);
  rightForearm.rotation.x = Math.PI / 2.3;
  rightForearm.rotation.z = -0.25;
  rightForearm.castShadow = true;
  rightArm.add(rightForearm);

  // Right Hand Palm
  const rightHand = new THREE.Mesh(new THREE.BoxGeometry(0.065, 0.028, 0.08), skinMat);
  rightHand.position.set(-0.04, 0.10, -0.23);
  rightHand.rotation.x = 0.25;
  rightHand.rotation.z = -0.1;
  rightArm.add(rightHand);

  // Right Fingers curl
  const rightFingers = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.024, 0.05), skinMat);
  rightFingers.position.set(-0.04, 0.085, -0.29);
  rightFingers.rotation.x = 0.6;
  rightArm.add(rightFingers);

  // Position right arm in viewport (lower right)
  rightArm.position.set(0.28, -0.24, -0.45);
  armsGroup.add(rightArm);

  // Base rest poses
  const baseLeftPos = new THREE.Vector3(-0.28, -0.24, -0.45);
  const baseRightPos = new THREE.Vector3(0.28, -0.24, -0.45);

  let walkSwayTimer = 0;
  let speakLerp = 0;

  const update = (delta: number, isWalking: boolean, isSprinting: boolean, isSpeaking: boolean) => {
    const timeSec = performance.now() / 1000;
    updateWatchScreen(isSpeaking, timeSec);

    // Walking sway physics
    if (isWalking) {
      walkSwayTimer += delta * (isSprinting ? 12 : 7.5);
    } else {
      // Gentle natural human breathing sway when idle
      walkSwayTimer += delta * 1.5;
    }

    const idleAmp = isWalking ? (isSprinting ? 0.035 : 0.02) : 0.005;
    const swayX = Math.cos(walkSwayTimer) * idleAmp;
    const swayY = Math.sin(walkSwayTimer * 2) * idleAmp;

    // Smooth transition into Push-To-Talk arm raise towards mouth/face
    const targetSpeakLerp = isSpeaking ? 1.0 : 0.0;
    speakLerp = THREE.MathUtils.lerp(speakLerp, targetSpeakLerp, delta * 8);

    // Normal arm positions with walking sway
    leftArm.position.x = baseLeftPos.x + swayX;
    leftArm.position.y = baseLeftPos.y + swayY;
    leftArm.position.z = baseLeftPos.z;

    rightArm.position.x = baseRightPos.x - swayX;
    rightArm.position.y = baseRightPos.y - swayY;
    rightArm.position.z = baseRightPos.z;

    // When speaking: Left arm raises wristwatch directly in front of the player's face/mouth
    if (speakLerp > 0.001) {
      // Raise left wrist towards center mouth
      leftArm.position.x = THREE.MathUtils.lerp(leftArm.position.x, -0.06, speakLerp);
      leftArm.position.y = THREE.MathUtils.lerp(leftArm.position.y, -0.09, speakLerp);
      leftArm.position.z = THREE.MathUtils.lerp(leftArm.position.z, -0.32, speakLerp);

      // Rotate wrist so the watch screen tilts up facing the eyes
      leftArm.rotation.x = THREE.MathUtils.lerp(0, 0.45, speakLerp);
      leftArm.rotation.y = THREE.MathUtils.lerp(0, 0.35, speakLerp);
      leftArm.rotation.z = THREE.MathUtils.lerp(0, -0.4, speakLerp);
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
