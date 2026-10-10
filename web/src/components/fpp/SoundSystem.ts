/**
 * SoundSystem.ts — Procedural Web Audio API Sound Effects for FPP Walkthrough
 * Zero external audio assets required. All sounds generated via Web Audio synthesizers:
 * - Dynamic footstep sounds (wood, tile, carpet based on coordinates)
 * - Tactile light switch clicks
 * - Push-to-talk mic chirp (walkie-talkie / radio on/off tone)
 * - Ambient smart-home electronic chime / success chord
 * - Background lounge music synthesizer for hall_music
 */

class SoundSystem {
  private ctx: AudioContext | null = null;
  private musicOscillators: OscillatorNode[] = [];
  private musicGain: GainNode | null = null;
  private isMusicPlaying = false;

  private getContext(): AudioContext {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new AudioCtx();
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
    return this.ctx;
  }

  /**
   * Procedural footstep sound with realistic acoustic envelope and surface filtering.
   */
  public playFootstep(surface: 'wood' | 'tile' | 'carpet' = 'wood') {
    try {
      const ctx = this.getContext();
      const now = ctx.currentTime;

      // Noise buffer for foot friction/impact
      const bufferSize = ctx.sampleRate * 0.06; // 60ms transient
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const output = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        output[i] = Math.random() * 2 - 1;
      }

      const whiteNoise = ctx.createBufferSource();
      whiteNoise.buffer = buffer;

      // Bandpass / Lowpass filter according to surface type
      const filter = ctx.createBiquadFilter();
      if (surface === 'wood') {
        filter.type = 'bandpass';
        filter.frequency.setValueAtTime(320 + Math.random() * 40, now);
        filter.Q.setValueAtTime(2.2, now);
      } else if (surface === 'tile') {
        filter.type = 'highpass';
        filter.frequency.setValueAtTime(800 + Math.random() * 80, now);
      } else {
        // Carpet: very dull, low pass
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(220, now);
      }

      // Quick snappy gain envelope
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.24, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.055);

      whiteNoise.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);

      whiteNoise.start(now);
      whiteNoise.stop(now + 0.06);

      // Low frequency heel thud
      const thudOsc = ctx.createOscillator();
      const thudGain = ctx.createGain();
      thudOsc.type = 'sine';
      thudOsc.frequency.setValueAtTime(surface === 'wood' ? 95 : 120, now);
      thudOsc.frequency.exponentialRampToValueAtTime(35, now + 0.045);

      thudGain.gain.setValueAtTime(0.35, now);
      thudGain.gain.exponentialRampToValueAtTime(0.001, now + 0.045);

      thudOsc.connect(thudGain);
      thudGain.connect(ctx.destination);

      thudOsc.start(now);
      thudOsc.stop(now + 0.05);
    } catch {
      // Audio context might be restricted before first gesture
    }
  }

  /**
   * Tactile relay click sound for toggling smart lights and appliances.
   */
  public playSwitchClick(state: boolean) {
    try {
      const ctx = this.getContext();
      const now = ctx.currentTime;

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'triangle';
      osc.frequency.setValueAtTime(state ? 1200 : 700, now);
      osc.frequency.exponentialRampToValueAtTime(state ? 1600 : 450, now + 0.025);

      gain.gain.setValueAtTime(0.2, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.03);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.035);
    } catch {
      // Audio context restriction guard
    }
  }

  /**
   * Walkie-talkie / PTT activation radio beep.
   */
  public playPTTChirp(start: boolean) {
    try {
      const ctx = this.getContext();
      const now = ctx.currentTime;

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sine';
      if (start) {
        osc.frequency.setValueAtTime(880, now);
        osc.frequency.setValueAtTime(1320, now + 0.035);
      } else {
        osc.frequency.setValueAtTime(1100, now);
        osc.frequency.setValueAtTime(660, now + 0.035);
      }

      gain.gain.setValueAtTime(0.18, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.07);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.075);
    } catch {
      // Audio context restriction guard
    }
  }

  /**
   * Ambient smart-home lounge synth chord when music is activated.
   */
  public setAmbientMusic(active: boolean) {
    try {
      const ctx = this.getContext();

      if (active && !this.isMusicPlaying) {
        this.isMusicPlaying = true;
        this.musicGain = ctx.createGain();
        this.musicGain.gain.setValueAtTime(0.0, ctx.currentTime);
        this.musicGain.gain.linearRampToValueAtTime(0.08, ctx.currentTime + 1.2);
        this.musicGain.connect(ctx.destination);

        // Chord frequencies (Cmaj9 relaxed ambient chord: C4, E4, G4, B4, D5)
        const notes = [261.63, 329.63, 392.00, 493.88, 587.33];
        this.musicOscillators = notes.map((freq, idx) => {
          const osc = ctx.createOscillator();
          osc.type = idx % 2 === 0 ? 'sine' : 'triangle';
          osc.frequency.setValueAtTime(freq, ctx.currentTime);

          // Subtle LFO vibrato
          const lfo = ctx.createOscillator();
          const lfoGain = ctx.createGain();
          lfo.frequency.setValueAtTime(0.2 + idx * 0.05, ctx.currentTime);
          lfoGain.gain.setValueAtTime(2.0, ctx.currentTime);
          lfo.connect(lfoGain);
          lfoGain.connect(osc.frequency);
          lfo.start();

          osc.connect(this.musicGain!);
          osc.start();
          return osc;
        });
      } else if (!active && this.isMusicPlaying) {
        this.isMusicPlaying = false;
        if (this.musicGain) {
          this.musicGain.gain.linearRampToValueAtTime(0.001, ctx.currentTime + 0.8);
          setTimeout(() => {
            this.musicOscillators.forEach((osc) => {
              try { osc.stop(); } catch { /* ignore */ }
            });
            this.musicOscillators = [];
            this.musicGain = null;
          }, 900);
        }
      }
    } catch {
      // Audio context restriction guard
    }
  }
}

export const soundSystem = new SoundSystem();
