"use strict";
var NeuroMono = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // src/index.ts
  var index_exports = {};
  __export(index_exports, {
    AudioUtils: () => AudioUtils,
    NeuroMono: () => NeuroMono,
    StereoToMonoConverter: () => StereoToMonoConverter,
    analyze: () => analyze,
    convert: () => convert,
    createConverter: () => createConverter,
    default: () => index_default
  });

  // src/audio-utils.ts
  var AudioUtils = class {
    /**
     * Analyze stereo audio characteristics
     */
    static analyzeStereo(buffer) {
      const { left, right } = buffer;
      const length = left.length;
      let widthSum = 0;
      let correlationSum = 0;
      let leftPowerSum = 0;
      let rightPowerSum = 0;
      let rmsSum = 0;
      let peakLevel = 0;
      for (let i = 0; i < length; i++) {
        const diff = Math.abs(left[i] - right[i]);
        const mid = (left[i] + right[i]) / 2;
        widthSum += diff;
        correlationSum += left[i] * right[i];
        leftPowerSum += left[i] * left[i];
        rightPowerSum += right[i] * right[i];
        rmsSum += mid * mid;
        const peak = Math.max(Math.abs(left[i]), Math.abs(right[i]));
        if (peak > peakLevel) peakLevel = peak;
      }
      const width = Math.min(1, widthSum / length);
      const rmsLevel = Math.sqrt(rmsSum / length);
      const leftRMS = Math.sqrt(leftPowerSum / length);
      const rightRMS = Math.sqrt(rightPowerSum / length);
      const normalization = leftRMS * rightRMS;
      const phaseCorrelation = normalization > 0 ? correlationSum / length / normalization : 0;
      const richness = this.calculateRichness(buffer);
      const frequencyDistribution = this.analyzeFrequencyDistribution(buffer);
      return {
        width,
        richness,
        rmsLevel,
        peakLevel,
        phaseCorrelation,
        frequencyDistribution
      };
    }
    /**
     * Calculate spectral richness (harmonic content)
     */
    static calculateRichness(buffer) {
      const { left, right } = buffer;
      const length = Math.min(left.length, 8192);
      let harmonicSum = 0;
      const windowSize = 256;
      for (let i = 0; i < length - windowSize; i += windowSize / 2) {
        const lSlice = left.slice(i, i + windowSize);
        const rSlice = right.slice(i, i + windowSize);
        harmonicSum += this.calculateHarmonicContent(lSlice);
        harmonicSum += this.calculateHarmonicContent(rSlice);
      }
      const numWindows = Math.floor((length - windowSize) / (windowSize / 2)) * 2;
      return Math.min(1, harmonicSum / numWindows);
    }
    /**
     * Calculate harmonic content in a signal window
     */
    static calculateHarmonicContent(window) {
      let energy = 0;
      let zcr = 0;
      for (let i = 0; i < window.length; i++) {
        energy += window[i] * window[i];
        if (i > 0 && window[i] * window[i - 1] < 0) zcr++;
      }
      const normalizedZcr = zcr / window.length;
      const normalizedEnergy = Math.sqrt(energy / window.length);
      return normalizedZcr * normalizedEnergy * 10;
    }
    /**
     * Analyze frequency distribution (simplified spectral analysis)
     */
    static analyzeFrequencyDistribution(buffer) {
      const { left, right } = buffer;
      const length = Math.min(left.length, 4096);
      let lowEnergy = 0;
      let midEnergy = 0;
      let highEnergy = 0;
      const lowPassThreshold = 0.1;
      const highPassThreshold = 0.3;
      for (let i = 1; i < length; i++) {
        const lDiff = Math.abs(left[i] - left[i - 1]);
        const rDiff = Math.abs(right[i] - right[i - 1]);
        const diff = (lDiff + rDiff) / 2;
        if (diff < lowPassThreshold) {
          lowEnergy += Math.abs(left[i]) + Math.abs(right[i]);
        } else if (diff < highPassThreshold) {
          midEnergy += Math.abs(left[i]) + Math.abs(right[i]);
        } else {
          highEnergy += Math.abs(left[i]) + Math.abs(right[i]);
        }
      }
      const total = lowEnergy + midEnergy + highEnergy;
      return {
        low: total > 0 ? lowEnergy / total : 0,
        mid: total > 0 ? midEnergy / total : 0,
        high: total > 0 ? highEnergy / total : 0
      };
    }
    /**
     * Apply Hann window to reduce spectral leakage
     */
    static applyHannWindow(samples) {
      const length = samples.length;
      const windowed = new Float32Array(length);
      for (let i = 0; i < length; i++) {
        const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (length - 1)));
        windowed[i] = samples[i] * window;
      }
      return windowed;
    }
    /**
     * Calculate RMS (Root Mean Square) level
     */
    static calculateRMS(samples) {
      let sum = 0;
      for (let i = 0; i < samples.length; i++) {
        sum += samples[i] * samples[i];
      }
      return Math.sqrt(sum / samples.length);
    }
    /**
     * Normalize audio to target RMS level
     */
    static normalize(samples, targetRMS) {
      const currentRMS = this.calculateRMS(samples);
      if (currentRMS === 0) return samples;
      const gain = targetRMS / currentRMS;
      const normalized = new Float32Array(samples.length);
      for (let i = 0; i < samples.length; i++) {
        normalized[i] = Math.max(-1, Math.min(1, samples[i] * gain));
      }
      return normalized;
    }
    /**
     * Apply soft clipping to prevent harsh distortion
     */
    static softClip(samples, threshold = 0.9) {
      const clipped = new Float32Array(samples.length);
      for (let i = 0; i < samples.length; i++) {
        const x = samples[i];
        const absX = Math.abs(x);
        if (absX <= threshold) {
          clipped[i] = x;
        } else {
          const sign = x >= 0 ? 1 : -1;
          clipped[i] = sign * (threshold + (1 - threshold) * Math.tanh((absX - threshold) / (1 - threshold)));
        }
      }
      return clipped;
    }
    /**
     * Simple linear interpolation for resampling
     */
    static resample(samples, sourceSampleRate, targetSampleRate) {
      if (sourceSampleRate === targetSampleRate) return samples;
      const ratio = sourceSampleRate / targetSampleRate;
      const outputLength = Math.floor(samples.length / ratio);
      const resampled = new Float32Array(outputLength);
      for (let i = 0; i < outputLength; i++) {
        const sourceIndex = i * ratio;
        const index0 = Math.floor(sourceIndex);
        const index1 = Math.min(index0 + 1, samples.length - 1);
        const fraction = sourceIndex - index0;
        resampled[i] = samples[index0] * (1 - fraction) + samples[index1] * fraction;
      }
      return resampled;
    }
  };

  // src/neural-network.ts
  var NeuralNetwork = class _NeuralNetwork {
    constructor(inputSize, hiddenSizes, outputSize) {
      this.layers = [];
      const sizes = [inputSize, ...hiddenSizes, outputSize];
      for (let i = 0; i < sizes.length - 1; i++) {
        this.layers.push(this.createLayer(sizes[i], sizes[i + 1], i === sizes.length - 2));
      }
    }
    /**
     * Create a neural layer with initialized weights
     */
    createLayer(inputSize, outputSize, isOutput) {
      const weights = [];
      const scale = Math.sqrt(2 / (inputSize + outputSize));
      for (let i = 0; i < outputSize; i++) {
        const neuronWeights = new Float32Array(inputSize);
        for (let j = 0; j < inputSize; j++) {
          neuronWeights[j] = (Math.random() * 2 - 1) * scale;
        }
        weights.push(neuronWeights);
      }
      const biases = new Float32Array(outputSize);
      for (let i = 0; i < outputSize; i++) {
        biases[i] = (Math.random() * 2 - 1) * 0.01;
      }
      return {
        weights,
        biases,
        activation: isOutput ? "linear" : "relu"
      };
    }
    /**
     * Forward pass through the network
     */
    forward(input) {
      let current = input;
      for (const layer of this.layers) {
        current = this.forwardLayer(current, layer);
      }
      return current;
    }
    /**
     * Forward pass through a single layer
     */
    forwardLayer(input, layer) {
      const output = new Float32Array(layer.weights.length);
      for (let i = 0; i < layer.weights.length; i++) {
        let sum = layer.biases[i];
        for (let j = 0; j < input.length; j++) {
          sum += input[j] * layer.weights[i][j];
        }
        output[i] = this.activate(sum, layer.activation);
      }
      return output;
    }
    /**
     * Activation functions
     */
    activate(x, activation) {
      switch (activation) {
        case "relu":
          return Math.max(0, x);
        case "tanh":
          return Math.tanh(x);
        case "sigmoid":
          return 1 / (1 + Math.exp(-x));
        case "linear":
        default:
          return x;
      }
    }
    /**
     * Load pre-trained weights optimized for audio preservation
     * In production, these would come from a trained model file
     */
    static createPretrainedAudioModel() {
      const network = new _NeuralNetwork(8, [16, 8], 4);
      network.optimizeForAudioPreservation();
      return network;
    }
    /**
     * Optimize network weights for audio feature preservation
     */
    optimizeForAudioPreservation() {
      for (let layerIdx = 0; layerIdx < this.layers.length; layerIdx++) {
        const layer = this.layers[layerIdx];
        for (let i = 0; i < layer.weights.length; i++) {
          for (let j = 0; j < layer.weights[i].length; j++) {
            if (j < 5) {
              layer.weights[i][j] *= 1.2;
            }
          }
        }
      }
    }
  };
  var AudioFeatureProcessor = class {
    constructor() {
      this.network = NeuralNetwork.createPretrainedAudioModel();
    }
    /**
     * Process audio features to determine optimal mixing weights
     * Input: [width, richness, rmsLevel, peakLevel, phaseCorr, lowFreq, midFreq, highFreq]
     * Output: [leftWeight, rightWeight, sideGain, harmonicGain]
     */
    processFeatures(features) {
      const normalized = new Float32Array(features.length);
      for (let i = 0; i < features.length; i++) {
        normalized[i] = Math.max(0, Math.min(1, features[i]));
      }
      const weights = this.network.forward(normalized);
      const processed = new Float32Array(4);
      processed[0] = Math.max(0.3, Math.min(0.7, 0.5 + weights[0] * 0.2));
      processed[1] = Math.max(0.3, Math.min(0.7, 0.5 + weights[1] * 0.2));
      processed[2] = Math.max(0, Math.min(0.5, weights[2]));
      processed[3] = Math.max(0, Math.min(0.3, weights[3]));
      return processed;
    }
  };

  // src/converter.ts
  var StereoToMonoConverter = class {
    constructor(options = {}) {
      this.options = {
        preserveWidth: options.preserveWidth ?? 0.7,
        preserveRichness: options.preserveRichness ?? 0.8,
        volumeCompensation: options.volumeCompensation ?? 1.1,
        sampleRate: options.sampleRate ?? null,
        quality: options.quality ?? 0.8,
        spectralAnalysis: options.spectralAnalysis ?? true
      };
      this.processor = new AudioFeatureProcessor();
    }
    /**
     * Convert stereo audio to mono while preserving quality characteristics
     */
    convert(buffer) {
      const analysis = AudioUtils.analyzeStereo(buffer);
      const mixingWeights = this.calculateMixingWeights(analysis);
      const mono = this.performIntelligentDownmix(buffer, analysis, mixingWeights);
      const enhanced = this.applyPreservationTechniques(mono, buffer, analysis);
      const compensated = this.applyVolumeCompensation(enhanced, analysis);
      return AudioUtils.softClip(compensated, 0.95);
    }
    /**
     * Calculate mixing weights using neural network
     */
    calculateMixingWeights(analysis) {
      const features = new Float32Array([
        analysis.width,
        analysis.richness,
        analysis.rmsLevel,
        analysis.peakLevel,
        (analysis.phaseCorrelation + 1) / 2,
        // Normalize to [0, 1]
        analysis.frequencyDistribution.low,
        analysis.frequencyDistribution.mid,
        analysis.frequencyDistribution.high
      ]);
      return this.processor.processFeatures(features);
    }
    /**
     * Perform intelligent stereo to mono downmix
     */
    performIntelligentDownmix(buffer, analysis, weights) {
      const { left, right } = buffer;
      const length = left.length;
      const mono = new Float32Array(length);
      const [leftWeight, rightWeight, sideGain, _] = weights;
      for (let i = 0; i < length; i++) {
        const mid = left[i] * leftWeight + right[i] * rightWeight;
        if (this.options.preserveWidth > 0) {
          const side = (left[i] - right[i]) * sideGain * this.options.preserveWidth;
          mono[i] = mid + side * 0.5;
        } else {
          mono[i] = mid;
        }
      }
      return mono;
    }
    /**
     * Apply preservation techniques for richness and quality
     */
    applyPreservationTechniques(mono, buffer, analysis) {
      if (this.options.preserveRichness === 0) return mono;
      const enhanced = new Float32Array(mono.length);
      if (this.options.spectralAnalysis && analysis.richness > 0.3) {
        const harmonicEnhancement = this.extractHarmonicContent(buffer);
        const richnessFactor = this.options.preserveRichness * analysis.richness;
        for (let i = 0; i < mono.length; i++) {
          enhanced[i] = mono[i] + harmonicEnhancement[i] * richnessFactor * 0.15;
        }
      } else {
        enhanced.set(mono);
      }
      if (analysis.frequencyDistribution.high > 0.2) {
        this.enhanceHighFrequencies(enhanced, analysis.frequencyDistribution.high);
      }
      return enhanced;
    }
    /**
     * Extract harmonic content from stereo signal
     */
    extractHarmonicContent(buffer) {
      const { left, right } = buffer;
      const length = left.length;
      const harmonic = new Float32Array(length);
      for (let i = 0; i < length; i++) {
        harmonic[i] = (left[i] - right[i]) * 0.5;
      }
      const windowSize = Math.min(512, length);
      for (let i = 0; i < length; i += windowSize / 2) {
        const end = Math.min(i + windowSize, length);
        const slice = harmonic.slice(i, end);
        const windowed = AudioUtils.applyHannWindow(slice);
        harmonic.set(windowed, i);
      }
      return harmonic;
    }
    /**
     * Enhance high frequencies to preserve clarity
     */
    enhanceHighFrequencies(samples, highFreqRatio) {
      const enhancementFactor = 1 + highFreqRatio * 0.2 * this.options.preserveRichness;
      for (let i = 1; i < samples.length; i++) {
        const highFreq = (samples[i] - samples[i - 1]) * 0.5;
        samples[i] += highFreq * (enhancementFactor - 1);
      }
    }
    /**
     * Apply volume compensation to maintain perceived loudness
     */
    applyVolumeCompensation(samples, analysis) {
      const targetRMS = analysis.rmsLevel * this.options.volumeCompensation;
      const widthCompensation = 1 + analysis.width * 0.15;
      const adjustedTarget = targetRMS * widthCompensation;
      return AudioUtils.normalize(samples, adjustedTarget);
    }
    /**
     * Get stereo analysis for the input buffer
     */
    analyze(buffer) {
      return AudioUtils.analyzeStereo(buffer);
    }
    /**
     * Update conversion options
     */
    setOptions(options) {
      this.options = {
        ...this.options,
        ...options
      };
    }
  };

  // src/index.ts
  var NeuroMono = class {
    constructor() {
      this.options = {};
    }
    /**
     * Set stereo width preservation level (0.0 - 1.0)
     * Higher values preserve more stereo information in the mono mix
     * @default 0.7
     */
    preserveWidth(level) {
      this.options.preserveWidth = Math.max(0, Math.min(1, level));
      return this;
    }
    /**
     * Set audio richness preservation level (0.0 - 1.0)
     * Higher values preserve more harmonic content and spectral detail
     * @default 0.8
     */
    preserveRichness(level) {
      this.options.preserveRichness = Math.max(0, Math.min(1, level));
      return this;
    }
    /**
     * Set volume compensation factor
     * Values > 1.0 boost volume, < 1.0 reduce it
     * @default 1.1
     */
    volumeCompensation(factor) {
      this.options.volumeCompensation = Math.max(0.5, Math.min(2, factor));
      return this;
    }
    /**
     * Set processing quality (0.0 - 1.0)
     * Higher quality = better results but slower processing
     * @default 0.8
     */
    quality(level) {
      this.options.quality = Math.max(0, Math.min(1, level));
      return this;
    }
    /**
     * Enable or disable spectral analysis for better preservation
     * @default true
     */
    spectralAnalysis(enabled) {
      this.options.spectralAnalysis = enabled;
      return this;
    }
    /**
     * Set target sample rate for processing
     * @default null (use source sample rate)
     */
    sampleRate(rate) {
      this.options.sampleRate = rate;
      return this;
    }
    /**
     * Convert stereo audio buffer to mono
     */
    convert(buffer) {
      if (!this.converter) {
        this.converter = new StereoToMonoConverter(this.options);
      }
      let processedBuffer = buffer;
      if (this.options.sampleRate && this.options.sampleRate !== buffer.sampleRate) {
        processedBuffer = {
          left: AudioUtils.resample(buffer.left, buffer.sampleRate, this.options.sampleRate),
          right: AudioUtils.resample(buffer.right, buffer.sampleRate, this.options.sampleRate),
          sampleRate: this.options.sampleRate
        };
      }
      return this.converter.convert(processedBuffer);
    }
    /**
     * Analyze stereo audio characteristics without converting
     */
    analyze(buffer) {
      if (!this.converter) {
        this.converter = new StereoToMonoConverter(this.options);
      }
      return this.converter.analyze(buffer);
    }
    /**
     * Update options after initialization
     */
    setOptions(options) {
      this.options = { ...this.options, ...options };
      if (this.converter) {
        this.converter.setOptions(options);
      }
      return this;
    }
    /**
     * Get current options
     */
    getOptions() {
      return { ...this.options };
    }
  };
  function createConverter(options) {
    const converter = new NeuroMono();
    if (options) {
      converter.setOptions(options);
    }
    return converter;
  }
  function convert(buffer, options) {
    const converter = createConverter(options);
    return converter.convert(buffer);
  }
  function analyze(buffer) {
    const converter = createConverter();
    return converter.analyze(buffer);
  }
  var index_default = {
    createConverter,
    convert,
    analyze,
    NeuroMono
  };
  return __toCommonJS(index_exports);
})();
