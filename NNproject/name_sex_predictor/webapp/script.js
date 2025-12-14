/*
 * JavaScript implementation of a sex predictor for Japanese names.
 *
 * The model loads a weight vector from model.json on page load.
 * Each name is encoded to a 50-bit vector following the same scheme
 * as the Python training code (5 characters × 10 bits each).
 * The sign of the dot product between (2*v - 1) and weights determines
 * the predicted sex.
 */

(() => {
  // Constants matching train_model.py
  const NAME_LENGTH = 8;          // Characters for first name (increased for better accuracy)
  const BITS_PER_CHAR = 10;       // Bits per character (A-Z encoded as 1-26)
  const TOTAL_BITS = NAME_LENGTH * BITS_PER_CHAR;  // 80 bits total

  // Store the weights globally after loading
  let weights = null;

  /**
   * Convert a single uppercase letter into its 10-bit binary representation.
   * A -> 0000000001, B -> 0000000010, ..., Z -> 0000011010 (26).
   * Unknown characters or empty slots return ten zeros.
   * @param {string} letter A single character
   * @returns {number[]} Array of 10 bits (0 or 1)
   */
  function letterToBits(letter) {
    if (!letter || letter.length !== 1) {
      return new Array(BITS_PER_CHAR).fill(0);
    }
    const ch = letter.toUpperCase();
    if (ch >= 'A' && ch <= 'Z') {
      // Convert A=1, B=2, ..., Z=26
      const value = ch.charCodeAt(0) - 'A'.charCodeAt(0) + 1;
      // Convert value to 10-bit binary string
      const bits = value.toString(2).padStart(BITS_PER_CHAR, '0');
      return bits.split('').map((c) => parseInt(c, 10));
    }
    return new Array(BITS_PER_CHAR).fill(0);
  }

  /**
   * Encode a first name into a 50-bit vector in {0,1}^{50}.
   *
   * The encoding reserves 5 characters (50 bits) for the first name.
   * If the name is shorter than 5 characters, its encoding is right-aligned
   * and padded on the left with groups of ten zeros.
   * @param {string} name A string containing the first name (e.g., "HARUTO")
   * @returns {number[]} Array of 50 bits (0/1) representing the encoded name
   */
  function encodeName(name) {
    if (!name) {
      return new Array(TOTAL_BITS).fill(0);
    }
    
    // Remove spaces and convert to uppercase, keep only letters
    const normalized = name
      .trim()
      .replace(/[^a-zA-Z]/g, '')
      .toUpperCase();
    
    if (normalized.length === 0) {
      return new Array(TOTAL_BITS).fill(0);
    }
    
    // Keep only the rightmost NAME_LENGTH characters if longer
    const trimmed = normalized.slice(-NAME_LENGTH);
    
    // Build 10-bit codes for each character
    const codes = [];
    for (let i = 0; i < trimmed.length; i++) {
      codes.push(letterToBits(trimmed[i]));
    }
    
    // Left-pad with [0]*10 groups to length NAME_LENGTH
    const padLen = NAME_LENGTH - codes.length;
    const padding = Array.from({ length: padLen }, () => new Array(BITS_PER_CHAR).fill(0));
    const full = padding.concat(codes);
    
    // Flatten list of lists to a 50-length array
    return full.flat();
  }

  /**
   * Predict the sex for a given name.
   * @param {string} name Name string (just the first name)
   * @returns {string} 'male' or 'female'
   */
  function predict(name) {
    if (!weights) {
      throw new Error('Model weights have not been loaded.');
    }
    const v = encodeName(name);
    // Convert to ±1 values: 2*v - 1
    const v2 = v.map((bit) => bit * 2 - 1);
    // Compute dot product
    let dot = 0;
    for (let i = 0; i < v2.length && i < weights.length; i++) {
      dot += weights[i] * v2[i];
    }
    return dot >= 0 ? 'male' : 'female';
  }

  /**
   * Load weights from model.json.
   */
  async function loadModel() {
    try {
      const response = await fetch('model.json');
      if (!response.ok) {
        throw new Error('Failed to fetch model.json');
      }
      weights = await response.json();
      console.log('Model loaded: ' + weights.length + ' weights');
    } catch (err) {
      console.error('Could not load model:', err);
      // Fallback to all +1 weights (always predicts male)
      weights = new Array(TOTAL_BITS).fill(1);
    }
  }

  // Setup form handler on DOM ready
  document.addEventListener('DOMContentLoaded', () => {
    loadModel();
    const form = document.getElementById('predict-form');
    const input = document.getElementById('name-input');
    const resultDiv = document.getElementById('result');
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const name = input.value.trim();
      if (!name) {
        resultDiv.textContent = '名前を入力してください';
        return;
      }
      try {
        const prediction = predict(name);
        const emoji = prediction === 'male' ? '👨' : '👩';
        const label = prediction === 'male' ? '男性' : '女性';
        resultDiv.textContent = '予測結果: ' + label + ' ' + emoji;
      } catch (err) {
        console.error(err);
        resultDiv.textContent = '予測中にエラーが発生しました';
      }
    });
  });
})();
