const els = {
  body: document.body,
  frame: document.querySelector("#slide-frame"),
  kicker: document.querySelector("#slide-kicker"),
  title: document.querySelector("#slide-title"),
  content: document.querySelector("#slide-body"),
  figure: document.querySelector("#slide-figure"),
  figureMedia: document.querySelector("#figure-media"),
  figureCaption: document.querySelector("#figure-caption"),
  count: document.querySelector("#slide-count"),
  fill: document.querySelector("#progress-fill"),
  prev: document.querySelector("#prev-slide"),
  next: document.querySelector("#next-slide"),
  standardMode: document.querySelector("#standard-mode"),
  fourFrameMode: document.querySelector("#four-frame-mode"),
  imageMode: document.querySelector("#image-mode")
};

let standardSlides = [];
let fourFrameSlides = [];
let imageSlides = [];
let slides = [];
let current = 0;
let mode = "standard";
const currentByMode = {
  standard: 0,
  fourFrame: 0,
  image: 0
};

const standardImage = (filename) => `Images/${filename}`;
const fourFrameImage = (filename) => `../four-frame/${filename}`;
const fourFrameDeckImage = fourFrameImage;

const fourFrameDeckFiles = [
  "01-model-at-a-glance.png",
  "02-majorana-algebra-and-hilbert-space.png",
  "03-random-couplings-and-normalization.png",
  "04-which-syk-model.png",
  "05-why-experimental-realization-is-difficult.png",
  "06-ultracold-atom-route.png",
  "07-majorana-nanowires-and-a-quantum-dot.png",
  "08-programmable-quantum-simulators.png",
  "09-observables-and-scale-map.png",
  "10-averaging-the-disorder.png",
  "11-bilocal-collective-fields.png",
  "12-schwinger-dyson-equations.png",
  "13-infrared-conformal-solution.png",
  "14-reparametrization-symmetry.png",
  "15-four-point-function-and-chaos.png",
  "16-schwarzian-effective-theory.png",
  "17-jackiw-teitelboim-gravity.png",
  "18-the-infrared-syk-jt-correspondence.png",
  "19-finite-n-spectrum.png",
  "20-what-is-established-and-what-remains.png"
];

const standardImageFiles = fourFrameDeckFiles.slice(0, 19);

const slideFigures = {
  "Model at a glance": {
    images: [
      {
        src: fourFrameImage("01-model-at-a-glance.png"),
        alt: "Four-panel overview from Majorana modes and all-to-all interactions to the large-N correlation function"
      }
    ],
    fourFrame: true,
    caption: "Every mode can interact with every other mode. For fixed $q$, the number of random terms grows as $\\binom{N}{q}$."
  },
  "Majorana algebra and Hilbert space": {
    images: [
      {
        src: fourFrameImage("02-majorana-algebra-and-hilbert-space.png"),
        alt: "Four-panel explanation of Majorana algebra, complex fermion pairing, Hilbert-space dimension, and parity sectors"
      }
    ],
    fourFrame: true,
    caption: ""
  },
  "Random couplings and normalization": {
    images: [
      {
        src: fourFrameImage("03-random-couplings-and-normalization.png"),
        alt: "Four-panel scaling sequence for weak random couplings and finite large-N interaction strength"
      }
    ],
    fourFrame: true,
    caption: ""
  },
  "Which SYK model?": {
    images: [
      {
        src: fourFrameImage("04-which-syk-model.png"),
        alt: "Four-panel comparison of quadratic, interacting, charge-preserving, and generalized SYK models"
      }
    ],
    fourFrame: true,
    caption: ""
  },
  "Why experimental realization is difficult": {
    images: [
      {
        src: fourFrameImage("05-why-experimental-realization-is-difficult.png"),
        alt: "Four-panel path from Hamiltonian complexity through control and coherence to measurement"
      }
    ],
    fourFrame: true,
    label: "From Hamiltonian to measurement",
    caption: "A practical emulator must compress an enormous coupling problem into controllable dynamics and measurable signatures."
  },
  "Ultracold-atom route": {
    images: [
      {
        src: fourFrameImage("06-ultracold-atom-route.png"),
        alt: "Four-panel ultracold-atom sequence from encoded states to correlation readout"
      }
    ],
    fourFrame: true,
    label: "Concept rendering",
    caption: "Atomic platforms seek programmable random interactions while retaining site-resolved preparation and readout."
  },
  "Majorana nanowires and a quantum dot": {
    images: [
      {
        src: fourFrameImage("07-majorana-nanowires-and-a-quantum-dot.png"),
        alt: "Four-panel nanowire and quantum-dot sequence for effective Majorana interactions"
      }
    ],
    fourFrame: true,
    label: "Concept rendering",
    caption: "The target low-energy interaction is $H_{\\mathrm{eff}}=\\sum J_{ijkl}\\gamma_i\\gamma_j\\gamma_k\\gamma_l$."
  },
  "Programmable quantum simulators": {
    images: [
      {
        src: fourFrameImage("08-programmable-quantum-simulators.png"),
        alt: "Four-panel digital and NMR simulation sequence with echo protocol and finite-size readout"
      }
    ],
    fourFrame: true,
    label: "Two control architectures",
    caption: "Digital gates and NMR pulse sequences emulate small-$N$ dynamics and measure observables such as the OTOC."
  },
  "Observables and scale map": {
    images: [
      {
        src: fourFrameImage("09-observables-and-scale-map.png"),
        alt: "Four-panel scale map for thermodynamics, correlations, scrambling, and spectral statistics"
      }
    ],
    fourFrame: true,
    caption: ""
  },
  "Averaging the disorder": {
    images: [
      {
        src: fourFrameImage("10-averaging-the-disorder.png"),
        alt: "Four-panel flow from disorder realizations through replication and averaging to collective fields"
      }
    ],
    fourFrame: true,
    label: "Disorder averaging flow",
    caption: "Disorder averaging removes the individual couplings and exposes the collective variables used at large $N$."
  },
  "Bilocal collective fields": {
    images: [
      {
        src: fourFrameImage("11-bilocal-collective-fields.png"),
        alt: "Four-panel explanation of the bilocal fields, fermion integration, effective action, and large-N saddle"
      }
    ],
    fourFrame: true,
    caption: "The large-$N$ description lives on the two-time plane $(\\tau_1,\\tau_2)$ rather than on a spatial lattice."
  },
  "Schwinger-Dyson equations": {
    images: [
      {
        src: fourFrameImage("12-schwinger-dyson-equations.png"),
        alt: "Four-panel self-consistency loop from propagator to self-energy and back"
      }
    ],
    fourFrame: true,
    label: "Self-consistency loop",
    caption: "The large-$N$ solution is a closed iteration between the propagator $G$ and self-energy $\\Sigma$."
  },
  "Infrared conformal solution": {
    images: [
      {
        src: fourFrameImage("13-infrared-conformal-solution.png"),
        alt: "Four-panel flow from ultraviolet oscillations to a thermal conformal circle"
      }
    ],
    fourFrame: true,
    label: "Flow to the conformal regime",
    caption: "The conformal answer emerges after the microscopic kinetic term loses control of the long-time saddle."
  },
  "Reparametrization symmetry": {
    images: [
      {
        src: fourFrameImage("14-reparametrization-symmetry.png"),
        alt: "Four-panel clock deformation sequence ending in a softly lifted reparametrization mode"
      }
    ],
    fourFrame: true,
    caption: ""
  },
  "Four-point function and chaos": {
    images: [
      {
        src: fourFrameImage("15-four-point-function-and-chaos.png"),
        alt: "Four-panel progression from ladder diagrams to operator growth and scrambling"
      }
    ],
    fourFrame: true,
    caption: "A simple operator spreads over $O(N)$ modes by the scrambling time $t_*\\sim(\\beta/2\\pi)\\log N$."
  },
  "Schwarzian effective theory": {
    images: [
      {
        src: fourFrameImage("16-schwarzian-effective-theory.png"),
        alt: "Four-panel Schwarzian sequence from the soft clock to low-temperature observables"
      }
    ],
    fourFrame: true,
    caption: "The Schwarzian action governs the soft boundary clock $f(\\tau)$ in low-energy SYK."
  },
  "Jackiw-Teitelboim gravity": {
    images: [
      {
        src: fourFrameImage("17-jackiw-teitelboim-gravity.png"),
        alt: "Four-panel reduction from the JT bulk to Schwarzian boundary dynamics"
      }
    ],
    fourFrame: true,
    caption: "JT gravity reduces its nearly-$AdS_2$ boundary dynamics to a Schwarzian clock."
  },
  "The infrared SYK/JT correspondence": {
    images: [
      {
        src: fourFrameImage("18-the-infrared-syk-jt-correspondence.png"),
        alt: "Four-panel comparison of SYK and JT gravity through their shared low-energy sector"
      }
    ],
    fourFrame: true,
    caption: "The shared Schwarzian sector matches low-energy observables without equating the microscopic theories."
  },
  "Finite-$N$ spectrum": {
    images: [
      {
        src: fourFrameImage("19-finite-n-spectrum.png"),
        alt: "Four-panel spectral sequence from symmetry sectors to the dip-ramp-plateau signature"
      }
    ],
    fourFrame: true,
    caption: "Global density, microscopic level repulsion, and the spectral ramp probe different energy and time scales."
  }
};

const fallbackText = String.raw`
## 20-slide blueprint: the SYK model in detail

Framing line: SYK is a random, all-to-all quantum many-body model whose large-$N$ dynamics closes on a two-point function.

### 1. Model at a glance
- Let $N$ and $q$ be even. The model contains $N$ Majorana fermions with $\psi_i^*=\psi_i$ and $\{\psi_i,\psi_j\}=\delta_{ij}$.
- Its Hamiltonian is
$$
H=i^{q/2}\sum_{1\leq i_1<\cdots<i_q\leq N}
J_{i_1\cdots i_q}\psi_{i_1}\cdots\psi_{i_q}.
$$
- Every $q$-tuple interacts, so there is no spatial geometry. Randomness replaces locality, while the $N\to\infty$ limit supplies a controlled expansion.
- The standard interacting model is $q=4$; $q=2$ is a useful free-fermion comparison.

### 2. Majorana algebra and Hilbert space
- The convention $\{\psi_i,\psi_j\}=\delta_{ij}$ gives $\psi_i^2=1/2$.
- Pair the Majoranas into $N/2$ complex fermions:
$$
c_a=\frac{\psi_{2a-1}+i\psi_{2a}}{\sqrt 2},
\qquad
\{c_a,c_b^\dagger\}=\delta_{ab}.
$$
- The Hilbert space therefore has dimension $2^{N/2}$.
- For even $q$, the phase $i^{q/2}$ makes each interaction term Hermitian. Fermion parity $(-1)^F$ is conserved, so the Hamiltonian splits into parity blocks.

### 3. Random couplings and normalization
- The couplings are independent centered Gaussians, up to antisymmetry:
$$
\mathbb E J_I=0,
\qquad
\mathbb E J_I^2=\frac{(q-1)!\,J^2}{N^{q-1}}.
$$
- A single coupling is of order $N^{-(q-1)/2}$, but each fermion participates in order $N^{q-1}$ interactions.
- This balance gives a nontrivial extensive large-$N$ limit and keeps the self-energy of order one.
- The parameter $J$ fixes the microscopic energy scale. Strong coupling means $\beta J\gg 1$.

### 4. Which SYK model?
- Majorana $q=2$: a random quadratic Hamiltonian, exactly diagonalizable and not maximally chaotic.
- Majorana $q\geq4$: interacting models with infrared fermion dimension $\Delta=1/q$.
- Complex SYK: complex fermions and random charge-preserving interactions; it has a $U(1)$ symmetry and supports finite density.
- Coupled, supersymmetric, and tensor variants isolate different features such as wormhole-like phases, protected structure, or disorder-free melonic limits.
- The remaining slides concern the standard Majorana model unless stated otherwise.

### 5. Why experimental realization is difficult
- A literal device needs $N$ coherent modes and a separately disordered coupling for many $q$-tuples:
$$
M_q=\binom{N}{q},
\qquad
J_I\sim N^{-(q-1)/2}.
$$
- For $q=4$, the interaction count grows approximately as $N^4/24$, far faster than the number of control channels in ordinary hardware.
- The useful dynamics must occur before decoherence, yet strong coupling requires $\beta J\gg1$ and scrambling requires a time of order $\log N$.
- Experiments therefore emulate selected signatures rather than reproducing an arbitrarily large, ideal Hamiltonian term by term.

### 6. Ultracold-atom route
- Atomic proposals encode complex fermions in long-lived internal states or orbitals and use optical fields or cavity-mediated processes to generate random couplings.
- The desired charge-preserving interaction has the form
$$
H_{\mathrm{int}}=\sum_{i,j,k,l}
J_{ij;kl}\,c_i^\dagger c_j^\dagger c_k c_l.
$$
- Preparation, tunable interactions, and correlation readout are natural strengths; dense independent four-body couplings are the central engineering challenge.
- These platforms are best viewed as proposed analog emulators of SYK-like dynamics, not yet as large ideal realizations.

### 7. Majorana nanowires and a quantum dot
- A solid-state proposal couples many Majorana zero modes $\gamma_i$ to a disordered interacting quantum dot.
- Projecting to low energy can generate
$$
H_{\mathrm{eff}}=\sum_{i<j<k<l}
J_{ijkl}\gamma_i\gamma_j\gamma_k\gamma_l.
$$
- The dot supplies irregular overlaps, while superconducting nanowires supply the Majorana degrees of freedom.
- The demanding ingredients are many well-isolated zero modes, controlled charging interactions, low temperature, and parity-sensitive readout.

### 8. Programmable quantum simulators
- Digital processors decompose $e^{-iHt}$ into implementable gates; analog NMR uses shaped radio-frequency pulses to encode a small effective Hamiltonian.
- Echo and time-reversal protocols can measure the out-of-time-order correlator
$$
F(t)=\langle W^\dagger(t)V^\dagger W(t)V\rangle.
$$
- The advantage is flexible control over disorder realizations and observables.
- The limitation is scale: these experiments test finite-$N$ spectra and scrambling signatures rather than the full thermodynamic conformal regime.

### 9. Observables and scale map
- Thermal partition function and free energy:
$$
Z(\beta)=\operatorname{Tr}e^{-\beta H},
\qquad
F(\beta)=-\beta^{-1}\log Z(\beta).
$$
- Disorder-averaged two-point function:
$$
G(\tau)=\frac1N\sum_{i=1}^N
\mathbb E\langle T_\tau\psi_i(\tau)\psi_i(0)\rangle.
$$
- Four-point functions test operator growth; the spectrum tests level repulsion and late-time universality.
- The main regimes are microscopic times $\tau J\lesssim1$, a conformal window $1\ll\tau J\ll N$, and finite-$N$ late-time physics.

### 10. Averaging the disorder
- The Euclidean action is quadratic in each Gaussian variable $J_I$. Averaging $Z^n$ therefore produces a deterministic interaction coupling two times and, in the replica method, two replicas.
- The quenched free energy is formally recovered from
$$
\mathbb E\log Z
=\lim_{n\to0}\frac{\mathbb E Z^n-1}{n}.
$$
- At the replica-diagonal large-$N$ saddle, the fundamental variables are not individual fermions but collective two-time fields.
- Annealed and quenched averages need not agree at every temperature; the replica assumption is part of the analysis, not an identity.

### 11. Bilocal collective fields
- Introduce
$$
G(\tau_1,\tau_2)=\frac1N\sum_i
\psi_i(\tau_1)\psi_i(\tau_2)
$$
and a Lagrange multiplier $\Sigma(\tau_1,\tau_2)$ enforcing this definition.
- After integrating out the $N$ fermions, the effective action per fermion is
$$
\frac{I[G,\Sigma]}N
=-\log\operatorname{Pf}(\partial_\tau-\Sigma)
+\frac12\int d\tau_1d\tau_2
\left[\Sigma G-\frac{J^2}{q}G^q\right].
$$
- The overall factor $N$ makes the saddle-point approximation exact at leading order. Fluctuations are organized in powers of $1/N$.

### 12. Schwinger-Dyson equations
- Varying the bilocal action gives
$$
G(i\omega_n)=\frac{1}{-i\omega_n-\Sigma(i\omega_n)},
\qquad
\Sigma(\tau)=J^2G(\tau)^{q-1}.
$$
- Equivalently, $(\partial_\tau-\Sigma)*G=\delta$, where $*$ denotes convolution in Euclidean time.
- These equations resum the melonic Feynman diagrams that dominate at large $N$.
- Numerically, one alternates between time and frequency space until $G$ and $\Sigma$ are self-consistent.

### 13. Infrared conformal solution
- For $\beta J\gg1$ and time separations $J^{-1}\ll|\tau|\ll\beta$, the derivative term is subleading.
- The saddle then has
$$
G_c(\tau)=b\,\frac{\operatorname{sgn}\tau}{|J\tau|^{2\Delta}},
\qquad
\Delta=\frac1q,
$$
$$
b^q=\frac{1-2\Delta}{2\pi}\tan(\pi\Delta).
$$
- At temperature $1/\beta$, the line is mapped to the thermal circle, replacing $|\tau|^{-2\Delta}$ by $[\pi/(\beta\sin(\pi\tau/\beta))]^{2\Delta}$.
- The power law describes a non-Fermi liquid: there is no quasiparticle pole.

### 14. Reparametrization symmetry
- After dropping $\partial_\tau$, the infrared equations are covariant under $\tau\mapsto f(\tau)$:
$$
G(\tau_1,\tau_2)\mapsto
[f'(\tau_1)f'(\tau_2)]^\Delta
G(f(\tau_1),f(\tau_2)).
$$
- A conformal saddle selects one clock and breaks this symmetry to $SL(2,\mathbb R)$.
- The microscopic derivative term weakly breaks the larger symmetry and turns the reparametrization mode into a soft mode.
- This soft mode controls the leading low-temperature correction to thermodynamics and correlators.

### 15. Four-point function and chaos
- The connected four-point function is order $1/N$. Its leading diagrams form a ladder series with a kernel built from the conformal two-point function.
- In the out-of-time-order channel, the growing contribution behaves as
$$
\frac{1}{N}e^{\lambda_L t},
\qquad
\lambda_L=\frac{2\pi}{\beta}
$$
at strong coupling, up to finite-coupling corrections.
- The scrambling time is therefore $t_*\sim(\beta/2\pi)\log N$.
- Saturation of the chaos bound is a property of the interacting low-energy regime, not of the $q=2$ model.

### 16. Schwarzian effective theory
- The soft clock $f(\tau)$ is governed at low energy by the Schwarzian action
$$
I_{\mathrm{Sch}}
=-\frac{N\alpha_S}{J}\int d\tau\,
\left\{\tan\frac{\pi f(\tau)}{\beta},\tau\right\}.
$$
- Its configuration space is $\operatorname{Diff}(S^1)/SL(2,\mathbb R)$.
- The coefficient is of order $N/J$, so fluctuations of the soft clock are controlled by the large-$N$ expansion.
- This one-dimensional theory gives the leading low-temperature correction to thermodynamics and enhances the chaos channel of the four-point function.

### 17. Jackiw-Teitelboim gravity
- Jackiw-Teitelboim gravity is a two-dimensional dilaton theory whose bulk equation fixes the metric to have constant negative curvature, $R=-2$ in AdS units.
- The dilaton controls the strength of gravitational effects while the bulk metric has no propagating graviton.
- With nearly-$AdS_2$ boundary conditions, the physical low-energy degree of freedom is the shape of the regulated boundary curve.
- After holographic renormalization, the action for that boundary trajectory is a Schwarzian action for its clock $f(\tau)$.

### 18. The infrared SYK/JT correspondence
- Low-energy SYK and nearly-$AdS_2$ JT gravity therefore share the same Schwarzian boundary theory.
- Matching the Schwarzian coefficient relates the SYK scale $N/J$ to the renormalized boundary-dilaton coupling in gravity.
- This common sector explains matching low-temperature thermodynamics, enhanced four-point functions, and maximal Lyapunov growth.
- The relation is an infrared, large-$N$ correspondence. It does not identify a finite SYK Hamiltonian with one classical spacetime.

### 19. Finite-$N$ spectrum
- Fermion parity and antiunitary symmetries determine the random-matrix class; the pattern depends periodically on $N$ modulo $8$.
- The global density is well approximated by a $q$-Hermite form for standard finite-$q$ SYK, while local unfolded spacings show random-matrix level repulsion.
- The spectral form factor displays the characteristic dip, ramp, and plateau after suitable averaging.
- Very late times probe discreteness of the $2^{N/2}$-dimensional spectrum and are not captured by the leading conformal saddle.

### 20. What is established, and what remains
- Rigorous results include limiting global spectral laws in several $q_N$ regimes, central limit theorems for linear statistics, concentration statements, and high-temperature free-energy limits.
- The large-$N$ bilocal saddle, conformal solution, and ladder analysis are controlled physics calculations with extensive numerical support.
- Random-matrix universality for local statistics and the precise single-sample meaning of the gravity relation remain subtler than the disorder-averaged saddle suggests.
- The core mechanism is: random $q$-body interactions lead to melonic closure, melonic closure leads to conformal dynamics, and its soft mode leads to maximal chaos and the Schwarzian theory.
`;

function normalizeMath(text) {
  return text
    .replace(/\\\[/g, () => "$$")
    .replace(/\\\]/g, () => "$$")
    .replace(/\\\(/g, "$")
    .replace(/\\\)/g, "$");
}

function stripReferences(text) {
  const lines = normalizeMath(text).replace(/\r\n/g, "\n").split("\n");
  const output = [];
  let skipping = false;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    const heading = /^(#{1,6}\s*)?(references|bibliography|works cited|learn more)\b/i.test(line);
    const citationOnly = /^(\[\d+\]|\d+\.|\-\s*)\s*(doi:|arxiv:|https?:\/\/)/i.test(line);

    if (heading) {
      skipping = true;
      continue;
    }

    if (/^-{3,}$/.test(line)) {
      continue;
    }

    if (skipping && /^#{1,6}\s+\S/.test(line)) {
      skipping = false;
    }

    if (!skipping && !citationOnly) {
      output.push(removeInlineReferences(rawLine));
    }
  }

  return output.join("\n").trim();
}

function removeInlineReferences(line) {
  return line
    .replace(/\s*Ref:[^.\n]*(\.\s*)?/gi, " ")
    .replace(/\s*\(\[[^\]]+\]\([^)]+\)\)/g, "")
    .replace(/\s*\[[^\]]+\]\([^)]+\)/g, "")
    .replace(/\s{2,}/g, " ")
    .trimEnd();
}

function parseSlides(text) {
  const cleaned = stripReferences(text);
  const lines = cleaned.split("\n");
  const sections = [];
  let active = null;
  let framing = "";

  for (const line of lines) {
    const trimmed = line.trim();
    const framingMatch = /^(?:\*\*)?Framing line:(?:\*\*)?\s*(.+)$/i.exec(trimmed);
    const conceptMatch = /^#{3,4}\s+\d+\.\s*(.+)$/.exec(trimmed);

    if (framingMatch) {
      framing = framingMatch[1].replace(/\*/g, "").trim();
      continue;
    }

    if (conceptMatch) {
      if (active) sections.push(active);
      active = {
        title: conceptMatch[1].replace(/\s+\{#.+\}$/, "").trim(),
        body: []
      };
      continue;
    }

    if (!active) {
      continue;
    }

    active.body.push(line);
  }

  if (active) sections.push(active);

  return sections
    .map((section, index) => ({
      title: section.title || `Concept ${index + 1}`,
      body: markdownToHtml(withFraming(section.body.join("\n").trim(), index, framing))
    }))
    .filter((section) => section.title || section.body);
}

function withFraming(body, index, framing) {
  if (!framing || index !== 0) return body;
  return `${framing}\n\n${body}`;
}

function markdownToHtml(markdown) {
  const lines = markdown.split("\n");
  const output = [];
  let paragraph = [];
  let listType = "";
  let listItems = [];
  let math = [];
  let inMath = false;

  const flushParagraph = () => {
    if (!paragraph.length) return;
    output.push(`<p>${inlineFormat(paragraph.join(" "))}</p>`);
    paragraph = [];
  };

  const flushList = () => {
    if (!listType) return;
    output.push(`<${listType}>${listItems.join("")}</${listType}>`);
    listType = "";
    listItems = [];
  };

  const flushMath = () => {
    if (!math.length) return;
    output.push(`<div class="math-block">${escapeHtml(math.join("\n"))}</div>`);
    math = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (inMath) {
      math.push(rawLine);
      if (line.endsWith("$$")) {
        inMath = false;
        flushMath();
      }
      continue;
    }

    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }

    if (line.startsWith("$$")) {
      flushParagraph();
      flushList();
      math.push(rawLine);
      if (line.length > 2 && line.endsWith("$$")) {
        flushMath();
      } else {
        inMath = true;
      }
      continue;
    }

    if (/^#{3,6}\s+/.test(line)) {
      flushParagraph();
      flushList();
      output.push(`<h2>${inlineFormat(line.replace(/^#{3,6}\s+/, ""))}</h2>`);
      continue;
    }

    const bullet = /^[-*]\s+(.+)$/.exec(line);
    if (bullet) {
      flushParagraph();
      if (listType && listType !== "ul") flushList();
      listType = "ul";
      listItems.push(`<li>${inlineFormat(bullet[1])}</li>`);
      continue;
    }

    const numbered = /^\d+\.\s+(.+)$/.exec(line);
    if (numbered) {
      flushParagraph();
      if (listType && listType !== "ol") flushList();
      listType = "ol";
      listItems.push(`<li>${inlineFormat(numbered[1])}</li>`);
      continue;
    }

    flushList();
    paragraph.push(rawLine.trim());
  }

  flushParagraph();
  flushList();
  flushMath();

  if (!output.length) {
    return "<p>This concept has no body text in the blueprint.</p>";
  }

  return output.join("");
}

function inlineFormat(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function buildFourFrameSlides(sourceSlides) {
  return fourFrameDeckFiles.map((filename, index) => {
    const title = sourceSlides[index]?.title || `四コマ ${index + 1}`;

    return {
      title,
      body: "",
      figure: {
        images: [
          {
            src: fourFrameDeckImage(filename),
            alt: `${title} 四コマ`
          }
        ],
        fourFrame: true,
        caption: ""
      }
    };
  });
}

function buildImageSlides(sourceSlides) {
  return standardImageFiles.map((filename, index) => {
    const title = sourceSlides[index]?.title || `Image ${index + 1}`;

    return {
      title,
      body: "",
      figure: {
        images: [
          {
            src: standardImage(filename),
            alt: `${title} image`
          }
        ],
        imageDeck: true,
        caption: ""
      }
    };
  });
}

function activeSlides(nextMode = mode) {
  if (nextMode === "fourFrame") return fourFrameSlides;
  if (nextMode === "image") return imageSlides;
  return standardSlides;
}

function initialMode() {
  const requested = new URLSearchParams(window.location.search).get("mode");
  if (["image", "images"].includes(requested)) return "image";
  return ["4koma", "four-frame", "fourFrame", "yonnkoma", "よんこま", "四コマ"].includes(requested)
    ? "fourFrame"
    : "standard";
}

function syncUrl() {
  const params = new URLSearchParams(window.location.search);
  params.set("slide", String(current + 1));

  if (mode === "fourFrame") {
    params.set("mode", "4koma");
  } else if (mode === "image") {
    params.set("mode", "images");
  } else {
    params.delete("mode");
  }

  const query = params.toString();
  const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`;
  window.history.replaceState(null, "", nextUrl);
}

function updateModeControls() {
  const isFourFrame = mode === "fourFrame";
  const isImage = mode === "image";
  els.standardMode.classList.toggle("is-active", mode === "standard");
  els.fourFrameMode.classList.toggle("is-active", isFourFrame);
  els.imageMode.classList.toggle("is-active", isImage);
  els.standardMode.setAttribute("aria-pressed", String(mode === "standard"));
  els.fourFrameMode.setAttribute("aria-pressed", String(isFourFrame));
  els.imageMode.setAttribute("aria-pressed", String(isImage));
}

function render() {
  const slide = slides[current];

  if (!slide) return;

  if (window.MathJax?.typesetClear) {
    window.MathJax.typesetClear([els.title, els.content, els.figure]);
  }

  els.kicker.textContent = mode === "fourFrame" ? "四コマ" : "SYK notes";
  els.title.textContent = slide.title;
  els.content.innerHTML = slide.body;
  els.frame.classList.toggle("is-four-frame-deck", mode === "fourFrame");
  els.frame.classList.toggle("is-image-deck", mode === "image");
  renderFigure(slide);
  updateTitleLayout();
  const renderedTextLength = els.content.textContent.length;
  const renderedMathCount = els.content.querySelectorAll(".math-block").length;

  els.content.classList.toggle(
    "dense",
    mode === "standard" && (renderedTextLength > 650 || renderedMathCount > 1)
  );
  els.content.classList.toggle(
    "extra-dense",
    mode === "standard" && (renderedTextLength > 900 || renderedMathCount > 2)
  );
  els.content.classList.remove("is-entering");
  void els.content.offsetWidth;
  els.content.classList.add("is-entering");
  els.count.textContent = `${current + 1} / ${slides.length}`;
  els.fill.style.width = `${((current + 1) / slides.length) * 100}%`;

  els.prev.disabled = current === 0;
  els.next.disabled = current === slides.length - 1;
  currentByMode[mode] = current;
  updateModeControls();
  syncUrl();

  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise([els.title, els.content, els.figure]).then(
      updateTitleLayout
    );
  }
}

function updateTitleLayout() {
  els.title.classList.remove("multiline-title");

  const styles = window.getComputedStyle(els.title);
  const lineHeight = Number.parseFloat(styles.lineHeight);
  const lineCount = lineHeight > 0
    ? Math.round(els.title.getBoundingClientRect().height / lineHeight)
    : 1;

  els.title.classList.toggle("multiline-title", lineCount > 1);
}

function renderFigure(slide) {
  const figure = slide.figure;
  els.figureMedia.innerHTML = "";
  els.figureCaption.innerHTML = "";
  els.figure.hidden = !figure;
  els.frame.classList.toggle("has-figure", Boolean(figure));
  els.frame.classList.toggle("has-four-frame", Boolean(figure?.fourFrame));
  els.figure.classList.toggle("figure-gallery", (figure?.images?.length ?? 0) > 1);
  els.figure.classList.toggle("figure-sequence", Boolean(figure?.panels));
  els.figure.classList.toggle("figure-four-frame", Boolean(figure?.fourFrame));
  els.figure.classList.toggle("figure-image-deck", Boolean(figure?.imageDeck));

  if (!figure) return;

  if (figure.panels) {
    els.figureMedia.setAttribute("role", "list");
    els.figureMedia.setAttribute("aria-label", figure.label || "Four-frame explanation");

    figure.panels.forEach((panel, index) => {
      const item = document.createElement("article");
      item.className = "figure-panel";
      item.setAttribute("role", "listitem");
      item.style.setProperty("--panel-index", index);
      item.innerHTML = `
        <div class="panel-visual visual-${panel.visual}" aria-hidden="true">
          ${renderPanelVisual(panel.visual)}
        </div>
        <p class="panel-stage">${index + 1}. ${escapeHtml(panel.stage)}</p>
        <h2>${panel.title}</h2>
        <p>${panel.text}</p>
      `;
      els.figureMedia.appendChild(item);
    });
  } else {
    els.figureMedia.removeAttribute("role");
    els.figureMedia.removeAttribute("aria-label");

    for (const image of figure.images) {
      const img = document.createElement("img");
      img.src = image.src;
      img.alt = image.alt;
      img.loading = "eager";
      img.decoding = "sync";
      els.figureMedia.appendChild(img);
    }
  }

  const label = figure.label
    ? `<span class="figure-label">${escapeHtml(figure.label)}</span>`
    : "";
  els.figureCaption.innerHTML = `${label}${figure.caption || ""}`;
}

function renderPanelVisual(type) {
  const visuals = {
    network: `
      <span class="network-ring"></span>
      ${Array.from({ length: 7 }, (_, index) => `<i style="--node:${index}"></i>`).join("")}
    `,
    sliders: `${[26, 68, 43, 81].map((value) => `<i style="--value:${value}%"></i>`).join("")}`,
    clock: "<i></i><span></span>",
    signal: `${[34, 72, 49, 88, 61, 29, 76].map((value) => `<i style="--signal:${value}%"></i>`).join("")}`,
    samples: `${[30, 78, 48, 91, 57, 22, 69, 41].map((value) => `<i style="--sample:${value}%"></i>`).join("")}`,
    layers: "<i></i><i></i><i></i>",
    gaussian: "<i></i><span>$\\mathbb E_J$</span>",
    bilocal: `${Array.from({ length: 16 }, (_, index) => `<i style="--cell:${index}"></i>`).join("")}`,
    curve: '<svg viewBox="0 0 120 48" focusable="false"><path d="M4 42 C18 10, 32 8, 43 25 S65 43, 78 20 S101 9, 116 15"/></svg>',
    sigma: "<span>$G$</span><i></i><span>$\\Sigma$</span>",
    dyson: "<span>$\\Sigma$</span><i></i><span>$G$</span>",
    loop: "<i></i><i></i><span>$G\\leftrightarrow\\Sigma$</span>",
    uv: "<span>$\\partial_\\tau-\\Sigma$</span>",
    cutoff: "<span class=\"faded\">$\\partial_\\tau$</span><i></i><span>$\\Sigma$</span>",
    power: '<svg viewBox="0 0 120 48" focusable="false"><path d="M5 7 C18 10, 28 16, 40 23 S72 35, 116 42"/></svg>',
    thermal: "<i></i><span>$\\beta$</span>"
  };

  return visuals[type] || "";
}

function move(delta) {
  current = Math.min(Math.max(current + delta, 0), slides.length - 1);
  render();
}

function switchMode(nextMode) {
  if (nextMode === mode) return;

  const parallelIndex = current;
  currentByMode[mode] = current;
  mode = nextMode;
  slides = activeSlides();
  current = Math.min(parallelIndex, slides.length - 1);
  currentByMode[mode] = current;
  render();
}

function initialSlideIndex(total) {
  const requested = Number.parseInt(
    new URLSearchParams(window.location.search).get("slide"),
    10
  );

  if (!Number.isInteger(requested)) return 0;
  return Math.min(Math.max(requested - 1, 0), total - 1);
}

async function loadBlueprint() {
  let text = fallbackText;

  try {
    const response = await fetch("../blueprint.txt", { cache: "no-store" });
    if (!response.ok) throw new Error("Blueprint request failed");
    text = await response.text();
  } catch {
    text = fallbackText;
  }

  standardSlides = parseSlides(text);
  fourFrameSlides = buildFourFrameSlides(standardSlides);
  imageSlides = buildImageSlides(standardSlides);
  mode = initialMode();
  slides = activeSlides();
  current = initialSlideIndex(slides.length);
  currentByMode[mode] = current;
  render();
}

els.prev.addEventListener("click", () => move(-1));
els.next.addEventListener("click", () => move(1));
els.standardMode.addEventListener("click", () => switchMode("standard"));
els.fourFrameMode.addEventListener("click", () => switchMode("fourFrame"));
els.imageMode.addEventListener("click", () => switchMode("image"));

window.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight" || event.key === "PageDown") move(1);
  if (event.key === "ArrowLeft" || event.key === "PageUp") move(-1);
});

window.addEventListener("resize", updateTitleLayout);

loadBlueprint();
