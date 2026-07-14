# 解答

## 1. 陰関数の微分

与えられた式

$$
x^2 + xy + y^2 = 7
$$

を $x$ で微分する。

$$
2x + \left(x\frac{dy}{dx} + y\right) + 2y\frac{dy}{dx} = 0
$$

よって、

$$
\left(x + 2y\right)\frac{dy}{dx} = -(2x+y)
$$

となるから、

$$
\boxed{\frac{dy}{dx} = -\frac{2x+y}{x+2y}}
$$

である。

点 $(1,2)$ では、

$$
\frac{dy}{dx} = -\frac{2\cdot1+2}{1+2\cdot2} = -\frac45
$$

となる。したがって、接線の方程式は

$$
\boxed{y-2=-\frac45(x-1)}
$$

である。

```svg
<svg viewBox="0 0 460 390" xmlns="http://www.w3.org/2000/svg">
	<rect x="45" y="20" width="370" height="330" fill="white" stroke="#d5dde3"/>
	<g transform="translate(230 190) scale(48 -48)">
		<path d="M-3.85 0H3.85M0-3.35V3.35" stroke="#52616d" stroke-width=".025"/>
		<ellipse rx="2.16" ry="3.74" transform="rotate(45)" fill="none" stroke="#174a7a" stroke-width=".06"/>
		<path d="M-.875 3.5L3.85-.28" stroke="#c23b3b" stroke-width=".06"/>
		<circle cx="1" cy="2" r=".11" fill="#c23b3b"/>
	</g>
	<g font-family="sans-serif" font-size="14" fill="#17212b">
		<text x="402" y="184">x</text><text x="237" y="35">y</text>
		<text x="278" y="86" fill="#c23b3b">(1, 2)</text>
		<text x="274" y="316" fill="#c23b3b">接線</text>
		<text x="68" y="334" fill="#174a7a">x² + xy + y² = 7</text>
	</g>
</svg>
```

青が曲線 $x^2+xy+y^2=7$、赤が点 $(1,2)$ における接線である。

---

## 2. ラグランジュ未定乗数法（最大値）

制約条件を $x^2+y^2=1$ とする。ラグランジュ関数を

$$
\Phi(x,y,\lambda):=3x+4y-\lambda(x^2+y^2-1)
$$

と定義する。$\Phi$ の極値候補は、各変数に関する偏微分がすべて $0$ となる点である。したがって、

$$
\Phi_x=3-2\lambda x=0,\qquad
\Phi_y=4-2\lambda y=0,\qquad
\Phi_\lambda=-(x^2+y^2-1)=0
$$

を連立して解く。最初の二式から $x=3/(2\lambda),\ y=2/\lambda$ であり、これを制約条件に代入すると、

$$
\lambda=\pm\frac52
$$

となる。よって極値候補は

$$
(x,y)=\left(\frac35,\frac45\right),\quad
\left(-\frac35,-\frac45\right)
$$

である。それぞれにおける関数値は $5,-5$ なので、最大値を与える点は

$$
\boxed{(x,y)=\left(\frac35,\frac45\right)}
$$

となる。このとき、

$$
f\left(\frac35,\frac45\right)=3\cdot\frac35+4\cdot\frac45=5
$$

である。したがって、

$$
\boxed{\text{最大値は }5}
$$

である。
