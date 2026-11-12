# Face Studio Camera & Digital Image Forgery Detection
## Comprehensive 30-Minute Defense Pitch & Panel Presentation Guide

**Project Topic:** *Face Studio Camera & Forgery Detection: Forgery Detection and Localization using ELA, SIFT and CNN*  
**Academic Unit:** Image Processing & Computer Vision (IPCV) · Section A · Team No. 1  
**Team Members:**
1. **Aashish Gupta Rauniyar (20233501):** Studio Editing Pipeline, Enhancement Filters, Presets, Quality Metrics (PSNR, SSIM)
2. **Bishal Ranjan Sah (20233515):** Face Geometry, BlazeFace Tiled Detection, 478-pt Mesh, Head Pose, Blinks, Haar Benchmark
3. **Anubhav Kaushik (20233068):** Person Segmentation, Background Compositing, AR Overlays, Beauty & Privacy Filters
4. **Amit Kumar (20233046):** Digital Image Forensics, Error Level Analysis (ELA), SIFT Copy-Move, CNN Classification, PRNU Noise & JPEG Ghost
5. **Ashish Gupta (20233511):** IPCV Concepts Lab (8 Syllabus Topics), Live Camera Processing, UI/UX Engineering, Testing & Documentation

---

# PART 1: 30-MINUTE PANEL DEFENSE TIMELINE & SCRIPT

```
Total Allocation: 30 Minutes
├── 00:00 - 03:00 (3 min) : Executive Opening, Real-World Problem & Unified Scope
├── 03:00 - 07:00 (4 min) : System Architecture, Data Flow & Engineering Philosophy
├── 07:00 - 15:00 (8 min) : Deep Mathematical & Algorithmic Foundations
├── 15:00 - 24:00 (9 min) : Live Application Demonstration (Screen-by-Screen Output Proof)
├── 24:00 - 27:00 (3 min) : Robustness Experiments, Quantitative Benchmarks & Limitations
└── 27:00 - 30:00 (3 min) : Conclusion, Future Scope & Q&A Defense Handoff
```

---

### Segment 1: Executive Opening, Problem & Scope (00:00 – 03:00)

**Spoken Script:**
> *"Respected panel members and faculty, good morning/afternoon. Today, Team 1 of Section A presents our capstone Image Processing and Computer Vision project: **Face Studio Camera & Forgery Detection**.*
>
> *Every day, billions of mobile camera users interact with features like portrait mode, beauty smoothing, augmented reality face filters, and privacy anonymization. However, modern commercial software treats these features as proprietary, black-box filters with zero mathematical transparency. Simultaneously, visual media is facing an unprecedented credibility crisis: spliced composites, copy-move duplications, and generative tampering are now imperceptible to the naked human eye.*
>
> *Our project addresses both sides of this coin through a unified, dual-core system:  
> **Core 1: Creative & Analytical Camera Studio** — Rebuilding 15 commercial-grade camera effects entirely from fundamental IPCV mathematical equations (bilateral filtering, alpha compositing, CLAHE, morphological dilation, and Gram-Schmidt orthonormalization) combined with lightweight deep-learning vision models.*  
> **Core 2: Digital Image Forensics & Forgery Detection** — A rigorous multi-technique forensic suite that uncovers invisible tampering using Error Level Analysis (ELA), Scale-Invariant Feature Transform (SIFT) keypoint clustering, camera sensor noise patterns (PRNU), and deep convolutional neural networks (CNN).*
>
> *Critically, our entire pipeline is engineered to run **100% locally on a standard laptop CPU** without requiring expensive GPUs, paid cloud APIs, or sending sensitive user photos across the network."*

---

### Segment 2: System Architecture & Engineering Philosophy (03:00 – 07:00)

**Spoken Script:**
> *"To achieve real-time interactivity on CPU, our primary design principle is: **'Compute the expensive things once, reuse across cheap pixel operations.'**
>
> *When an image enters our pipeline:*
> 1. *It undergoes EXIF-orientation normalization (via Pillow) and is capped at 1400 px using area interpolation (`cv2.INTER_AREA`), reducing raw pixel counts by up to 8× without aliasing.*
> 2. *Three lightweight neural networks run exactly once:*
>    - *Google MediaPipe BlazeFace short-range detector (yielding bounding boxes and 6 keypoints).*
>    - *MediaPipe FaceLandmarker (predicting 478 3-D landmarks per face).*
>    - *MediaPipe Selfie Segmenter (outputting a per-pixel person confidence mask).*
> 3. *Their mathematical outputs are cached in memory. Because subsequent visual stages (enhancement, beauty, style, AR, privacy) perform strictly photometric and morphological transformations without geometric warping, these landmarks and masks remain permanently valid.*
> 4. *When a user adjusts a parameter slider, only the lightweight pixel operations re-execute, executing in under 25 milliseconds instead of hundreds of milliseconds.*
> 5. *We enforce strict **uint8 overflow protection**: standard 8-bit image arrays wrap around modulo 256 (for example, $250 + 10 = 4$). We upcast arrays to `int16` or `float32`, apply vector operations across NumPy arrays, clamp with `np.clip(..., 0, 255)`, and downcast back to `uint8`.*
> 6. *Finally, the system integrates a live forensic bridge: any active image in the editing session can be verified for tampering with a single click, transmitting the in-memory array to our digital forensics engine."*

---

### Segment 3: Deep Mathematical & Algorithmic Foundations (07:00 – 15:00)

*(This is where the panel evaluates your theoretical rigor. Speak with absolute mathematical precision.)*

#### 1. Peak Signal-to-Noise Ratio (PSNR) & Structural Similarity Index (SSIM)
> *"To quantitatively evaluate how much an operation changes an image, we deploy two complementary metrics:*
> - *First, **Mean Squared Error (MSE)** and **PSNR**:*
>   $$\text{MSE} = \frac{1}{M \cdot N} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} [I(x, y) - K(x, y)]^2$$
>   $$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{255^2}{\text{MSE}} \right) \text{ dB}$$
>   *PSNR treats every pixel difference equally. If identical, $\text{MSE}=0$ and $\text{PSNR}=\infty$. Above 40 dB, edits are imperceptible; below 20 dB, large changes have occurred.*
> - *However, human vision is sensitive to structural relationships, not raw pixel noise. Thus, we compute **SSIM** (Wang et al., 2004) over an $11 \times 11$ circular Gaussian sliding window ($\sigma = 1.5$):*
>   $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
>   *where $\mu$ is local luminance, $\sigma^2$ is local contrast variance, and $\sigma_{xy}$ is cross-correlation covariance. A critical viva insight: **SSIM on a pure greyscale conversion yields exactly 1.000**, because SSIM operates solely on luminance structure; removing chrominance preserves 100% of structural correlation."*

#### 2. Bilateral Skin Smoothing vs. Gaussian Blur
> *"A naive Gaussian blur smooths facial pores, but catastrophically smears high-contrast structural edges like the jawline, nostrils, and lips. Our beauty engine applies a **bilateral filter**:*
> $$I^{\text{filtered}}(p) = \frac{1}{W_p} \sum_{q \in S} I(q) \cdot \exp\left( -\frac{\|p - q\|^2}{2\sigma_s^2} \right) \cdot \exp\left( -\frac{\|I(p) - I(q)\|^2}{2\sigma_r^2} \right)$$
> *The spatial kernel $\sigma_s$ weights pixels by Euclidean distance, while the range kernel $\sigma_r$ weights pixels by radiometric color similarity. On flat skin where intensity differences are small ($\Delta \approx 5$), the range weight is $\approx 1.0$, producing full smoothing. Across a sharp jaw edge ($\Delta \approx 120$), the range exponential drops to near-zero, preserving edge sharpness completely. We combine this with a landmark-derived skin mask so eyebrows, eyes, and teeth are never touched."*

#### 3. Contrast-Limited Adaptive Histogram Equalization (CLAHE)
> *"Global histogram equalization maps cumulative distribution functions ($s = 255 \cdot \text{CDF}(r)$), but in photos with dark backgrounds, it violently over-amplifies noise. CLAHE divides the image into an $8 \times 8$ grid of contextual tiles, equalizes each independently, and clips histogram bins at clip limit $2.0$. The clipped excess is distributed uniformly across all bins before computing the CDF. Adjacent tiles are blended via bilinear interpolation to eliminate artificial boundary seams. To avoid color shifts, CLAHE is executed exclusively on the $L^*$ (lightness) channel in LAB color space, followed by a slight saturation boost."*

#### 4. Error Level Analysis (ELA) for Image Forgery
> *"In digital forensics, **Error Level Analysis** exploits the mathematics of lossy JPEG Discrete Cosine Transform (DCT) quantization. When an image is resaved repeatedly at quality $Q=90$, its 8×8 DCT coefficients reach an asymptotic error equilibrium. If a malicious user splices an object from another photo, that foreign object has a different compression history.  
> We resave the image in-memory at $Q=90$, calculate the absolute matrix residual:*
> $$E(x, y) = |I_{\text{original}}(x, y) - I_{\text{resaved}}(x, y)|$$
> *and scale it by amplification factor $S=15$: $E_{\text{amp}} = \min(255, 15 \cdot E(x,y))$. Untouched background pixels differ by $\le 1$ unit, remaining dark $(0, 15, 15)$, whereas spliced borders mismatch by $10\text{--}15$ units, fluorescing brightly at $(150, 180, 180)$."*

#### 5. SIFT Keypoint Matching for Copy-Move Tampering
> *"In copy-move attacks, a user copies an authentic region (e.g., foliage or a crowd member) and pastes it over another object. Because both patches share identical sensor noise and lighting, ELA alone is insufficient. We compute **SIFT (Scale-Invariant Feature Transform)** keypoints at Difference-of-Gaussians (DoG) scale-space extrema:*
> $$D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y)$$
> *Each keypoint generates a 128-dimensional gradient orientation descriptor. Descriptors are matched using FLANN. We enforce two crucial filters:*
> 1. *Lowe's ratio test: $d_1 / d_2 < 0.75$, ensuring the best match is distinct.*
> 2. *Spatial Euclidean pruning: $\|p_1 - p_2\| \ge 30\text{ px}$. This prevents false pairings between adjacent repetitive textures and isolates true spatial cloning displacements."*

#### 6. 3D Head Pose via Gram-Schmidt Orthonormalization
> *"Rather than using standard `cv2.solvePnP`—which fits a generic rigid 3D mannequin head and produces a $-36^\circ$ pitch error on frontal faces—we build the subject's own head coordinate frame directly from 3D MediaPipe landmarks using **Gram-Schmidt orthonormalization**:*
> - *Vector $x_h$: right eye corner (landmark 33) to left eye corner (landmark 263), normalized to unit length.*
> - *Vector $y_h$: forehead (landmark 10) to chin (landmark 152), with its component along $x_h$ subtracted to enforce strict $90^\circ$ orthogonality, then normalized.*
> - *Vector $z_h = x_h \times y_h$: cross product perpendicular to both, pointing into the screen.*
> *This produces rotation matrix $R = [x_h, y_h, z_h]$, from which true Euler angles (yaw, pitch, roll) are computed via $\text{atan2}$."*

---

### Segment 4: Live System Demonstration & Output Proof (15:00 – 24:00)

*(Walk the panel through each screen. Show what is clicked, and explain what is mathematically happening under the hood.)*

| Step | Tab / Screen | Action to Perform | What to Explain to the Panel |
|---|---|---|---|
| **1** | **Studio** | Click preset **Portrait Pro** | *"Notice how the background blurs immediately while the hair and shoulders stay sharp. The selfie segmenter generated a float32 mask $\alpha$, we applied a $21 \times 21$ Gaussian feather to simulate depth-of-field focus fall-off, and blended via $O = \alpha I + (1-\alpha)B$. Dragging the before/after slider shows zero color fringing."* |
| **2** | **Studio** | Adjust Brightness & Contrast; open **Results & Metrics** | *"Notice the real-time PSNR and SSIM update. The live histogram plots R, G, and B distributions. When we increase contrast, values stretch away from 128 using $F = \frac{259(c+255)}{255(259-c)}$."* |
| **3** | **Studio** | Turn on **Sunglasses & Party Hat** (AR Overlays) | *"These AR overlays are 100% procedural—no external PNG stickers. They are calculated in our face-local coordinate frame where the inter-ocular distance $d$ is unit length 1.0. When the face tilts or scales, the glasses automatically follow."* |
| **4** | **Face Analysis** | Load a group photo; toggle **Multi-Scale Tiling** | *"BlazeFace short-range resizes inputs to $128 \times 128$. On an $864 \times 890$ group photo, a 40-px face shrinks to 6 px and is missed (finding only 2 of 20 faces). When we toggle multi-scale tiling on, our 29 overlapping crops enlarge small faces by 3×, recovering 16 of 20 faces with zero false boxes under Non-Maximum Suppression (IoU 0.3)."* |
| **5** | **Face Analysis** | Show **Head Pose & Blink Counter** | *"Here the 3D axes are drawn on the nose tip. EAR (Eye Aspect Ratio) reads 0.34 when open, dropping below 0.20 when closed. A 2-frame state machine counts natural blinks while rejecting single-frame capture noise."* |
| **6** | **IPCV Lab** | Select **Noise & Denoising** (Salt & Pepper $\rightarrow$ Median Filter) | *"Here we demonstrate syllabus fundamentals: Gaussian blur smears salt-and-pepper noise, dropping PSNR, whereas a $5 \times 5$ median filter removes impulsive noise almost perfectly because an extreme 0 or 255 value never becomes the median."* |
| **7** | **IPCV Lab** | Select **Frequency Domain (2D FFT)** | *"We compute `np.fft.fft2`, apply `fftshift` to place DC at the center, and multiply by a 2nd-order Butterworth low-pass filter. The smooth transition avoids the ringing artifacts (Gibbs phenomenon) of an ideal filter."* |
| **8** | **Image Forgery** | Click **Run Full Forensic Analysis** on a tampered photo | *"Notice: no re-upload needed; it reads directly from the sidebar. The ELA map lights up the manipulated patch in bright fluorescent colors. The SIFT matching draws green correlation lines between duplicated copy-move regions. The deep CNN predicts authenticity confidence and overlays a Grad-CAM localization heatmap."* |
| **9** | **Live Camera** | Launch desktop window (`python live_camera.py`) | *"Pressing keys 1 through 0 toggles AR glasses, head pose, beauty mode, and portrait mode live at 30+ FPS on a standard CPU, proved by the on-screen FPS counter."* |

---

### Segment 5: Benchmarks, Robustness Testing & Limitations (24:00 – 27:00)

**Spoken Script:**
> *"To ensure scientific rigor, we conducted exhaustive comparative benchmarks across 8 test images (20 ground-truth faces) and 84 degraded test conditions:*
> 1. * **Rotation Robustness:** At $30^\circ$ head rotation, classical Haar cascades collapsed to finding faces in only 1 of 7 images because Haar rectangular features require upright facial geometry. MediaPipe BlazeFace succeeded in 6 of 7 images up to $45^\circ$.*
> 2. * **Speed:** A single BlazeFace detector pass executes in 4.8 ms on CPU, making it roughly **30× faster than a Haar cascade** (152 ms).*
> 3. * **Forensic Accuracy:** SIFT keypoint matching achieved 100% precision on copy-move duplicates with zero false pairings under our 30-px spatial Euclidean constraint.*
>
> *In the spirit of honest scientific reporting, we also document four key system limitations:*
> - *Under severe Gaussian noise ($\sigma = 50$), tiling produced 13 low-confidence false-positive boxes across 84 trials.*
> - *Fine, wispy hair strands are slightly softened by the selfie segmentation mask due to downsampling resolution.*
> - *Head pose pitch exhibits a consistent $\approx -12^\circ$ anatomical offset because human forehead-to-chin profiles are not strictly vertical.*
> - *Very small, dark faces (under 20 pixels) can still escape detection in heavy shadow."*

---

### Segment 6: Conclusion & Panel Defense Handoff (27:00 – 30:00)

**Spoken Script:**
> *"In summary, our project demonstrates that classical image processing and lightweight modern neural networks are not competing paradigms—they are complementary.*
>
> *By unifying creative photo studio editing, real-time facial analytics, and digital image forensics into a single explainable, CPU-efficient framework, we have built a complete, production-ready system that proves the practical power of IPCV principles.*
>
> *All source code, full datasets, unit tests, our 39-page technical guide, 5-page report, and presentation slides are publicly available in our GitHub repository.*
>
> *Thank you for your time and attention. We now welcome questions from the evaluation panel."*

---

# PART 2: THE EXAMINER VIVA DEFENSE PLAYBOOK
### (Top 12 Questions Panels Love to Ask & Winning Answers)

#### Q1: "Why not just use Viola-Jones Haar Cascades for face detection?"
* **Answer:** *"We benchmarked Haar against BlazeFace directly. While Haar is effective on upright frontal faces, it fundamentally fails under head rotation (dropping to 14% recall at $30^\circ$ rotation vs. 86% for BlazeFace). Furthermore, Haar returns only a bounding box with no landmarks or confidence calibration, and takes ~150 ms per pass compared to BlazeFace’s 4.8 ms (30× speedup). We retain Haar in the app solely as a measured comparative baseline."*

#### Q2: "Why does SSIM give 1.000 when comparing a color image to its greyscale version?"
* **Answer:** *"SSIM is mathematically formulated to evaluate luminance structure. In standard implementations, images are converted to greyscale ($Y = 0.299R + 0.587G + 0.114B$) before computing local means, variances, and cross-correlation over the $11 \times 11$ Gaussian window. Because greyscale conversion preserves 100% of the luminance distribution, structural cross-correlation $\sigma_{xy}$ equals variance $\sigma_x^2$, yielding $\text{SSIM}=1.000$. This proves that SSIM measures structural change rather than perceptual color fidelity."*

#### Q3: "What is the physical principle behind Error Level Analysis (ELA)?"
* **Answer:** *"Lossy JPEG compression quantizes 8×8 DCT frequency coefficients. In an authentic JPEG, repetitive compression reaches asymptotic error equilibrium. When a foreign element is spliced in or modified with an editing tool, that region has a different compression generation and higher entropy. Resaving at reference quality $Q=90$ and scaling by 15 illuminates this difference: untouched areas have stabilized error ($\Delta \le 1$), while spliced patches show high residual error ($\Delta \ge 11$), fluorescing brightly."*

#### Q4: "In SIFT copy-move detection, how do you prevent false matches on repetitive textures like grass or brick walls?"
* **Answer:** *"We implement a two-stage filter: first, Lowe’s ratio test ($d_1 / d_2 < 0.75$) checks that the nearest descriptor is distinctly closer than the second-nearest. Second, and most importantly, we enforce a Euclidean spatial distance threshold $\|p_1 - p_2\| \ge 30\text{ px}$. Identical features within the same texture patch are located in close physical proximity and are discarded, ensuring only non-local spatial displacements (cloned patches) are flagged."*

#### Q5: "What is PRNU, and how does it detect image tampering?"
* **Answer:** *"Photo-Response Non-Uniformity (PRNU) is an intrinsic high-frequency stochastic pattern caused by microscopic silicon physical imperfections in digital camera sensors. It acts as an invisible physical fingerprint. We extract this residual noise using a Gaussian high-pass filter ($W = I - G_\sigma(I)$) and measure local variance across $32 \times 32$ blocks. In a spliced composite or generative image, the noise variance of the tampered patch diverges significantly from the global sensor mean (exceeding 2.5 standard deviations)."*

#### Q6: "Why did you use Gram-Schmidt orthonormalization for head pose instead of solvePnP?"
* **Answer:** *"OpenCV's `solvePnP` fits 2D landmarks to an idealized, rigid 3D mannequin model. Because human facial proportions vary significantly, fitting a generic mannequin produced a massive $-36^\circ$ pitch error on frontal faces looking straight at the camera. Gram-Schmidt uses MediaPipe's own predicted 3D coordinate depths to construct the subject's actual personal facial axes, eliminating the rigid model assumption and reducing error to natural baseline values."*

#### Q7: "Why does BlazeFace miss small faces in group photos, and how did tiling fix it?"
* **Answer:** *"BlazeFace short-range is optimized for mobile front-camera selfies; it internally downsamples the full input frame to $128 \times 128$. In an $864 \times 890$ group photo, a 40-px face becomes only $\approx 6\text{ px}$, falling below the convolutional receptive field. Our multi-scale tiling system runs an image pyramid of 29 overlapping crops ($2\times 2$, $3\times 3$, $4\times 4$ with 25% overlap). Each tile enlarges small faces by up to 3×, lifting recall from 2/20 (10%) to 16/20 (80%) with duplicates merged via NMS (IoU 0.3)."*

#### Q8: "How does the bilateral filter smooth skin without blurring edges?"
* **Answer:** *"The bilateral filter multiplies two Gaussian kernels: a spatial kernel based on geometric distance ($\sigma_s$) and a radiometric range kernel based on color intensity difference ($\sigma_r$). Within smooth skin pores, intensity differences are small ($\Delta \approx 5$), so the range weight is close to 1.0 (acting like a normal blur). Across high-contrast boundaries like the jawline or lips ($\Delta \ge 100$), the range exponential decays to near zero, giving boundary pixels zero weight and preserving sharp edges."*

#### Q9: "Why is CLAHE applied to the L* channel of LAB rather than RGB?"
* **Answer:** *"Applying histogram equalization to R, G, and B independently corrupts color balance because the relative channel ratios are distorted. Converting to CIE LAB separates perceptual lightness ($L^*$) from chromatic opponent channels ($a^*$ green-red, $b^*$ blue-yellow). Equalizing only $L^*$ enhances local contrast without altering chromaticity."*

#### Q10: "How do your procedural AR overlays adapt to face tilt and scale?"
* **Answer:** *"We define a normalized facial coordinate system (`FaceFrame`) using the 3D eye landmarks. The origin is the midpoint between both eye centers, the unit length $d$ is the Euclidean inter-ocular distance, and axes $e_1$ and $e_2$ align with the eye vector and chin direction. Overlays like sunglasses are mathematically constructed in units of $d$ (e.g., lens radius $0.47d$, center at $u = \pm 0.5$). Mapping to pixels via $\text{pixel} = \text{origin} + d(u \cdot e_1 + v \cdot e_2)$ guarantees automatic scaling and rotation without raster artifacts."*

#### Q11: "Why does Butterworth filtering avoid the ringing artifacts of ideal frequency filters?"
* **Answer:** *"In the frequency domain, an ideal filter has a discontinuous sharp cutoff, which corresponds in the spatial domain to a Sinc function ($\frac{\sin x}{x}$) with oscillating side lobes (the Gibbs phenomenon), causing visible ringing halos around edges. The $n$-th order Butterworth filter transfer function $H(u,v) = \frac{1}{1 + (D(u,v)/D_0)^{2n}}$ provides a smooth, continuous transition without spatial oscillations."*

#### Q12: "How does the live webcam run 15 complex effects at 30 FPS entirely on a laptop CPU?"
* **Answer:** *"Three reasons: First, live mode processes frames at $960 \times 540$ or $640 \times 480$, bypassing heavy multi-scale tiling. Second, MediaPipe's lightweight TFLite models run inference in ~5 ms with CPU XNNPACK vector instructions. Third, all compositing and filter operations use compiled C-backed NumPy vectorization and OpenCV primitives without Python pixel loops."*
