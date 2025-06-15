# From: https://www.nature.com/articles/s41467-024-55236-4#peer-review

sample_paper_one_abstract = """
Non-Hermitian models describe the physics of ubiquitous open systems with gain and loss. One intriguing aspect of non-Hermitian models is their inherent topology that can produce intriguing boundary phenomena like resilient higher-order topological insulators (HOTIs) and non-Hermitian skin effects (NHSE). Recently, time-multiplexed lattices in synthetic dimensions have emerged as a versatile platform for the investigation of these effects free of geometric restrictions. Despite holding broad applications, studies of these effects have been limited to static cases so far, and full dynamical control over the non-Hermitian effects has remained elusive. Here, we demonstrate the emergence of topological non-Hermitian corner skin modes with remarkable temporal controllability and robustness in a two-dimensional photonic synthetic time lattice. Specifically, we showcase various dynamic control mechanisms for light confinement and flow, including spatial mode tapering, sequential non-Hermiticity on-off switching, dynamical corner skin mode relocation, and light steering. Moreover, we establish the corner skin mode’s robustness in the presence of intensity modulation randomness and quantitatively determine its breakdown regime. Our findings extend non-Hermitian and topological photonic effects into higher synthetic dimensions, offering remarkable flexibility and real-time control possibilities. This opens avenues for topological classification, quantum walk simulations of many-body dynamics, and robust Floquet engineering in synthetic landscapes.
"""

sample_paper_one_main = """
### Introduction
Non-Hermitian systems host a range of intriguing phenomena in
physics, such as reconfigurable light routing1, potential for enhanced
sensitivity2,3 and unidirectional invisibility4, that are deeply rooted in
symmetry and topology. One such phenomenon is the non-Hermitian
skin effect (NHSE) where a macroscopic fraction of the eigenmodes of
a finite system become exponentially localized at its boundary5,6. This
localization is a direct consequence of the nontrivial (topological)
winding of the system’s eigenvalues in the complex energy plane7–9.
Disorder and small variations in the systemdo not change the winding
number which is a topological invariant9.
Over the last few years, the NHSE has been demonstrated on a
variety of platforms5,10,11. Exemplary platforms include acoustics and
phononics12, topo-electric circuits13, and photonics14. These developments
are in part motivated by the profound impact of NHSE on band
topology7,15,16, spectral symmetry17, and dynamics18,19. Particularly in
photonics, recently the NHSE has enabled intriguing demonstrations
of the tuneable directional flow of light20, near-field beam steering21,
engineering arbitrary band topology22 and topological funneling of
light14. Nevertheless, these demonstrations have been limited to systems
that can be effectively described by time-independent
Hamiltonians23. The introduction of time-dependent non-Hermitian
Hamiltonians can lead to a dynamic control over the skin effect and
also lead to fundamental advances in novel non-Hermitian topological
phases that are not accessible using time-independent systems. Here
we demonstrate dynamical control of the two-dimensional non-Hermitian
photonic skin effect, that is, corner skin modes, using purely
synthetic temporal dimensions. Specifically, using time-multiplexed
light pulses in fiber loops,we showmanipulation of the gain/loss in the
system at a scale that is faster than the dynamics of light pulses in the
lattice. Using this dynamical manipulation, we demonstrate gradual
control over the degree of localization of the corner skin modes, gradual
tweezing of light where we move the corner skin modes along a
predefined trajectory in the lattice, and 2D funneling of light where
photons always funnel to the corner skin modes irrespective of their
initial position in the 2D lattice. Finally, by introducing controlled
disorder in the system in the form of random variations in gain and
loss, we quantitatively investigate the robustness of the corner skin
modes against such disorders. Our work opens up an avenue to
explore the rich physics of time-dependent non-Hermitian models
such as non-Hermitian Floquet systems.
2d quantum walk with non-Hermitian topology
Our system simulates a discrete-time quantum walk of photons on a
two-dimensional non-Hermitian square lattice, as illustrated schematically
in Fig. 1a. Specifically, we implement a split-step walk where the
walker first randomly steps to either left or right (corresponding to the
X direction) with equal probability, then up or down (corresponding to
the Y direction). To introduce non-Hermiticity, we introduce an additional
gain e+δx when the walker steps to the left, and an additional loss
e−δx when the walker steps to the right. Similarly, the walker experiences
a gain e+δy whenmoving down and a loss of e−δy when moving up.
The parameters δx and δy then indicate the degree of non-Hermiticity
of the walk.
For this quantum walk, a concept of complex energy can analogously
be defined, by solving for the eigenmodes of the non-unitary
quantum walk evolution operator Û and taking the logarithm of the
corresponding eigenvalue uj. Namely, this can be formulated as ϵj =
ilog(uj), where Û |uj〉 = uj |uj〉. If we further impose periodic boundary
conditions (PBCs) in both x and y for the bulk in Fig. 1a, we can apply
the Bloch theorem for the walk and obtain the complex energy bands
ϵup/down(kx, ky). (The two bands seen in Fig. 1b arise due to the up/down
channel configuration of our experiment, see Supplementary Information
(SI) for derivation details). The non-unitary time evolution of
the walk leads to a nontrivial winding of ϵ(kx, ky) for each band in the
complex energy plane as one continuously varies Bloch vector (kx, ky)

Fig. 1 | Two-dimensional quantum walk in non-Hermitian synthetic lattice.
a Example of photonic quantum walk in a 2D synthetic lattice. The blue and red
curved arrows show the direction-dependent loss and gain. b Winding of effective
eigenenergies ϵup/down(kx, ky) in the complex energy plane for a single bulk non-
Hermitian lattice with periodic boundary condition, showing line-gapped topology
(indicated by the green line). Herewechoose five different values ky ≡ 0, ± π/4, ± 3π/
8. c Four bulk lattices with different gain-loss patterns are glued along their edges
to form a corner. Note that δx > 0 implies gain for a step towards −X and loss for a
step toward +X. For δx < 0 the gain-loss is inverted. A similar rule applies for δy.
d Averaged spatial profile of corner skin modes formed in the systemshown in c, by
taking non-Hermitian parameters δx = δy = 0.175. The lattice size is 30 × 30. e The
time-multiplexed experimental scheme, with which the lattice parameters can be
(dynamically) controlled by the intensity modulators. EDFA: Erbium-doped fiber
amplifier. PD: Photodiode.

Fig. 2 | Light localization and light funneling for static control. a Light localization
at the corner skin modes located at (x, y) = (0, 0) for non-Hermitian parameter
|δx | =|δy |= 0.175. Here a single pulse is initialized at (x, y) = (0, 0) in the up
channel. b Light funneling for the same lattice parameter and pulse initialization,
but the corner skin mode is located at (x, y) = (−10, 10). Here the skin effect allows
light to flow to the corner skinmode and localize there. In both a and b, fromleft to
right the snapshots are shown for time steps 1, 9, 21, respectively.

along a certain curve in the Brillouin zone. To illustrate this, in Fig. 1b,
we plot the complex energies ϵup/down(kx, ky) of the bulk lattice shown
in Fig. 1a as we vary kx from −π to π while keeping ky fixed to different
values 0, ± π/4, ± 3π/8. As (kx, ky) varies along each of these directed
horizontal curves in the Brillouin zone, both ϵup(kx, ky) andϵdown(kx, ky)
winds one loop in the counterclockwise direction, thus exhibiting an
integer-valued winding number −1. This is a topological invariant for
our non-Hermitian quantum walk. Also, the two winding loops contributed
from the two bands ϵup/down exhibit a line-gapped topology24,
such that the two winding loops never cross the line Re(ϵ) = 0 in the
complex plane.Windings of complex energy along other curves in the
Brillouin zone are shown in the supplementary information (SI)
section 3.
In a finite system, the nontrivial winding of the complex energies
and the associated 2D non-Hermitian skin effect24 is manifested as
corner skin modes, that is, localization of the walker can happen at an
interface between regions with opposite windings (or bulk band
topologies). Figure 1c shows one exemplary casewhich consists of four
distinct regions, represented by the four different color patches. The
gray patch is identical to the system described in Fig. 1a. The other
three regions exhibit an inverted gain-loss relation (indicated by a
change in the sign of the gain parameter) either along the x or y-axis, or
both. This inversion of gain-loss leads to different windings for each
region. Non-Hermitian skin effect occurs in such a system, and we
numerically verify in Fig. 1d that the averaged eigenmodes of the
quantum walk exhibit clustering at the junction between the four
regions - as indicated by the red dot in Fig. 1c.
To simulate the quantum walk described above, we use classical
light pulses in a time-multiplexed setup shown in Fig. 1e. We note that
for this linear system, the evolution of classical light pulses in the
lattice exactly follows that of the quantum walk of single photons in
the lattice. We map the state space of the 2D square lattice of size
30 × 30 into different time-delays in two fiber feedback loops, as
introduced in previous works25,26. To introduce non-Hermiticity, we
use four intensity modulators that introduce individually controllable
loss when the walker moves along any direction. We also use two
erbium-doped fiber amplifiers (EDFAs) that provide gain in the system,
and together with the intensity modulators, introduce a gain-loss
mechanism that can be controlled at each step of the walker. We
specifically choose electro-optic modulators with a high bandwidth to
allow reconfigurability of the system’s topology at each step of the
quantum walk. A full discussion of the experimental setup is provided
in the SI sections 1 and 2.

### Results
Results
Skin effect under static control
To show the presence of non-Hermitian corner skin modes, we first
construct the model as shown in Fig. 1c, with the corner located at the
lattice origin (x=0, y = 0). We inject a single light pulse into the time
bin corresponding to the lattice origin and choose non-Hermitian
parameters |δx| and |δy| to be 0.175 as in Fig. 1c. In Fig. 2a, we plot the
snapshots of the light distribution in the lattice for different time steps
1, 9, and 21, which are obtained by measuring the pulse power at each
time bin. The evolution of distribution shows that the walker stays
localized at the origin, confirming the presenceof a corner skinmode.
In sharp contrast, when we set δx and δy to 0, we observe a significant
spreading of the intensity distribution, indicating the absence of any
corner skin modes (see Supplementary Sections 4 and 5 in the SI for
the experimental data).
Having shown the localization of light at the corner skinmode, we
next demonstrate the skin-effect-induced funneling of light. Namely,
the system dynamics bring any initial state towards the corner skin
modes. We set the corner skin mode to be at the lattice site (x = −10,
y = 10) while light pulses are still injected at x=0, y = 0, which is now in
the bulk of the lattice (Fig. 2b). As the system evolves, initially light
spreads in bulk, but finally converges to the corner site. As shown in
the SI for several different lattice configurations, light pulses always
converge to the corner regardless of the initialization location. This
funneling of light to the corner skinmode is amanifestation of the skin
effect where all the eigenmodes of the system are localized at the
corner. Schematic illustration of this funneling effect can be seen in
Supplementary Movie 1. Our experimental results are in good agreement
with our theoretical prediction shown in Fig. 2b.
Dynamically controlling the non-Hermitian lattice and
skin effect
The use of time as a synthetic dimension allows us to dynamically
reconfigure our non-Hermitian lattice as a function of time. Specifically,
by controlling the intensity modulators at each time step of the
quantum walk, we achieve temporal modulation of the gain/loss
parameters δx(t) and δy(t) such that they are time-dependent. Using
this time dependence, first, we demonstrate dynamical control over
the degree of localization of the corner skin modes. At the start of the
evolution, we adopt the configuration as in Fig. 1c and set |δx(0)| =
|δy(0)| = 0.175, and inject a single light pulse at the corner skin mode
situated at the origin. As the systemevolves, we reduce both |δx|,|δy| by
50% for every four time-steps and continue doing so until step 16

Fig. 3 | Dynamical control of the corner skinmode. a Dynamical control of corner
skin mode spatial profile. As the non-Hermitian parameter is gradually reduced
from |δx,max | = |δy,max | = 0.175 to |δx,max | = |δy,max | = 0.02 and back to |δx,max | = |
δy,max | = 0.175, the corner skin mode becomes delocalized and then localized. From
left to right the snapshots are shown for time steps 1, 9, 17, 25, 37, respectively.
bDynamically tweezing localized light along a designed “L”-shaped trajectory using
the skin effect. Localized light is firstmoved in the +Y direction for 8 steps and then
to the −X direction for 10 steps.

(Fig. 3a). Because of this reduction, we observe that the corner skin
modes become less confined to the origin. This is because the smaller
non-Hermitian parameter exhibits eigenmodes distributed over a larger
area, as predicted theoretically (see Supplementary Fig. S4 in the
SI). Thereafter, starting fromstep 17, we reverse the process, that is,we
increase the gain /loss parameters |δx|,|δy| back to its original value at
the same rate. We now observe a relocalization of light at the origin.
Next, we demonstrate gradual repositioning of the corner skin
modes in the lattice. We use the same lattice geometry shown in Fig. 1c
and fix the non-Hermitian parameter to δx = δy = 0.175. As the system
evolves, we gradually move the interface between the four distinct
topological regions, repositioning the corner skin mode as a function
of time. We first move the position of the corner skin mode upwards
for 8 unit cells, and then leftward for 10 unit cells. As before, we inject
light pulses at the corner skinmode. As the systemevolves,we observe
that the center of the intensity distribution follows the position of the
corner skin mode as it gradually moves along the given L-shaped trajectory
from its initial location (x=0, y=0) to its final location at
(x = −10, y= 8). Furthermore, during this process, the intensity distribution
remains tightly localized close to the corner skin mode. Evidently,
the corner serves as a non-Hermitian tweezer of light, which
allows us to gradually move trapped photons along a given trajectory
in the synthetic lattice. Note that non-Hermitian light steering has been
demonstrated in real-space lattices1, and our demonstration in synthetic
time dimensions portends the potential for such photonic
control using the temporal degree of freedom of light.

Fig. 4 | Robustness of the skin effect at the presence of different degrees of
lattice disorder. The randomness is introduced to the intensity modulation of the
lattice and the pulse is injected at (x, y) = (0, 0). a, b Experimental observation of
robustness of the corner skin mode and skin effect in a lattice with moderate
disorder (ηy = ηx = 0.5, 1).Here the disorder leads to a relaxed spatial confinement of
lightwithout breaking the localization of light. c, d Breakdown of the localization in
the presence of strong disorder (ηy = ηx = 1.5, 2). Light can diffuse arbitrarily far
away until they are limited by the size of the experiment. In a–d, fromleft to right
the snapshots are shown for time steps 1, 5, 13, respectively.

Fig. 5 | Breakdown threshold of the robustness. a Theory and b Experimental
evolution of space-averaged displacement <r2> = <x2 + y2> as a function of step
number under different lattice disorders. For disorder strength lower than the
threshold η = 1, the disorder increases the effective spatial diameter of the corner
skin mode, as shown in the evolution of average displacement with time. For disorders
higher than the threshold, light diffusively spreads to large distances on the
lattice. Standard deviations are also shown at each step.

Schematic illustrations of the tapering and relocation effects can
be seen in Supplementary Movies 2 and 3, respectively.
Robustness of the skin effect
The topological nature of the non-Hermitian skin effect ensures its
robustness against disorder in gain/loss parameters δx, δy. To quantitatively
investigate this robustness, we introduce a disorder on the gain/
loss. At each lattice site, we randomly pick both δx, δy from a uniform
distribution on the interval [δmax(1 − η), δmax], where max is the maximum
gain parameter and is the disorder parameter which quantifies
the variance of the gain parameter. In our experiment, we vary the
disorder parameter between 0 (no disorder) and 2 (max disorder).
We find that the skin effect is robust when the disorder parameter
η <1. InFig. 4a, b,we plot the evolution of light pulses in the lattice for two
different values η=0.5 and η = 1. For both cases, we inject light pulses at
the corner skinmode located at the origin. Weobserve that, even though
the localization of the intensity distribution reduces as the disorder
increases, the distribution still stays confined around the origin, indicating
the existence of corner skin modes even in the presence of disorder.
Nevertheless, once we increase the disorder parameter to 1.5 and 2
(Fig. 4c, d), the intensity distribution diffusively spreads away from the
origin, indicating the breakdown of corner skinmode. Our experimental
observation agrees with the intuitively expected behavior that, for η <1,
even though there is a disorder in themodulation amplitudes, the gain for
the step towards the corner is always larger than that of the outwards
direction. Thus the four regions maintain their distinct non-Hermiticity
and the corner skin mode exists. But, when η > 1, a direction-dependent
gain for the time steps is no longer always valid and therefore the four
regions are no longer distinct and the corner skin mode ceases to exist.
To better characterize the robustness and breakdown of the skin
effect, we compute the evolution of the mean-square displacement of
the intensity distribution in the lattice as a function of time. Themeansquare
displacement is quantified as <r2> (n) = Σx,y Px,y(n)(x2 + y2),
where Px,y(n) is the time-varying intensity distribution of light. Figure 5
shows the calculated <r2> (n) for several values of the disorder parameter
for both theoretical calculations and experimentally measured
values. Each experimental curve corresponds to an average of eight
independent experimental realizations of disorder, while the theory
corresponds to eight averages. Due to the limited size of the lattice
(30 × 30), we only collect data from step 1 to step 15, and plot <r2> (n)
for the odd steps. The violet, red, and green curves correspond to the
weak disorder, with disorder parameters being 0, 0.5, and 1, respectively.
All three curves saturate to a fixed value which is well below a
certain threshold. This behavior thus implies the robustness of the skin
effect. However, for larger disorders of 1.5 and 2, corresponding to the
yellow and blue curve, the mean squared distance does not converge.
Instead, it spreads out until it becomes limited by the finite size of the
lattice, indicating a complete breakdown of the skin effect.

### Discussion
In conclusion, we demonstrated robust dynamical control over the
photonic non-Hermitian skin effect in a 2D synthetic lattice. We created
a corner skinmode that localizes light and dynamically tuned the
degree of light localization.Moreover,we dynamically steered trapped
light along any given trajectory in the 2D lattice. We also demonstrate
the robustness of the skin effect under lattice disorder below a certain
threshold. Our results demonstrate that useful control mechanisms in
spatial landscapes such as reconfigurable light steering1 can be
extended to synthetic dimensions.
Looking forward, the dynamic techniques developed in this work
can be further applied to investigate Floquet non-Hermitianmodels27–29,
in particular in synthetic dimensions30. Further, one can create an analogue
of on-site interaction by imposing a nonlinear phase shifter after
the linear optical transformations, and investigate non-Hermitian
models of interacting particles31. Such nonlinearities could also have
implications in the recently discovered regime of topological frequency
combs32–34 as well as temporalmode-locked lasers35 due to the periodic
temporal pulses that define our platform. Moreover, the two-fold spin
characteristics in our system can potentially be extended to non-
Hermitian models for lattice gauge theories with higher spins and non-
Abelian statistics, by increasing the number of loops36. Another intriguing
direction can be exploring NHSE-enabled morphing of photonic
topological modes which was recently demonstrated in mechanical
lattices37. Finally, our non-hermitian lattice can be enriched with engineered
synthetic gauge fields38 as demonstrated recently for both
Hermitian39 and non-Hermitian models20, to explore intriguing proposals
such as the quantum Skin Hall effect40.

### Methods
Experimental setup
To encode the 2D lattice in time we consider two fiber loops shown in
Fig. 1, labeled up channel and down channel. The length of each fiber
loop is ∼3 (km), and one circulation of light in the loop is equivalent to
one step of the walk. Hence, we can encode the entire 2D lattice within
a time-duration (or time-delay) of ∼15,000 (ns) without mixing timebins
in step n and step (n + 1).We first encode 30 Y-time bins in both
the up and down channels, each of time duration 250 (ns) in a total
time duration of 7500 (ns). Each Y-time bin is then occupied by 30
X-time bins, each of time duration 7.5 (ns). At any time, the state of the
system is thus represented by a complex vector (Ux,y, Dx,y), encoded in
the phase and amplitude of the light pulse circulating in the two
fiber loops.

Measurement
To initialize the system, we inject a single pulse into the down channel
of the fiber loop. We use a continuous wave CW laser with 1550 (nm)
wavelength (Optilab DFB-1551-SM-10) and by modulation of this laser
using a Thorlabs SOA (SOA1013SXS), we have generated pulses of
width ∼6 (ns) at a repetition rate of 1 (pulse/ms). We then control the
polarization of the laser with an inline fiber polarization control (PC)
before injecting the light into the down channel with a 90/10 beam
splitter. Note that we use two identical 90/10 beam splitters, one for
each channel. The 90/10 beam splitter in the down channel is used to
inject light into the quantum walk, whereas the 90/10 beam splitter in
the up channel is used to weakly couple light pulses out of the quantum
walk so that we can measure the pulse power after n steps of
evolution using the up channel’s PD. Note that the EDFA placed
immediately prior to the up channel’s PD ismerely used to amplify the
light pulses coming out of the quantum walk experiment, making it
easier for the PD to detect it.
As a pulse enters the system, by defaultwe recognize it as entering
the (x=0, y = 0) time bin, and thus the initial state is D0,0 = 1.The pulse
then sequentially passes through a 50/50 beamsplitter denoted as ±Xbeamsplitter,
a pair of time-varying intensitymodulators (Optilab IMP-
1550−20-PM) is used to impose the correct gain/loss as each time bin
(x, y) passes through it, controlled by RF signal generated from Teledyne
Lecroy arbitrary waveform generator (T3AWG3252). We then
impose a delay of 3 (m) in the up channel and no delay in the down
channel. The same procedure then repeats for Y.
To combat photon loss in the walk, we use two Thorlabs erbiumdoped
fiber amplifiers (EDFA) (EDFA100S), one for each channel.
Before amplifying the pulse, we use wavelength division multiplexers
(WDM) (DWDM-SM-1-34-L-1−2) to couple a 1543 (nm) CW laser (DFB-
1543-SM-30) to the pulses so that the spontaneous emission noise
during the amplification is reduced. We decouple the 1550 (nm) pulses
from the 1543 (nm) CW laser with the same WDM after the amplification
is done. Finally,we use PC to ensure the correct linear polarization
for the 1550 (nm) signal pulses. After this, a complete quantum walk
step is finished.
"""

sample_paper_one_references = """
1. Chalabi, H. et al. Synthetic gauge field for two-dimensional timemultiplexed
quantum random walks. Phys. Rev. Lett. 123, 150503
(2019).
2. Chalabi, H. et al.Guiding and confining of light in a two-dimensional
synthetic space using electric fields. Optica 7, 506–513 (2020).
3. Chen, W., Kaya Özdemir, Ş., Zhao, G., Wiersig, J. & Yang, L.
Exceptional points enhance sensing in an optical microcavity.
Nature 548, 192–196 (2017).
4. Feng, L. et al. Experimental demonstration of a unidirectional
reflectionless parity-time metamaterial at optical frequencies. Nat.
Mater. 12, 108–113 (2013).
5. Flower, C. J. et al. Observation of topological frequency combs.
Science 384, 1356–1361 (2024).
6. Gliozzi, J., De Tomasi, G. & Hughes, T. L. Many-body non-hermitian
skin effect for multipoles. arXiv 2401, 04162 (2024).
7. Gong, Z. et al. Topological phases of non-hermitian systems. Phys.
Rev. X 8, 031079 (2018).
8. Mehrabad, M. J. & Hafezi, M. Strain-induced landau levels in photonic
crystals. Nat. Photonics 18, 527–528 (2024).
9. Mehrabad, M. J., Mittal, S. & Hafezi, M. Topological photonics:
Fundamental concepts, recent developments, and future directions.
Phys. Rev. A 108, 040101 (2023).
10. Kawabata, K., Shiozaki, K. & Ueda,M. Anomalous helical edge states
in a non-hermitian chern insulator. Phys. Rev. B 98, 165148 (2018).
11. Leefmans, C. R. et al. Topological temporally mode-locked laser.
Nat. Phys. 20, 1–7 (2024).
12. Liang, Q. et al. Dynamic signatures of non-hermitian skin effect and
topology in ultracold atoms. Phys. Rev. Lett. 129, 070401 (2022).
13. Lin, Q., Yi, W. & Xue, P. Manipulating directional flow in a twodimensional
photonic quantum walk under a synthetic magnetic
field. Nat. Commun. 14, 6283 (2023).
14. Lin, R., Tai, T., Li, L. & Lee, C. H. Topological non-hermitian skin
effect. Front. Phys. 18, 53605 (2023).
15. Lin, Z. et al. Observation of topological transition in floquet nonhermitian
skin effects in silicon photonics. Phys. Rev. Lett. 133,
073803 (2024).
16. Liu, Y. G. N. et al. Complex skin modes in non-hermitian coupled
laser arrays. Light.: Sci. Appl. 11, 336 (2022).
17. Yuhao, M. A. & Hughes, T. L. Quantumskin hall effect. Phys. Rev. B
108, L100301 (2023).
18. Mittal, S., Moille, G., Srinivasan, K., Chembo, Y. K. & Hafezi, M.
Topological frequency combs and nested temporal solitons. Nat.
Phys. 17, 1169–1176 (2021).
19. Nasari, H., Pyrialakos, G. G., Christodoulides, D. N. & Khajavikhan, M.
Non-hermitian topological photonics.Optical Mater. Express 13,
870–885 (2023).
20. Nitsche, T. et al. Quantum walks with dynamical control: graph
engineering, initial state preparation and state transfer. N. J. Phys.
18, 063017 (2016).
21. Okuma, N. & Sato, M. Non-hermitian topological phenomena: A
review. Annu. Rev. Condens. Matter Phys. 14, 83–107 (2023).
22. Okuma, N., Kawabata, K., Shiozaki, K. & Sato, M. Topological origin
of non-hermitian skin effects. Phys. Rev. Lett. 124, 086801 (2020).
23. Pang, Z., Wong, B. T. T., Hu, J. & Yang, Y. Synthetic non-abelian
gauge fields for non-hermitian systems. Phys. Rev. Lett. 132,
043804 (2024).
24. Schreiber, A. et al. A 2d quantum walk simulation of two-particle
dynamics. Science 336, 55–58 (2012).
25. Wang, K., Dutt, A., Wojcik, C. C. & Fan, S. Topological complexenergy
braiding of non-hermitian bands. Nature 598, 59–64 (2021).
26. Wang, K. et al. Generating arbitrary topological windings of a nonhermitian
band. Science 371, 1240–1245 (2021).
27. Wang, K., Xiao, L., Budich, J. C., Yi,W. & Xue, P. Simulating exceptional
non-hermitian metals with single-photon interferometry.
Phys. Rev. Lett. 127, 026404 (2021).
28. Wang, W., Wang, X. & Ma, G. Non-hermitian morphing of topological
modes. Nature 608, 50–55 (2022).
29. Weidemann, S. et al. Topological funneling of light. Science 368,
311–314 (2020).
30. Weidemann, S., Kremer, M., Longhi, S. & Szameit, A. Topological
triple phase transition in non-hermitian floquet quasicrystals. Nature
601, 354–359 (2022).
31. Wiersig, J. Review of exceptional point-based sensors. Photonics
Res. 8, 1457–1467 (2020).
32. Xiao, L. et al. Observation of non-bloch parity-time symmetry and
exceptional points. Phys. Rev. Lett. 126, 230402 (2021).
33. Yao, S. & Wang, Z. Edge states and topological invariants of nonhermitian
systems. Physical Rev. Lett. 121, 086803 (2018).
34. Yokomizo, K. & Murakami, S. Non-bloch band theory of nonhermitian
systems. Phys. Rev. Lett. 123, 066404 (2019).
35. Yuan, L., Lin,Q., Xiao,M. & Fan, S. Synthetic dimension in photonics.
Optica 5, 1396–1405 (2018).
36. Zhang, X., Zhang, T., Lu, M.-H. & Chen, Y.-F. A review on nonhermitian
skin effect. Adv. Phys. X 7, 2109431 (2022).
37. Zhao, H. et al. Non-hermitian topological light steering. Science
365, 1163–1166 (2019).
38. Zhou, L. & Zhang, D.-J. Non-hermitian floquet topological matter–a
review. arXiv 2305, 16153 (2023).
39. Zhou, Q. et al. Observation of exceptional points and skin effect
correspondence in non-hermitian phononic crystals.Nat.Commun.
14, 4569 (2023).
40. Zou, D. et al. Observation of hybrid higher-order skin-topological
effect in non-hermitian topolectrical cir- cuits. Nat. Commun. 12,
7201 (2021).
"""

one_shot_review_example = """
Reviewer #1 (Remarks to the Author):
Manuscript by Ding et al.
Title: Paleomagnetic evidence for Neoarchean plate mobilism
This manuscript presents a new 2.48 Ga paleomagnetic pole from the Elbow Creek mafic dykes of the Wyoming carton. The paleomagnetic data presented here is of high-quality (key pole), with a positive field stability test (positive baked contact test). The new 2.48 Ga pole is used here to explore the relative horizontal plate motions between crustal blocks (Wyoming and Superior cratons) and the data is used to study the operation of plate tectonics. (i.e., plate mobilism). Based on differing paleomagnetic apparent polar wander paths of Wyoming and Superior in during 2.7-2.5 Ga authors make a well justified suggestion that the plate tectonics was in operation prior to 2.5 Ga.
Plate tectonics have had major impact on Earth’s geologic, climatic and biologic record, but onset time of plate tectonics is debated. Time estimates range from the Hadean to the Neoproterozoic depending on the used proxy (e.g., geology, geochemistry, paleomagnetism, geodynamic models). The results from this study will interest many people in several geoscience disciplines as the onset of plate tectonics is one of the major questions in the Earth sciences. As geological record of the operation of plate tectonic may have been lost, the paleomagnetic method remains one of the most robust methods to explore the relative horizontal motion of tectonic plates (i.e. plate tectonics) at Archean.
The manuscript is well written and it is an excellent and clear compilation of data. By combining paleomagnetic data with proxies for lithospheric rigidity and petrological and metamorphic evidence for subduction the manuscript presents a very nice study of the onset of plate tectonics. Illustration of the manuscript is of very high quality, especially Figure 3, which combines proxies for plate tectonics.
The chosen selection between main body text and supplementary information is adequate. I have one major concern and few minor comments on the manuscript.
Baked contact test.
Baked contact tests are the key for this study in providing proofs of the primary magnetization. Authors claim that there are positive baked contact tests for the dykes J20S24 (cutting anorthosite, ca. 2710 Ma Stillwater Complex) and J20S28 (cutting granite, ca 2800 Ma Mouat quartz monzonite in Fig. 1).
Dyke J20S24. Authors write that the samples from the site J21S12 (baked anorthosite) yielded an HTC that is directed SE and down with moderate inclination, broadly similar to that of the dyke. However, the mean directions from the dyke and from the baked host rock seem to be 40 apart (Fig. S6) and their alpha95 confidence cones are not overlapping. Therefore, the directions are not the same and this does not support a positive baked contact test.
Dyke J20S28. This dyke site shows more convincing positive baked contact test than the dyke J20S24. However, the unbaked direction is obtained only from three samples. Please discuss this.
How reliable are the unbaked directions from these sites? There are only three unbaked samples for the dyke J20S28, more for the dyke J20S24. Directions obtained from unbaked samples for these dykes are very different. Please address this.
Minor comments:
Both AF and TH methods were used for cleaning the magnetization and authors choose to use only TH results because there is difference between these. The manuscript is lacking a discussing of possible reasons for this, and it should be added. Moreover, if only TH method is used to define the pole, the criteria #2 of Meert et al (2021) is not fulfilled and the pole scores 0 from this criterion  fulfils 5 of the 7 seven criteria. Not 6 as indicated in the manuscript.
Line 226 (supplementary). Dyke JLQ15? Should it be the dyke J20S24?
Lines 231-232. There is some words missing from the sentence starting: ”The HTCs of the baked samples from sites J20S12, J20S15...”
Lines 243-259: Confusion with component B and component C. Text reads as: “...with dike J20S14, the results suggest that its HTC (component B) should be primary...” “...high-temperature component C from dike J20S14 should be primary...”
"""

sample_paper_one_reviews = """
### Reviewer #1
(Remarks to the Author)
In this work, the authors have extended the time-multiplexed lattices into higher dimensions by incorporating two additional side loops. Based on this construction, the authors have demonstrated several key results: dynamically tuning of the localization of the topological corner states, adiabatic steering of the corner states along predefined trajectories and examination of the robustness of the non-Hermitian skin effect against disorder.
Overall, the manuscript is well-written, and the results are very interesting, showing excellent agreement with the theoretical predictions. I am certainly inclined to recommend the publication of this work. But before I do so, I have some technical questions that I hope the authors can clarify:
1. The extension of ‘two-loop’ architecture into ‘four-loop’ is not new, which has already been theoretically proposed (Science 336, 55-58 (2012)) and experimentally studied (Phys. Rev. Lett. 123, 253903 (2019)) in earlier works. One major difference between the current experimental platform with the previous one is that instead of allocating the loop length equally in the four loops (30 km, 30 km, 30 km + 600m, 30 km + 6m), here the authors have chosen a different way of allocation (3 km, 3 km +100 m, 0 m, 3 m). Is there a practical concern over this length allocation difference?
2. In Fig. S1(a), the authors have just showed 1 PD for the up channel in the entire loop configuration. Should there be another PD for the down channel as well? I am also quite curious about the role of the EDFA directly before the PD.
3. I am particularly surprised with the experimental results in Fig.3. Based on the background color, it seems that the signal to noise ratio decreases from step 1 to step 17, but increases from step 17 to 37, both for the dynamical control and tweezing cases. This is quite counter-intuitive, as I expect the signal to noise ratio to be always decreasing with respect to time in such a non-Hermitian system, as demonstrated in Fig. 2(b) and Fig. S6. If this is the case here, is there any explanation behind this observation?
4. Recently there have been works demonstrating how NHSE could shape the wavefunctions of topological modes in a mechanical system (Nature 608, 50-55 (2022)). It seems that similar SSH lattices can also be realized in the current 2D set- up (changing the 50/50 coupler to a variable beamsplitter). The results would be much richer if you can take this into consideration.

### Reviewer #2
(Remarks to the Author)
This paper experimentally realized a specific type of two-dimensional non-Hermitian tight-binding lattice using pulsed light in time bins in fiber loops and demonstrated a two-dimensional non-Hermitian effect that can be controlled dynamically by slowly varying the lattice Hamiltonian parameters.
While this work presents some new results in (i) the realization of dynamical control of 2D non-Hermitian skin effect and (ii) the realization of 2D non-Hermitian skin effect on this specific experimental platform (synthetic time dimension), the significance of these results is limited to the field of topological photonics. 2D non-Hemermitian skin effect has been realized
  
in [Zhang, X., Tian, Y., Jiang, JH. et al. Observation of higher-order non-Hermitian skin effect. Nat Commun 12, 5377 (2021)]. The dynamical control of the 2D non-Hermitian skin effect does not give rise to any qualitative difference. Therefore, I don't find the significance and potential impact of the paper sufficient to be published in Nature Communications. After adequately addressing the following comments, the paper may be publishable in a more specialized journal.
(1) The authors wrote, "two-dimensional non-Hermitian photonic skin effect, that is, corner states," which I find misleading as corner states usually refer to the boundary state from 2D high-order topological insulator (HOTI). The latter are bound states (discrete energies outside the continuum), while the skin effect, 1D or 2D, comes from bulk states (continuum).
(2) The authors claim that their dynamical variation of parameters for such a non-Hermitian system is adiabatic. However, the adiabatic theorem is based on having no crosstalk between eigenstates. In such a non-Hermitian system with open boundary conditions, the eigenstates are not orthogonal, and thus, the adiabatic condition is not fulfilled, or at least not well defined. It is better to term it as a gradual control.
(3) In the results reported in Fig. 4, I don't find it too meaningful to test the robustness of the skin modes against disorders. As these skin modes are just a localized version of the bulk states, they are not necessarily robust against disorders. Unless the authors can provide or refer to a convincing theoretical argument on the robustness of non-Hermitian skin effects against disorders, this part of the results could be misleading.

### Reviewer #3
(Remarks to the Author)
In the manuscript by Zheng and co-authors, entitled “Dynamic control of 2D non-Hermitian photonic corner states in synthetic dimensions”, the authors experimentally realize a synthetic non-Hermitian lattice using time-multiplexed optical setup. They successfully demonstrate emergence of higher-order non-Hermitian boundary states localized at the corner of four domains with different complex hopping.
Time-multiplexed latices have indeed been shown to be a versatile platform for realization of topological phenomena, and this work is another demonstration of how such optical platform can be used to realize fascinating physical properties, in this case emerging from nontrivial topology in complex energy space. Here the authors predict that in their platform they can attain nontrivial topological characteristic (winding of the complex energy bands), which gives rise to the emergence of corner states.
More importantly, here the authors also show two other interesting results directly stemming from the versatility of the experimental platform 1) the possibility to dynamically control topology of synthetic dimensional domains to move corner modes around and change degree of their localization and 2) to prove topological robustness by introducing disorder into non-Hermitian parameters of the lattice. These are truly impressive results showcasing how powerful the experimental platform is.
The paper is very nicely written, and the supplement aids understanding when the main text appears vague. The results are very clearly presented. Therefore, I have only a couple of minor suggestions which I believe would further improve readability and rigor of the work.
1) In Fig. 1(b) loops in planes corresponding to several fixed values of k_y are shown, however, the plotting style gives an illusion that the loops are not laying in planes. I am afraid this may cause a confusion of a reader. I recommend replotting the loops on planes and stacking these planes vertically to make it clear that they lay in planes of constant k_y.
2) The manuscript would benefit from stating some sort of general argument, a bulk boundary correspondence principle, which would explain how flipping complex hopping in four domains leads to the formation (or even guarantees) emergence of the corner states. While the authors cite earlier works, this could make text more complete and accessible to a novice reader.
3) In Fig. 5 the authors plot average displacements as curves, which is odd. Please use some scatters and also show standard deviations. (This is also Nature Comm policy, I believe).
4) Finally, I suggest softening a very strong claim the authors made in the abstract about time-multiplexed latices, namely “effects free of geometric restrictions”. Time-multiplexed latices have their own geometric constraints, which is evidenced by the limited size of the domain the authors realize. Please rewrite the sentence in the abstract.
To summarize, provided the above minor revision is done, I believe this work will be of a major interest to very broad research community. I am also looking forward to seeing how this experimental platform will evolve to showcase even more exciting non-Abelian and Floquet topological phases.

"""