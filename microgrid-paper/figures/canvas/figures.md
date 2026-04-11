# AI Figure Prompt Pack

This file contains the final three AI-dialog prompts for the revised paper.
Only three structural images are needed:

1. one network-modeling figure for Section 02
2. one SAC-based MDP / DRL workflow figure for Section 03
3. one graphical abstract

## Fixed Visual Direction

- Recommended palette: `Navy + Coral`
- Main color: `#1A3A5C`
- Secondary color: `#E05A47`
- Accent color: `#D4A96A`
- Light background grouping: `#F9F6EE`
- Text color: `#22313F`
- Style: high-density academic infographic, journal-ready, vector-like, clean white background, no photorealism, no 3D rendering, no cartoon styling, no glossy gradients

## Figure A

- Target section: `02-*.tex`
- Figure role: network modeling figure
- Aspect ratio: `16:9`
- Suggested filename: `network_benchmark_overview`

### Purpose

Show the two benchmark networks in one unified academic figure so the reader immediately understands that the paper uses:

- a `CIGRE European LV` community-scale networked microgrid as the storage-value sanity case
- a `modified IEEE33` radial distribution feeder as the main stressed network-constrained case

The figure must make the storage placement, network topology, and engineering role of each benchmark visually obvious.

### Prompt

Create a highly detailed academic paper figure in a clean Elsevier / Applied Energy style, 16:9 landscape layout, combining two benchmark electrical networks in one unified comparison panel. The left half shows the CIGRE European low-voltage network as a community-scale microgrid benchmark with multiple buses, feeder branches, PCC, aggregated rooftop PV, community loads, transformer connection, and a community BESS placed near the feeder tail. The right half shows a modified IEEE 33-bus radial distribution feeder with long radial topology, distributed PV nodes, remote distribution-scale BESS placement, voltage-drop-prone downstream buses, and one or two visually highlighted stressed branches where local congestion and voltage support matter. The center top should include a compact title bar reading “Pandapower-based benchmark networks for network-constrained battery scheduling”. Each half should have a small-caps section title: “CIGRE European LV: storage-value sanity case” and “Modified IEEE33: stressed main case”. Add concise labels such as PCC, transformer, feeder tail BESS, aggregated PV, community load, radial feeder, remote BESS 1 MWh / 0.5 MW, voltage-drop region, stressed branch, and network support. Use white module boxes, thin navy borders, coral only for stressed lines or constrained nodes, warm sand for battery assets, subtle light background grouping, grey arrows for power-flow direction, and precise vector-diagram styling. The composition should feel information-dense, professional, and publication-ready, with no decorative clutter, no perspective effects, and no photorealistic components.

### Negative guidance

Do not generate a photorealistic power grid, do not use dark mode, do not use more than three accent colors, do not draw random city icons, and do not turn the diagram into a marketing poster.

## Figure B

- Target section: `03-*.tex`
- Figure role: MDP / SAC algorithm figure
- Aspect ratio: `16:9`
- Suggested filename: `sac_mdp_network_constrained_workflow`

### Purpose

Show that the paper is not using a generic MDP picture, but a specific SAC-based network-constrained battery scheduling loop:

- observation
- SAC actor-critic decision
- battery set-point
- constraint execution
- battery update under selected fidelity branch
- pandapower network settlement
- reward and next state feedback

It should also show the three-level fidelity ladder:

- `simple`
- `loss-aware Thevenin`
- `full Thevenin`

### Prompt

Generate a high-density academic framework diagram for a journal paper, 16:9 landscape layout, illustrating a SAC-based network-constrained battery scheduling workflow. The figure should read clearly from left to right. Start with an observation block containing time features, PV forecast or PV realization, load, electricity price, SOC, and net-demand information. Feed this into a central SAC agent module that explicitly contains actor network, critic networks, replay buffer, and policy update loop. From the SAC module, draw an action arrow labeled “battery power set-point”. Route that action into a constraint-execution block that clips or reshapes infeasible battery commands under SOC, power, and network-related limits. Then pass the executed action into a battery-transition block with an inset fidelity ladder showing three branches: simple, loss-aware Thevenin, and full Thevenin. After battery update, route the system state into a pandapower AC power-flow settlement block that outputs bus voltages, feeder loading, and grid import. Then create a reward block showing operating cost, network penalties, and SOC-shaping contributions, and feed both reward and next state back into the SAC learning loop. Use a clean white background, white rounded rectangles, navy borders for core learning modules, coral emphasis for constraint execution and network settlement, warm sand accents for battery-related modules, thin grey arrows, compact mathematical labels, and visually balanced spacing. The figure should look like a serious scientific workflow diagram rather than a generic AI illustration.

### Negative guidance

Do not draw humanoid robots, do not use neon AI aesthetics, do not use oversized neural-network icons, and do not omit the battery fidelity ladder or pandapower settlement block.

## Figure C

- Target use: graphical abstract
- Figure role: paper-wide one-panel summary
- Aspect ratio: approximately `5:2`
- Suggested filename: `graphical_abstract_reviewer_aligned`

### Purpose

Summarize the full reviewer-aligned story in one horizontal panel:

- two benchmark networks
- storage-value sanity
- fidelity ladder
- non-monotonic fidelity value
- mixed-fidelity narrow rescue window

### Prompt

Create a publication-grade graphical abstract for an energy journal paper, in a wide 5:2 horizontal composition. The left portion should show two miniature benchmark-network icons: one for the CIGRE European LV community-scale case and one for the modified IEEE33 stressed radial feeder. The middle portion should present a compact battery-model fidelity ladder with three levels: simple, loss-aware Thevenin, and full Thevenin, arranged from low to high fidelity. The right portion should summarize the paper’s findings visually rather than with long text: a small storage-vs-no-storage sanity motif showing that storage improves system performance, a small heatmap motif showing that the intermediate loss-aware pathway is best across deployment environments, and a small trade-off motif showing that mixed-fidelity helps but only within a narrow tuning window. Add only very short labels such as “storage matters”, “fidelity value is non-monotonic”, and “mixed-fidelity is a narrow stabilizer”. Use a crisp vector style, white background, deep navy structure, coral highlights for stressed or high-cost outcomes, warm sand highlights for storage assets, subtle grey connectors, and a polished Elsevier-style academic infographic feel. The whole image should be visually compact, legible after size reduction, and clearly focused on engineering insight instead of decorative elements.

### Negative guidance

Do not make it poster-like, do not use large paragraphs of text, do not add unrelated icons, do not use photorealistic batteries, and do not make the right-side results look like generic business charts.

## Usage Notes

- These prompts are intended for conversational image models such as ChatGPT image mode or Gemini image generation.
- If the generated figure is too decorative, ask for:
  - more vector style
  - less color fill
  - more white space
  - smaller labels
  - stronger journal infographic style
- If the generated figure is too empty, ask for:
  - denser annotations
  - more explicit labels on buses, storage, and workflow blocks
  - clearer engineering constraints

## Additional Top-Journal Case-Study Figures

The following prompts are optional but useful when the paper needs more visual support
in the case-study section or in Appendix B. These are not the three mandatory structural
figures above. They are extra top-journal-grade images that can strengthen reviewer
perception of mechanism, experimental design, and engineering insight.
