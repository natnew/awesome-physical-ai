# Awesome Physical AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![License: MIT](https://img.shields.io/github/license/natnew/awesome-physical-ai?color=blue)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Last Commit](https://img.shields.io/github/last-commit/natnew/awesome-physical-ai)](https://github.com/natnew/awesome-physical-ai/commits/main)
[![Contributors](https://img.shields.io/github/contributors/natnew/awesome-physical-ai)](https://github.com/natnew/awesome-physical-ai/graphs/contributors)
[![GitHub stars](https://img.shields.io/github/stars/natnew/awesome-physical-ai?style=social)](https://github.com/natnew/awesome-physical-ai/stargazers)

> A curated list of **awesome Physical AI** resources — a Physical AI roadmap covering robotics, embodied AI, simulation, world models, and production-grade Physical AI systems.

Physical AI and embodied AI represent the convergence of large-scale machine learning with robotics — enabling embodied agents that perceive, reason, and act in the physical world. This list is a curated, engineering-oriented map of **Physical AI resources**: robotics resources, robot learning, robotics foundation models, vision-language-action models (VLA models), world models, robotics simulation, sim-to-real techniques, simulation environments, Physical AI benchmarks, robotics datasets, robotics benchmarks, foundation models for robotics, generalist robot policies, and patterns for safe, production-grade Physical AI systems.

It is built for researchers and practitioners at every stage — from foundational learning to deployment of safe embodied AI systems — and for technical leaders evaluating how embodied intelligence will affect products, operations, and infrastructure.

## Contents

**Canonical categories**

- [Simulators](#simulators)
- [Datasets](#datasets)
- [Benchmarks](#benchmarks)
- [Evaluation Methodology](#evaluation-methodology)
- [Robotics Foundation Models](#robotics-foundation-models)
- [World Models](#world-models)
- [Manipulation](#manipulation)
- [Locomotion](#locomotion)
- [Sim-to-Real](#sim-to-real)
- [Safety & Robustness](#safety--robustness)
- [Governance & Policy](#governance--policy)
- [Production Patterns / Reference Architectures](#production-patterns--reference-architectures)
- [Courses](#courses)
- [Companies](#companies)

**Appendices**

- [Books](#books)
- [Tutorials & Guides](#tutorials--guides)
- [Key Papers](#key-papers)
- [Survey Papers](#survey-papers)
- [Hardware Platforms](#hardware-platforms)
- [Conferences](#conferences)
- [Community](#community)
- [Newsletters & Blogs](#newsletters--blogs)
- [People to Follow](#people-to-follow)
- [Getting Started Projects](#getting-started-projects)
- [Related Awesome Lists](#related-awesome-lists)
- [Contributing](#contributing)

---

## Simulators

Physics engines and high-fidelity simulation environments for robotics and embodied AI.

- [MuJoCo](https://mujoco.org/) — Multi-joint dynamics with contact; fast, accurate physics widely used for RL research.
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) — GPU-accelerated robotics simulator on Omniverse with photorealistic rendering and PhysX.
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) — Unified robot learning framework on Isaac Sim for RL, imitation learning, and motion planning.
- [Drake](https://drake.mit.edu/) — Model-based design toolbox from TRI/MIT for planning, control, and rigorous dynamics analysis.
- [Gazebo](https://gazebosim.org/) — Open-source robotics simulator with mature ROS integration and broad sensor support.
- [PyBullet](https://github.com/bulletphysics/bullet3) — Open-source physics engine (Bullet) with Python bindings, popular for prototyping and RL.
- [Habitat](https://aihabitat.org/) — Embodied AI platform optimised for high-throughput 3D navigation and instruction-following research.
- [SAPIEN](https://sapien.ucsd.edu/) — Physics-rich simulator with the PartNet-Mobility articulated object dataset.
- [Genesis](https://genesis-embodied-ai.github.io/) — Universal differentiable simulator for robotics and embodied AI with cross-platform physics solvers.

## Datasets

Large-scale teleoperation, demonstration, and interaction datasets used to train robot policies.

- [Open X-Embodiment](https://robotics-transformer-x.github.io/) — 1M+ trajectories across 22 embodiments; the de facto cross-embodiment training corpus.
- [DROID](https://droid-dataset.github.io/) — Large-scale, in-the-wild manipulation dataset collected across 13 institutions.
- [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) — Diverse manipulation behaviours designed to support broad generalisation.
- [RH20T](https://rh20t.github.io/) — Robot manipulation dataset with paired human demonstrations for one-shot learning research.
- [RoboMIND](https://x-humanoid-robomind.github.io/) — Multimodal bimanual mobile manipulation dataset with 310K+ trajectories.
- [AgiBot World](https://github.com/OpenDriveLab/AgiBot-World) — Large-scale dataset designed to train and evaluate robot foundation models.
- [Ego4D](https://ego4d-data.org/) — Massive-scale egocentric video dataset useful for pretraining perception and world models.
- [Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something) — Action recognition dataset frequently used to pretrain manipulation perception.

## Benchmarks

Task suites and standardised evaluations for manipulation, locomotion, and embodied reasoning.

- [LIBERO](https://libero-project.github.io/) — Lifelong robot learning benchmark with 130 diverse manipulation tasks.
- [RLBench](https://sites.google.com/view/rlbench) — Vision-guided manipulation benchmark covering 100+ tasks in CoppeliaSim.
- [MetaWorld](https://meta-world.github.io/) — Meta-RL benchmark with 50 manipulation tasks for multi-task and transfer studies.
- [CALVIN](https://github.com/mees/calvin) — Benchmark for long-horizon, language-conditioned manipulation.
- [HumanoidBench](https://humanoid-bench.github.io/) — Simulated humanoid benchmark for whole-body control across locomotion and manipulation.
- [FurnitureBench](https://clvrai.github.io/furniture-bench/) — Real-world long-horizon furniture assembly benchmark.
- [ARNOLD](https://arnold-benchmark.github.io/) — Language-grounded continuous-task benchmark in physically realistic scenes.
- [Colosseum](https://robot-colosseum.github.io/) — Generalisation benchmark perturbing 14 axes of variation for manipulation.
- [OpenEQA](https://open-eqa.github.io/) — Embodied question-answering benchmark over scanned real environments.

## Evaluation Methodology

Harnesses, metrics, and methodology for measuring robot policy performance, robustness, and sim-to-real validity.

- [RoboArena](https://robo-arena.github.io/) — Decentralised real-world evaluation protocol for generalist robot policies.
- [robomimic](https://robomimic.github.io/) — Standardised offline-RL and imitation-learning evaluation pipeline with reproducible baselines.
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) — Closed-loop evaluation protocol for end-to-end driving policies.
- [Eval-vs-Train Mismatch (Kumar et al.)](https://arxiv.org/abs/2306.13085) — Methodology paper on why offline metrics mispredict deployed robot performance.
- [SimplerEnv](https://simpler-env.github.io/) — Aligned simulator-based evaluation that correlates with real-robot performance for VLAs.
- [Statistical Reliability of RL Evaluations](https://agarwl.github.io/rliable/) — `rliable` library and methodology for confidence intervals on RL benchmarks.

## Robotics Foundation Models

Generalist policies and vision-language-action (VLA) models for robotic control.

- [π0 (Physical Intelligence)](https://www.physicalintelligence.company/) — Generalist policy combining multi-robot data with flow matching for dexterous manipulation.
- [Octo](https://octo-models.github.io/) — Open-source generalist robot policy trained on Open X-Embodiment with cross-embodiment fine-tuning.
- [OpenVLA](https://openvla.github.io/) — Open-source 7B-parameter vision-language-action model built on Prismatic VLMs.
- [RT-2](https://robotics-transformer2.github.io/) — Vision-language-action model that transfers web knowledge to robotic control.
- [RT-X](https://robotics-transformer-x.github.io/) — Cross-embodiment models demonstrating positive transfer across robot platforms.
- [Gemini Robotics](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) — Google DeepMind VLA family with embodied reasoning capabilities.
- [GR00T N1 (NVIDIA)](https://developer.nvidia.com/isaac/gr00t) — Open humanoid foundation model with a dual-system slow/fast architecture.
- [Helix (Figure)](https://www.figure.ai/) — Vision-language-action model targeting generalist humanoid control.

## World Models

Generative and predictive models of physical dynamics used for planning, simulation, and pretraining.

- [V-JEPA 2 (Meta FAIR)](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) — Self-supervised video world model trained on 1M+ hours enabling zero-shot robot planning.
- [NVIDIA Cosmos](https://developer.nvidia.com/cosmos) — World foundation models for physically-grounded synthetic data generation.
- [Genie 2 (DeepMind)](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) — Foundation world model that generates interactive, controllable 3D environments.
- [DreamerV3](https://danijar.com/project/dreamerv3/) — General world-model algorithm achieving strong results across 150+ tasks with fixed hyperparameters.
- [DayDreamer](https://danijar.com/project/daydreamer/) — World models applied to physical robot learning for sample-efficient skill acquisition.
- [UniSim](https://universal-simulator.github.io/unisim/) — Universal simulator learning real-world interactions from diverse video data.
- [I-JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/) — Image joint-embedding predictive architecture; foundational to the JEPA world-model line.

## Manipulation

Methods, models, and tools for grasping, dexterous manipulation, and contact-rich tasks.

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) — Visuomotor policy learning via action diffusion; widely-used baseline for imitation.
- [ACT (Action Chunking Transformers)](https://tonyzhaozh.github.io/aloha/) — Transformer policy for bimanual fine manipulation from demonstrations.
- [Mobile ALOHA](https://mobile-aloha.github.io/) — Bimanual mobile manipulation system with low-cost teleoperation hardware.
- [ALOHA Unleashed](https://aloha-unleashed.github.io/) — Recipe for scaling robot dexterity via large-scale imitation learning.
- [Dex-Net](https://berkeleyautomation.github.io/dex-net/) — Datasets and models for analytic and learned robust grasping.
- [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) — Grasp pose generation directly from partial point clouds.
- [RoboCasa](https://robocasa.ai/) — Large-scale household-scene simulation suite for training generalist manipulation policies.
- [MIT 6.4210 — Robotic Manipulation](https://manipulation.csail.mit.edu/) — Russ Tedrake's reference text covering perception, planning, and control for manipulation.

## Locomotion

Legged, bipedal, and humanoid locomotion — controllers, learning approaches, and reference platforms.

- [Learning to Walk in Minutes (ETH/RSL)](https://leggedrobotics.github.io/legged_gym/) — Massively-parallel RL pipeline that trains quadruped locomotion policies from scratch in simulation.
- [RMA — Rapid Motor Adaptation](https://ashish-kmr.github.io/rma-legged-robots/) — Online adaptation method for robust real-world quadruped locomotion.
- [ANYmal Parkour (RSL)](https://www.anybotics.com/robotics/anymal/) — Perceptive locomotion enabling agile traversal of complex terrain.
- [Cassie Bipedal Locomotion](https://arxiv.org/abs/2105.08328) — Sim-to-real RL controllers for blind bipedal locomotion on Cassie.
- [HumanPlus](https://humanoid-ai.github.io/) — Humanoid shadowing of human motion for whole-body control via teleoperation and RL.
- [OmniH2O](https://omni.human2humanoid.com/) — Universal whole-body teleoperation and learning for humanoids.
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) — Fast PPO implementation from ETH Zurich tuned for legged-robot RL on GPU simulators.
- [legged_gym](https://github.com/leggedrobotics/legged_gym) — Reference Isaac Gym/Isaac Lab environments for legged locomotion research.

## Sim-to-Real

Techniques and case studies for transferring policies trained in simulation to real hardware.

- [Domain Randomization (Tobin et al.)](https://arxiv.org/abs/1703.06907) — Foundational paper on randomising simulator parameters for zero-shot transfer.
- [Learning Dexterous In-Hand Manipulation (OpenAI)](https://arxiv.org/abs/1808.00177) — Sim-to-real dexterous manipulation via automatic domain randomization.
- [Sim-to-Real via Sim-to-Sim (Koos et al. line)](https://arxiv.org/abs/1812.07252) — Bridging the reality gap with progressively more realistic intermediate simulators.
- [Eureka (NVIDIA)](https://eureka-research.github.io/) — LLM-driven reward design that enables sim-to-real transfer for dexterous skills.
- [DextrAH-G](https://sites.google.com/view/dextrah-g) — Sim-to-real dexterous arm-hand grasping pipeline using GPU-parallel RL.
- [Real-World Humanoid Locomotion (Radosavovic et al.)](https://arxiv.org/abs/2303.03381) — Sim-to-real RL for humanoid walking with strong robustness.
- [DayDreamer](https://danijar.com/project/daydreamer/) — World-model approach that reduces real-robot interaction needed for transfer.

## Safety & Robustness

Tools, benchmarks, and methodology for safe exploration, robustness testing, and failure-mode analysis.

- [Safety Gym (OpenAI)](https://github.com/openai/safety-gym) — Environments for benchmarking constrained and safe-exploration RL.
- [Safe Control Gym](https://github.com/utiasDSL/safe-control-gym) — Benchmark suite for safe learning-based control with constraints and disturbances.
- [Constrained Policy Optimization (Achiam et al.)](https://arxiv.org/abs/1705.10528) — Canonical algorithmic framework for constrained safe RL.
- [Realistic Adversarial Driving (Wang et al.)](https://arxiv.org/abs/2003.01197) — Methodology for stress-testing autonomous driving policies under adversarial conditions.
- [Robot Trust & Safety (Stanford CRFM)](https://crfm.stanford.edu/) — Foundation-model centre research including robotic safety, evaluation, and failure modes.
- [CARLA Leaderboard — Safety Scenarios](https://leaderboard.carla.org/) — Standardised stress-test scenarios for driving-policy robustness.
- [Verifiable Reinforcement Learning (DeepMind)](https://arxiv.org/abs/2308.13247) — Survey-style work on formal verification approaches for RL controllers.

## Governance & Policy

Standards, frameworks, and policy guidance relevant to deploying Physical AI systems.

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) — Voluntary framework for managing risks across the AI lifecycle, applicable to robotics.
- [EU AI Act](https://artificialintelligenceact.eu/) — Regulation establishing risk-tiered obligations for AI systems sold or operated in the EU.
- [ISO 10218 / ISO/TS 15066](https://www.iso.org/standard/73934.html) — Industrial robot and collaborative-robot safety standards underpinning workplace deployment.
- [IEEE 7000 Series](https://standards.ieee.org/initiatives/autonomous-intelligence-systems/standards/) — Standards on ethical and value-based design for autonomous and intelligent systems.
- [OECD AI Principles](https://oecd.ai/en/ai-principles) — Intergovernmental principles guiding trustworthy AI deployment, including embodied systems.
- [UK AI Safety Institute](https://www.aisi.gov.uk/) — Government body publishing evaluations and guidance on frontier AI risks.
- [White House Executive Order on AI (14110)](https://www.federalregister.gov/documents/2023/11/01/2023-24283/safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence) — US federal directive on safe and trustworthy AI development relevant to robotics deployers.

## Production Patterns / Reference Architectures

Middleware, runtime stacks, and reference patterns for shipping robots in production.

- [ROS 2](https://docs.ros.org/) — De facto middleware for production robot software, with QoS, security, and real-time profiles.
- [NVIDIA Isaac ROS](https://developer.nvidia.com/isaac-ros) — GPU-accelerated ROS 2 perception and navigation packages for production robots.
- [MoveIt 2](https://moveit.ros.org/) — Production-grade motion planning framework integrated with ROS 2.
- [Nav2](https://docs.nav2.org/) — ROS 2 navigation stack with behaviour trees, planners, and recovery patterns.
- [Open-RMF](https://www.open-rmf.org/) — Open-source framework for multi-robot, multi-vendor fleet orchestration.
- [DDS Security (OMG Spec)](https://www.omg.org/spec/DDS-SECURITY/) — Reference standard for authenticated, encrypted robot data buses (used by ROS 2).
- [Foxglove](https://foxglove.dev/) — Observability and visualisation stack for robotics telemetry, logs, and replay.

## Courses

University courses and structured learning programs in robot learning and embodied AI.

- [CS 336 — Robot Learning (Stanford)](https://cs336.stanford.edu/) — Modern robot learning, covering imitation and reinforcement learning.
- [CS 224R — Deep RL for Robotics (Stanford)](http://cs224r.stanford.edu/) — Deep reinforcement learning for real-world robots.
- [CS 287 — Advanced Robotics (Berkeley)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/) — Planning, learning, and control for robotics.
- [16-831 — Introduction to Robot Learning (CMU)](https://16-831-s24.github.io/) — CMU's foundations of robot learning.
- [MIT 6.4210 — Robotic Manipulation](https://manipulation.csail.mit.edu/) — Perception, planning, and control for manipulation by Russ Tedrake.
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/) — Educational resource covering deep RL fundamentals with reference implementations.
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/) — Free hands-on course on deep reinforcement learning.
- [NVIDIA DLI Robotics](https://www.nvidia.com/en-us/training/) — Self-paced courses on Isaac Sim, ROS, and robot learning.

## Companies

Organisations advancing Physical AI through foundation models, humanoids, and applied robotics.

- [Physical Intelligence (π)](https://www.physicalintelligence.company/) — Foundation models for general-purpose robots; developer of π0.
- [Figure](https://www.figure.ai/) — Humanoid robotics company building general-purpose bipedal platforms with VLA-driven control.
- [1X Technologies](https://www.1x.tech/) — Humanoid robots designed for safe human interaction and home environments.
- [Boston Dynamics](https://bostondynamics.com/) — Pioneers of dynamic legged and humanoid platforms (Spot, Atlas) with active research arm.
- [Agility Robotics](https://agilityrobotics.com/) — Maker of Digit, a bipedal logistics robot deployed in commercial warehouses.
- [Apptronik](https://apptronik.com/) — Humanoid robotics company building Apollo for industrial applications.
- [Skild AI](https://www.skild.ai/) — Building scalable, generalist robot intelligence trained across diverse embodiments.
- [Covariant](https://covariant.ai/) — Foundation-model-driven AI for robotic picking and warehouse manipulation.
- [Wayve](https://wayve.ai/) — Embodied AI for end-to-end autonomous driving using world models and VLAs.
- [Pollen Robotics (Hugging Face)](https://www.pollen-robotics.com/) — Open-source humanoid robotics; maker of Reachy 2 and Reachy Mini.

---

# Appendices

The sections below complement the canonical taxonomy with related learning resources, hardware, community pointers, and practitioner recommendations.

## Books

Foundational and advanced textbooks.

- [Probabilistic Robotics](https://mitpress.mit.edu/9780262201629/) - Thrun, Burgard, Fox. Essential text on probabilistic methods for robotics.
- [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) - Lynch & Park. Mechanics, planning, and control with free online version.
- [Robotics, Vision and Control](https://petercorke.com/rvc/) - Corke. MATLAB/Python-based introduction to robotics fundamentals.
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) - Sutton & Barto. The classic RL textbook, freely available online.
- [Planning Algorithms](http://lavalle.pl/planning/) - LaValle. Comprehensive coverage of motion planning, free online.
- [A Mathematical Introduction to Robotic Manipulation](https://www.cds.caltech.edu/~murray/mlswiki/) - Murray, Li, Sastry. Free online textbook on manipulation.
- [Introduction to Autonomous Robots](https://introduction-to-autonomous-robots.github.io/) - Correll et al. Open-source robotics textbook.

## Tutorials & Guides

Hands-on learning resources.

- [LeRobot Tutorials](https://github.com/huggingface/lerobot) - Getting started with robot learning using Hugging Face's framework.
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/) - Comprehensive guides for NVIDIA's robot learning framework.
- [MuJoCo Documentation](https://mujoco.readthedocs.io/) - Official docs with modeling and programming guides.
- [ROS 2 Tutorials](https://docs.ros.org/en/rolling/Tutorials.html) - Official tutorials for getting started with ROS 2.
- [PyBullet Quickstart](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/) - Getting started with PyBullet simulation.
- [Open X-Embodiment Tutorial](https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb) - Working with the largest robotics dataset.
- [Diffusion Policy Tutorial](https://diffusion-policy.cs.columbia.edu/) - Implementation guide for diffusion-based robot policies.

## Key Papers

Influential research papers in Physical AI.

- [V-JEPA 2](https://arxiv.org/abs/2506.09985) - LeCun et al. Self-supervised video world model enabling zero-shot robot control from Meta FAIR.
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864) - Cross-embodiment robot learning datasets and RT-X models from Google DeepMind.
- [DROID](https://arxiv.org/abs/2403.12945) - Large-scale in-the-wild robot manipulation dataset.
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) - Visuomotor policy learning via action diffusion.
- [RT-2](https://arxiv.org/abs/2307.15818) - Vision-language-action models transfer web knowledge to robots.
- [Octo](https://arxiv.org/abs/2405.12213) - Open-source generalist robot policy.
- [Mobile ALOHA](https://arxiv.org/abs/2401.02117) - Learning bimanual mobile manipulation with low-cost hardware.
- [ALOHA Unleashed](https://aloha-unleashed.github.io/) - Simple recipe for robot dexterity at scale.
- [I-JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/) - LeCun's image joint-embedding predictive architecture.

## Survey Papers

Comprehensive overviews of key areas.

- [Foundation Models in Robotics](https://arxiv.org/abs/2312.07843) - Survey on how foundation models are transforming robotics.
- [Neural Fields in Robotics](https://arxiv.org/abs/2410.20220) - Survey on neural implicit representations for robotics applications.
- [A Survey on LLM-based Autonomous Agents](https://arxiv.org/abs/2308.11432) - Comprehensive survey on LLM-based agents.
- [Robot Learning Survey](https://arxiv.org/abs/2312.08591) - Overview of robot learning from demonstration.
- [3D Gaussian Splatting in Robotics](https://arxiv.org/abs/2410.12262) - Survey on gaussian splatting applications in robotics.
- [World Models Survey](https://arxiv.org/abs/2403.02622) - Survey on world models for autonomous systems.

## Hardware Platforms

Physical robots for research and development.

**Arms & Manipulators**
- [Franka Emika](https://www.franka.de/) - Research-grade 7-DOF arm with torque sensing and compliant control.
- [Universal Robots](https://www.universal-robots.com/) - Collaborative robot arms widely used in research and industry.
- [Kinova](https://www.kinovarobotics.com/) - Lightweight assistive and research robot arms.
- [xArm](https://www.ufactory.cc/) - Affordable 6/7-DOF robot arms for research and development.
- [Kuka iiwa](https://www.kuka.com/) - Industrial arm designed for human collaboration.

**Humanoids**
- [Reachy 2 (Pollen Robotics / Hugging Face)](https://www.pollen-robotics.com/reachy/) - Fully open-source humanoid robot for embodied AI research with Python SDK and ROS 2 support.
- [Reachy Mini](https://huggingface.co/blog/reachy-mini) - Compact open-source desktop robot from $299 for AI experimentation.
- [Boston Dynamics Atlas](https://bostondynamics.com/atlas/) - Advanced research humanoid with dynamic locomotion.
- [Tesla Optimus](https://www.tesla.com/optimus) - Humanoid designed for manufacturing and consumer applications.
- [Unitree H1/G1](https://www.unitree.com/) - Affordable humanoid platforms for research.
- [Agility Robotics Digit](https://agilityrobotics.com/) - Bipedal robot for logistics.
- [Fourier Intelligence GR-1](https://www.fftai.com/) - Open-platform humanoid robot.

**Mobile Robots**
- [Boston Dynamics Spot](https://bostondynamics.com/spot/) - Quadruped with manipulation capabilities.
- [Unitree Go2/B2](https://www.unitree.com/) - Affordable quadruped robots for research.
- [Clearpath Robotics](https://clearpathrobotics.com/) - Research-grade mobile platforms including Husky and Jackal.
- [Hello Robot Stretch](https://hello-robot.com/) - Affordable mobile manipulator for home assistance research.
- [PAL Robotics TIAGo](https://pal-robotics.com/) - Mobile manipulator for service robotics.

**Low-Cost & DIY**
- [ALOHA Hardware](https://tonyzhaozh.github.io/aloha/) - Low-cost bimanual teleoperation system (~$20k).
- [Gello](https://wuphilipp.github.io/gello/) - General, low-cost, and intuitive teleoperation framework.
- [Open Dynamic Robot Initiative](https://open-dynamic-robot-initiative.github.io/) - Open-source modular robot for legged locomotion research.
- [Stanford Pupper](https://stanfordstudentrobotics.org/pupper) - Open-source quadruped robot kit.
- [Open Manipulator](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/overview/) - Affordable open-source robot arm from Robotis.
- [LeRobot Hardware](https://github.com/huggingface/lerobot) - Reference designs for low-cost robot arms.
- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) - Open-source anthropomorphic robot arm.

## Conferences

Major venues for robotics and AI research.

- [CoRL](https://www.corl.org/) - Conference on Robot Learning.
- [RSS](https://roboticsconference.org/) - Robotics: Science and Systems.
- [ICRA](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra) - IEEE International Conference on Robotics and Automation.
- [IROS](https://www.ieee-ras.org/conferences-workshops/financially-co-sponsored/iros) - IEEE/RSJ International Conference on Intelligent Robots and Systems.
- [NeurIPS](https://neurips.cc/) - Conference on Neural Information Processing Systems.
- [ICML](https://icml.cc/) - International Conference on Machine Learning.
- [ICLR](https://iclr.cc/) - International Conference on Learning Representations.
- [HRI](https://humanrobotinteraction.org/) - ACM/IEEE International Conference on Human-Robot Interaction.
- [Humanoids](https://www.ieee-ras.org/conferences-workshops/financially-co-sponsored/humanoids) - IEEE-RAS International Conference on Humanoid Robots.

## Community

Forums, discussions, and meetups.

- [ROS Discourse](https://discourse.ros.org/) - Official ROS community discussion forum.
- [Robotics Stack Exchange](https://robotics.stackexchange.com/) - Q&A for robotics professionals and enthusiasts.
- [r/robotics](https://www.reddit.com/r/robotics/) - Reddit community for robotics discussion.
- [Hugging Face Discord](https://huggingface.co/join/discord) - Community discussions including LeRobot and Reachy.
- [Pollen Robotics Discord](https://discord.gg/pollen-robotics) - Community for Reachy and open-source robotics.
- [Robot Learning Discord](https://discord.gg/robotlearning) - Community for robot learning researchers.

## Newsletters & Blogs

Stay updated with the latest developments.

**Deep Dives & Analysis**
- [Chipstrat](https://www.chipstrat.com/) - Austin Lyons. Semiconductors, AI, and robotics strategy. Excellent "Robots That See" series on computer vision for robotics.
- [Robots & Startups](https://robotsandstartups.substack.com/) - Andra Keay (Silicon Valley Robotics). Robot startups and industry trends from the epicenter of the robot revolution.
- [Import AI](https://importai.substack.com/) - Jack Clark. Weekly analysis of AI breakthroughs, policy, and implications for robotics.
- [Interconnects](https://www.interconnects.ai/) - Nathan Lambert. Technical deep dives on AI from an actual model trainer.
- [Ahead of AI](https://magazine.sebastianraschka.com/) - Sebastian Raschka. Research-focused ML/AI coverage.
- [The Batch](https://www.deeplearning.ai/the-batch/) - Andrew Ng's weekly AI newsletter with educational focus.

**Industry News**
- [The Robot Report](https://www.therobotreport.com/) - News and analysis on robotics industry.
- [IEEE Spectrum Robotics](https://spectrum.ieee.org/topic/robotics/) - IEEE's robotics coverage and technical features.
- [Robotics 24/7](https://www.robotics247.com/) - Industry news and research updates.

**Research & Company Blogs**
- [Hugging Face Blog](https://huggingface.co/blog) - Updates on LeRobot, Reachy, and open-source robotics.
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/category/robotics/) - Isaac, Cosmos, and robotics AI updates.
- [Meta AI Blog](https://ai.meta.com/blog/) - V-JEPA, research updates from Yann LeCun's team.
- [Boston Dynamics Blog](https://bostondynamics.com/blog/) - Advanced locomotion and manipulation research.
- [Google DeepMind Blog](https://deepmind.google/blog/) - Gemini Robotics, RT-2, and embodied AI updates.

## People to Follow

Researchers, engineers, and practitioners shaping Physical AI.

**Research Leaders**
- [Yann LeCun](https://x.com/ylecun) - Chief AI Scientist at Meta. V-JEPA, world models, and self-supervised learning.
- [Fei-Fei Li](https://x.com/drfeifei) - Stanford HAI, World Labs. Computer vision and spatial intelligence pioneer.
- [Pieter Abbeel](https://x.com/paboratory) - Berkeley BAIR, Covariant. Robot learning and RL.
- [Sergey Levine](https://x.com/svlevine) - Berkeley. Reinforcement learning and robot learning.
- [Chelsea Finn](https://x.com/chelaboratory) - Stanford. Meta-learning and robot learning.
- [Russ Tedrake](https://x.com/russtedrake) - MIT, Toyota Research Institute. Manipulation and control.
- [Dieter Fox](https://x.com/dieterfox) - NVIDIA, UW. Perception and robot learning.

**Robotics & Hardware**
- [Kate Darling](https://x.com/gaboratory) - MIT Media Lab, BD AI Institute. Robotics ethics and human-robot interaction.
- [Rodney Brooks](https://x.com/rodneyabrooks) - iRobot co-founder, Robust.AI. Robotics pioneer and blogger.
- [Angelica Lim](https://x.com/petitegeek) - SFU. Social robotics and emotional AI.
- [Andra Keay](https://x.com/robohub) - Silicon Valley Robotics. Robot ecosystem and startups.

**Industry Leaders**
- [Brett Adcock](https://x.com/adcock_brett) - Figure CEO. Humanoid robotics at scale.
- [Austin Lyons](https://x.com/austinlyons) - Chipstrat. Semiconductor and robotics strategy.
- [Soumith Chintala](https://x.com/soumithchintala) - PyTorch co-founder, Meta. Open-source AI.
- [Andrej Karpathy](https://x.com/karpathy) - Ex-Tesla AI, educator. Neural networks and autonomous systems.

## Getting Started Projects

Hands-on projects to progress from beginner to mastery. All runnable on laptop or mobile.

### Beginner: Foundations (Week 1-2)

Start here to build intuition for robot learning concepts.

- **Gymnasium Cart-Pole** - Classic control problem. Balance a pole using RL in 50 lines of Python. [Tutorial](https://gymnasium.farama.org/tutorials/training_agents/cart_pole/)
- **MuJoCo Playground** - Explore physics simulation in the browser. No install required. [Playground](https://mujoco.github.io/mujoco_wasm/)
- **Hugging Face Deep RL Course** - Free, hands-on RL fundamentals with Colab notebooks. [Course](https://huggingface.co/learn/deep-rl-course/)
- **Spinning Up in Deep RL** - OpenAI's educational resource. Well-documented implementations. [Guide](https://spinningup.openai.com/)

### Intermediate: Simulation & Imitation (Week 3-6)

Build robot policies in simulation environments.

- **LeRobot Tutorials** - Train manipulation policies using Hugging Face's framework. Includes pre-trained models. [Repo](https://github.com/huggingface/lerobot)
- **Diffusion Policy Colab** - Implement diffusion-based robot policies step by step. [Colab](https://colab.research.google.com/github/real-stanford/diffusion_policy/blob/main/diffusion_policy.ipynb)
- **MuJoCo Humanoid Control** - Train a humanoid to walk using PPO. [MuJoCo Docs](https://mujoco.readthedocs.io/)
- **Isaac Lab Getting Started** - GPU-accelerated robot learning. Requires NVIDIA GPU. [Docs](https://isaac-sim.github.io/IsaacLab/)
- **RoboMimic** - Framework for robot learning from demonstrations. [Repo](https://github.com/ARISE-Initiative/robomimic)

### Advanced: Foundation Models & Transfer (Week 7-12)

Work with state-of-the-art models and real-world transfer.

- **OpenVLA Fine-tuning** - Fine-tune a 7B vision-language-action model on custom tasks. [Guide](https://openvla.github.io/)
- **Octo Custom Dataset** - Train generalist policies on Open X-Embodiment data. [Repo](https://octo-models.github.io/)
- **Open X-Embodiment Colab** - Explore 1M+ trajectories across 22 robot embodiments. [Colab](https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb)
- **RT-X Evaluation** - Evaluate cross-embodiment transfer with RT-X models. [Paper](https://robotics-transformer-x.github.io/)

### Hardware Projects (Optional)

For those ready to touch real robots.

- **SO-100 Arm ($110)** - LeRobot-compatible low-cost robot arm. [Build Guide](https://github.com/TheRobotStudio/SO-ARM100)
- **ALOHA ($20K)** - Bimanual teleoperation platform with extensive documentation. [Repo](https://github.com/tonyzhaozh/aloha)
- **Reachy Mini ($299)** - Desktop humanoid from Pollen Robotics with Python SDK. [Shop](https://www.pollen-robotics.com/)
- **NVIDIA JetBot** - Autonomous driving on Jetson Nano. Great for perception projects. [Wiki](https://github.com/NVIDIA-AI-IOT/jetbot/wiki)

## Related Awesome Lists

Other curated lists covering adjacent topics.

- [Awesome LLM Robotics](https://github.com/GT-RIPL/Awesome-LLM-Robotics) - LLM/VLM applications in robotics.
- [Awesome Robotics](https://github.com/kiloreux/awesome-robotics) - General robotics resources.
- [Awesome Robotics Libraries](https://github.com/jslee02/awesome-robotics-libraries) - Robotics software libraries.
- [Awesome Robotics 3D](https://github.com/zubair-irshad/Awesome-Robotics-3D) - 3D vision for robotics.
- [Awesome Embodied Agent](https://github.com/zchoi/Awesome-Embodied-Robotics-and-Agent) - Embodied AI with VLMs and LLMs.
- [Awesome World Models](https://github.com/operator22th/awesome-world-models-for-robots) - World models for robotics.
- [Awesome Generative AI](https://github.com/natasha-rye/awesome-generative-ai) - Broader generative AI resources.
- [Awesome Deep RL](https://github.com/kengz/awesome-deep-rl) - Deep reinforcement learning resources.
- [Awesome Imitation Learning](https://github.com/kristery/Awesome-Imitation-Learning) - Learning from demonstrations.

---

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.
