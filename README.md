# Awesome Physical AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of resources to learn, build, deploy, and stay current in Physical AI / Embodied AI.

Physical AI and Embodied AI represent the convergence of large-scale machine learning with robotics—enabling machines that perceive, reason, and act in the physical world. This list focuses on practical resources for researchers and practitioners at every stage: from foundational learning to production deployment.

## Contents

- [Learn](#learn)
  - [Courses](#courses)
  - [Books](#books)
  - [Tutorials & Guides](#tutorials--guides)
  - [Key Papers](#key-papers)
  - [Survey Papers](#survey-papers)
- [Build](#build)
  - [Foundation Models](#foundation-models)
  - [World Models](#world-models)
  - [Simulation Platforms](#simulation-platforms)
  - [Learning Frameworks](#learning-frameworks)
  - [Robot Software Stacks](#robot-software-stacks)
  - [Perception & Representations](#perception--representations)
  - [Motion Planning & Control](#motion-planning--control)
- [Deploy](#deploy)
  - [Hardware Platforms](#hardware-platforms)
  - [Datasets & Benchmarks](#datasets--benchmarks)
  - [Safety & Evaluation](#safety--evaluation)
- [Stay Current](#stay-current)
  - [Conferences](#conferences)
  - [Community](#community)
  - [Companies & Labs](#companies--labs)
  - [Newsletters & Blogs](#newsletters--blogs)
- [Related Awesome Lists](#related-awesome-lists)

---

## Learn

Resources to build foundational knowledge in robotics AI.

### Courses

University courses and structured learning programs.

- [CS 336: Robot Learning (Stanford)](https://cs336.stanford.edu/) - Modern robot learning approaches including imitation and reinforcement learning.
- [CS 224R: Deep RL for Robotics (Stanford)](http://cs224r.stanford.edu/) - Deep reinforcement learning for real-world robots.
- [CS 287: Advanced Robotics (Berkeley)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/) - Planning, learning, and control for robotics.
- [16-831: Introduction to Robot Learning (CMU)](https://www.cs.cmu.edu/~robotlearning/) - Foundations of robot learning from CMU.
- [MIT 6.4210: Robotic Manipulation](https://manipulation.csail.mit.edu/) - Perception, planning, and control for manipulation by Russ Tedrake.
- [Deep RL Bootcamp (Berkeley)](https://sites.google.com/view/deep-rl-bootcamp/) - Intensive deep RL course materials.
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI's educational resource for deep RL fundamentals.
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/) - Free hands-on course on deep reinforcement learning.
- [NVIDIA DLI Robotics](https://www.nvidia.com/en-us/training/) - Self-paced courses on Isaac Sim, ROS, and robot learning.

### Books

Foundational and advanced textbooks.

- [Probabilistic Robotics](https://mitpress.mit.edu/9780262201629/) - Thrun, Burgard, Fox. Essential text on probabilistic methods for robotics.
- [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) - Lynch & Park. Mechanics, planning, and control with free online version.
- [Robotics, Vision and Control](https://petercorke.com/rvc/) - Corke. MATLAB/Python-based introduction to robotics fundamentals.
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) - Sutton & Barto. The classic RL textbook, freely available online.
- [Planning Algorithms](http://lavalle.pl/planning/) - LaValle. Comprehensive coverage of motion planning, free online.
- [A Mathematical Introduction to Robotic Manipulation](https://www.cds.caltech.edu/~murray/mlswiki/) - Murray, Li, Sastry. Free online textbook on manipulation.
- [Introduction to Autonomous Robots](https://introduction-to-autonomous-robots.github.io/) - Correll et al. Open-source robotics textbook.

### Tutorials & Guides

Hands-on learning resources.

- [LeRobot Tutorials](https://github.com/huggingface/lerobot) - Getting started with robot learning using Hugging Face's framework.
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/) - Comprehensive guides for NVIDIA's robot learning framework.
- [MuJoCo Documentation](https://mujoco.readthedocs.io/) - Official docs with modeling and programming guides.
- [ROS 2 Tutorials](https://docs.ros.org/en/rolling/Tutorials.html) - Official tutorials for getting started with ROS 2.
- [PyBullet Quickstart](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/) - Getting started with PyBullet simulation.
- [Open X-Embodiment Tutorial](https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb) - Working with the largest robotics dataset.
- [Diffusion Policy Tutorial](https://diffusion-policy.cs.columbia.edu/) - Implementation guide for diffusion-based robot policies.

### Key Papers

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

### Survey Papers

Comprehensive overviews of key areas.

- [Foundation Models in Robotics](https://arxiv.org/abs/2312.07843) - Survey on how foundation models are transforming robotics.
- [Neural Fields in Robotics](https://arxiv.org/abs/2410.20220) - Survey on neural implicit representations for robotics applications.
- [A Survey on LLM-based Autonomous Agents](https://arxiv.org/abs/2308.11432) - Comprehensive survey on LLM-based agents.
- [Robot Learning Survey](https://arxiv.org/abs/2312.08591) - Overview of robot learning from demonstration.
- [3D Gaussian Splatting in Robotics](https://arxiv.org/abs/2410.12262) - Survey on gaussian splatting applications in robotics.
- [World Models Survey](https://arxiv.org/abs/2403.02622) - Survey on world models for autonomous systems.

---

## Build

Tools and frameworks for developing robotics AI systems.

### Foundation Models

Vision-language-action models and generalist policies.

- [π0 (Physical Intelligence)](https://www.physicalintelligence.company/) - Generalist policy combining multi-task, multi-robot data with flow matching for dexterous manipulation.
- [V-JEPA 2 (Meta FAIR)](https://ai.meta.com/vjepa/) - Yann LeCun's self-supervised world model achieving state-of-the-art video understanding and zero-shot robot planning.
- [VL-JEPA (Meta FAIR)](https://arxiv.org/abs/2512.10942) - Joint embedding predictive architecture for vision-language, 50% fewer parameters than standard VLMs.
- [Octo](https://octo-models.github.io/) - Open-source generalist robot policy trained on Open X-Embodiment, supports fine-tuning across embodiments.
- [OpenVLA](https://openvla.github.io/) - Open-source 7B-parameter vision-language-action model built on Prismatic VLMs.
- [RT-2](https://robotics-transformer2.github.io/) - Vision-language-action model transferring web knowledge to robotic control.
- [RT-X](https://robotics-transformer-x.github.io/) - X-embodiment models demonstrating positive transfer across platforms.
- [Gemini Robotics](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) - Google DeepMind's VLA models with embodied reasoning capabilities.
- [GR00T N1 (NVIDIA)](https://developer.nvidia.com/isaac/groot) - Open humanoid robot foundation model with dual-system architecture.
- [Helix (Figure)](https://www.figure.ai/) - Vision-language-action model for generalist humanoid control.
- [ACT](https://tonyzhaozh.github.io/aloha/) - Action Chunking with Transformers for learning from demonstrations.

### World Models

Models that learn to predict future states for planning and simulation.

- [V-JEPA 2 (Meta)](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) - World model trained on 1M+ hours of video enabling zero-shot robot planning with minimal robot data.
- [NVIDIA Cosmos](https://developer.nvidia.com/cosmos) - World foundation models for physically-based synthetic data generation.
- [Genie 2 (DeepMind)](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) - Foundation world model for generating interactive 3D environments.
- [UniSim](https://universal-simulator.github.io/unisim/) - Universal simulator learning real-world interactions from diverse data.
- [DayDreamer](https://danijar.com/project/daydreamer/) - World models for physical robot learning enabling sample-efficient skill acquisition.
- [Dreamer v3](https://danijar.com/project/dreamerv3/) - General algorithm for world model learning across diverse domains.
- [NVIDIA Cosmos Reason](https://developer.nvidia.com/cosmos) - Open reasoning VLM for physical world understanding.

### Simulation Platforms

Physics engines and simulation environments.

**Physics Engines**
- [MuJoCo](https://mujoco.org/) - Multi-joint dynamics with contact. Fast, accurate physics for RL research. Open-source.
- [PyBullet](https://pybullet.org/) - Open-source physics engine for robotics simulation with Python bindings.
- [Drake](https://drake.mit.edu/) - Model-based design toolbox for planning, control, and analysis.
- [Brax](https://github.com/google/brax) - Differentiable physics engine in JAX for massively parallel simulation.
- [MuJoCo XLA (MJX)](https://mujoco.readthedocs.io/en/stable/mjx.html) - JAX-based MuJoCo for GPU-accelerated simulation.
- [JAXSim](https://github.com/ami-iit/jaxsim) - Differentiable physics engine for control and robot learning.

**High-Fidelity Simulators**
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) - Robotics simulation on Omniverse with GPU physics and photorealistic rendering.
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) - Unified robot learning framework for RL, imitation learning, and motion planning.
- [Gazebo](https://gazebosim.org/) - Open-source simulator with robust physics and ROS integration.
- [CoppeliaSim](https://www.coppeliarobotics.com/) - Versatile robot simulator with integrated development environment.
- [Webots](https://cyberbotics.com/) - Open-source simulator with comprehensive documentation.
- [CARLA](https://carla.org/) - Open-source simulator for autonomous driving research.
- [Habitat](https://aihabitat.org/) - Platform for embodied AI research supporting 3D navigation.
- [AI2-THOR](https://ai2thor.allenai.org/) - Interactive 3D environments for embodied AI from Allen AI.
- [SAPIEN](https://sapien.ucsd.edu/) - Physics-rich simulation with articulated object dataset.

**RL-Focused Environments**
- [Gymnasium Robotics](https://robotics.farama.org/) - Robotics simulation environments for RL from Farama Foundation.
- [MetaWorld](https://meta-world.github.io/) - Benchmark for meta-RL with 50 manipulation tasks.
- [RoboCasa](https://robocasa.ai/) - Large-scale simulation for training generalist robots in everyday environments.
- [RoboSuite](https://robosuite.ai/) - Modular simulation framework with standardized interfaces.
- [ManiSkill](https://maniskill.ai/) - GPU-parallelized simulator with diverse manipulation tasks.
- [DM Control](https://github.com/google-deepmind/dm_control) - DeepMind's physics-based simulation and RL environment suite.

### Learning Frameworks

Libraries for robot learning research and development.

**Imitation Learning**
- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face's end-to-end library for real-world robotics with state-of-the-art methods.
- [robomimic](https://robomimic.github.io/) - Framework for robot learning from demonstration with standardized datasets.
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) - Visuomotor policy learning via action diffusion.
- [ACT](https://tonyzhaozh.github.io/aloha/) - Action Chunking with Transformers for bimanual manipulation.
- [VQ-BeT](https://sjlee.cc/vq-bet/) - Behavior generation with VQ-VAE for versatile robot control.
- [ALOHA](https://tonyzhaozh.github.io/aloha/) - Low-cost teleoperation system and imitation learning framework.
- [Mobile ALOHA](https://mobile-aloha.github.io/) - Learning bimanual mobile manipulation.

**Reinforcement Learning**
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - Reliable RL algorithm implementations in PyTorch.
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Single-file deep RL implementations with good defaults.
- [rl_games](https://github.com/Denys88/rl_games) - High-performance RL library for Isaac Gym and Isaac Lab.
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) - RL library for legged robots from ETH Zurich.
- [SKRL](https://skrl.readthedocs.io/) - Modular RL library supporting multiple ML frameworks.
- [TorchRL](https://pytorch.org/rl/) - PyTorch library for RL with modular architecture.
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/) - Scalable RL library for multi-agent and distributed training.

**General Libraries**
- [RLDS](https://github.com/google-research/rlds) - Ecosystem for recording, storing, and consuming RL datasets.
- [Gymnasium](https://gymnasium.farama.org/) - Standard API for RL environments from Farama Foundation.
- [PettingZoo](https://pettingzoo.farama.org/) - Multi-agent RL environment library.

### Robot Software Stacks

Operating systems and middleware.

- [ROS 2](https://docs.ros.org/) - Flexible framework for robot software with extensive tools and libraries.
- [ROS](https://wiki.ros.org/) - Original Robot Operating System with mature ecosystem.
- [micro-ROS](https://micro.ros.org/) - ROS 2 for microcontrollers enabling embedded robotics.
- [NVIDIA Isaac ROS](https://developer.nvidia.com/isaac-ros) - GPU-accelerated ROS 2 packages for perception and navigation.
- [MoveIt](https://moveit.ros.org/) - Motion planning framework integrated with ROS.
- [Nav2](https://docs.nav2.org/) - Navigation stack for ROS 2 with advanced planning.
- [Open-RMF](https://www.open-rmf.org/) - Open-source framework for multi-robot fleet management.
- [Robotics Library (RL)](https://www.roboticslibrary.org/) - Self-contained C++ library for kinematics, planning, and control.

### Perception & Representations

Tools for robot perception and scene understanding.

- [SAM 2](https://segment-anything.com/) - Segment Anything Model for promptable image and video segmentation.
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Open-set object detection with text prompts.
- [CLIP](https://openai.com/research/clip) - Contrastive language-image pretraining for visual understanding.
- [DINOv2](https://dinov2.metademolab.com/) - Self-supervised vision transformer features useful for robotics.
- [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - 3D scene representation for real-time novel view synthesis.
- [NeRF](https://www.matthewtancik.com/nerf) - Neural radiance fields for view synthesis and 3D reconstruction.
- [Open3D](http://www.open3d.org/) - Modern library for 3D data processing.
- [Point Cloud Library (PCL)](https://pointclouds.org/) - Large-scale library for 2D/3D image and point cloud processing.
- [OpenCV](https://opencv.org/) - Open-source computer vision library.
- [Depth Anything](https://depth-anything.github.io/) - Foundation model for monocular depth estimation.
- [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) - Grasp generation from partial point clouds.

### Motion Planning & Control

Libraries for trajectory planning and robot control.

- [OMPL](https://ompl.kavrakilab.org/) - Open Motion Planning Library with sampling-based algorithms.
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/) - Fast library for rigid body dynamics and kinematics.
- [OSQP](https://osqp.org/) - Operator splitting QP solver for real-time optimal control.
- [CasADi](https://web.casadi.org/) - Tool for nonlinear optimization and algorithmic differentiation.
- [Crocoddyl](https://github.com/loco-3d/crocoddyl) - Optimal control library for multi-contact and legged robots.
- [MuJoCo MPC](https://github.com/google-deepmind/mujoco_mpc) - Real-time predictive control using MuJoCo.
- [Ruckig](https://github.com/pantor/ruckig) - Real-time motion generation with jerk constraints.

---

## Deploy

Resources for real-world deployment.

### Hardware Platforms

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
- [Figure](https://www.figure.ai/) - General-purpose humanoid robot with AI-powered manipulation.
- [Boston Dynamics Atlas](https://bostondynamics.com/atlas/) - Advanced research humanoid with dynamic locomotion.
- [Tesla Optimus](https://www.tesla.com/optimus) - Humanoid designed for manufacturing and consumer applications.
- [Unitree H1/G1](https://www.unitree.com/) - Affordable humanoid platforms for research.
- [Agility Robotics Digit](https://agilityrobotics.com/) - Bipedal robot for logistics.
- [1X Technologies](https://www.1x.tech/) - Androids designed for safe human interaction.
- [Sanctuary AI Phoenix](https://sanctuary.ai/) - Humanoid with general-purpose AI.
- [Apptronik Apollo](https://apptronik.com/) - Humanoid robot for commercial applications.
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

### Datasets & Benchmarks

Data and evaluation standards for robotics.

**Large-Scale Manipulation Datasets**
- [Open X-Embodiment](https://robotics-transformer-x.github.io/) - Largest open-source real robot dataset with 1M+ trajectories across 22 embodiments.
- [DROID](https://droid-dataset.github.io/) - Large-scale in-the-wild manipulation dataset across 13 institutions.
- [RoboMIND](https://x-humanoid-robomind.github.io/) - Multimodal bimanual mobile manipulation dataset with 310K+ trajectories.
- [AgiBot World](https://github.com/OpenDriveLab/AgiBot-World) - Large-scale dataset for robot foundation models.
- [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) - Large and diverse dataset of robotic manipulation behaviors.
- [RH20T](https://rh20t.github.io/) - Large-scale robotic dataset with human demonstrations.
- [CALVIN](https://github.com/mees/calvin) - Benchmark for language-conditioned long-horizon manipulation.
- [RoboNet](https://www.robonet.wiki/) - Multi-robot video dataset for visual policy learning.
- [Dex-Net](https://berkeleyautomation.github.io/dex-net/) - Datasets and models for robust robotic grasping.

**Benchmarks**
- [LIBERO](https://libero-project.github.io/) - Benchmark for knowledge transfer with 130 diverse tasks.
- [FurnitureBench](https://clvrai.github.io/furniture-bench/) - Real-world furniture assembly benchmark.
- [RLBench](https://sites.google.com/view/rlbench) - Vision-guided manipulation benchmark with 100+ tasks.
- [HumanoidBench](https://sferrazza.cc/humanoidbench_site/) - Simulated humanoid benchmark for whole-body control.
- [ARNOLD](https://arnold-benchmark.github.io/) - Language-grounded task learning benchmark.
- [Colosseum](https://robot-colosseum.github.io/) - Generalization evaluation for robotic manipulation.
- [OpenEQA](https://open-eqa.github.io/) - Embodied question answering benchmark.

### Safety & Evaluation

Tools for robot safety and policy evaluation.

- [SafetyGym](https://github.com/openai/safety-gym) - Environments for safe exploration in RL.
- [NVIDIA Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) - Toolkit for adding guardrails to LLM-based applications.
- [Safe Control Gym](https://github.com/utiasDSL/safe-control-gym) - Benchmark environments for safe control and RL.
- [Isaac Lab-Arena](https://developer.nvidia.com/blog/nvidia-releases-new-physical-ai-models/) - Robot evaluation framework from NVIDIA.

---

## Stay Current

Resources for keeping up with the rapidly evolving field.

### Conferences

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

### Community

Forums, discussions, and meetups.

- [ROS Discourse](https://discourse.ros.org/) - Official ROS community discussion forum.
- [Robotics Stack Exchange](https://robotics.stackexchange.com/) - Q&A for robotics professionals and enthusiasts.
- [r/robotics](https://www.reddit.com/r/robotics/) - Reddit community for robotics discussion.
- [Hugging Face Discord](https://huggingface.co/join/discord) - Community discussions including LeRobot and Reachy.
- [Pollen Robotics Discord](https://discord.gg/pollen-robotics) - Community for Reachy and open-source robotics.
- [Robot Learning Discord](https://discord.gg/robotlearning) - Community for robot learning researchers.

### Companies & Labs

Organizations advancing Physical AI.

**Research Labs**
- [Meta FAIR](https://ai.meta.com/) - Yann LeCun's team developing V-JEPA, I-JEPA, and foundational AI research.
- [Google DeepMind Robotics](https://deepmind.google/discover/blog/?topic=robotics) - Pioneering VLA models and embodied AI.
- [NVIDIA Robotics](https://developer.nvidia.com/robotics) - Isaac platform, Cosmos, and GR00T development.
- [Stanford IRIS Lab](https://irislab.stanford.edu/) - Robot learning and manipulation research.
- [Berkeley BAIR](https://bair.berkeley.edu/) - AI research including robotics and RL.
- [MIT CSAIL](https://www.csail.mit.edu/) - Robotics and AI research at MIT.
- [CMU Robotics Institute](https://www.ri.cmu.edu/) - Comprehensive robotics research programs.

**Startups & Companies**
- [Physical Intelligence (π)](https://www.physicalintelligence.company/) - Foundation models for general-purpose robots.
- [Pollen Robotics / Hugging Face](https://www.pollen-robotics.com/) - Open-source humanoid robots including Reachy 2.
- [Figure](https://www.figure.ai/) - General-purpose humanoid robots.
- [Covariant](https://covariant.ai/) - AI for robotic picking and manipulation.
- [Skild AI](https://www.skild.ai/) - Scalable robot intelligence.
- [Generalist AI](https://generalistai.com/) - Embodied foundation models at scale.
- [1X Technologies](https://www.1x.tech/) - Safe, intelligent humanoid robots.
- [Sanctuary AI](https://sanctuary.ai/) - General-purpose AI in humanoid form.
- [Agility Robotics](https://agilityrobotics.com/) - Human-centric bipedal robots.
- [Boston Dynamics](https://bostondynamics.com/) - Advanced mobile robots and humanoids.
- [Apptronik](https://apptronik.com/) - Next-generation humanoid robots.
- [Wayve](https://wayve.ai/) - Embodied AI for autonomous driving.

### Newsletters & Blogs

Stay updated with the latest developments.

- [The Robot Report](https://www.therobotreport.com/) - News and analysis on robotics industry.
- [IEEE Spectrum Robotics](https://spectrum.ieee.org/topic/robotics/) - IEEE's robotics coverage.
- [Robotics 24/7](https://www.robotics247.com/) - Industry news and research updates.
- [Hugging Face Blog](https://huggingface.co/blog) - Updates on LeRobot, Reachy, and open-source robotics.
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/category/robotics/) - Isaac, Cosmos, and robotics AI updates.
- [Meta AI Blog](https://ai.meta.com/blog/) - V-JEPA, research updates from Yann LeCun's team.

---

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


