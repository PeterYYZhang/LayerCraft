# LayerCraft
This is the official repository for **"LayerCraft: Enhancing Text-to-Image Generation with CoT Reasoning and Layered Object Integration"**

![Workflow demonstration for LayerCraft. The user initially provides a simple prompt, “Alice in a wonderland,” and the framework
generates an image by employing Chain-of-Thought reasoning to determine both content and spatial arrangements. Subsequently, the user
applies a square mask to remove the second mushroom from the left and specifies the addition of a cute lion. After manual region selection, the framework seamlessly integrates the lion into the scene.](Teaser-double.png)
## Abstract
Text-to-image generation (T2I) has become a key area of research with broad applications. However, existing methods often struggle with complex spatial relationships and fine-grained control over multiple concepts. Many existing approaches require significant architectural modifications, extensive training, or expert-level prompt engineer-ing. To address these challenges, we introduce **LayerCraft**, an automated framework that leverages large language models (LLMs) as autonomous agents for structured procedural generation. LayerCraft enables users to customize objects within an image and supports narrative-driven creation with minimal effort. At its core, the system includes a coordinator agent that directs the process, along with two specialized agents: **ChainArchitect**, which employs chain-of-thought (CoT) reasoning to generate a dependency-aware 3D layout for precise instance-level control, and the Object-**Integration Network (OIN)**, which utilizes LoRA fine-tuning on pre-trained T2I models to seamlessly blend objects into specified regions of an image based on textual prompts—without requiring architectural changes. Extensive evaluations demonstrate LayerCraft’s versatility in applications ranging from multi-concept customization to storytelling. By providing non-experts with intuitive, precise control over T2I generation, our framework democratizes creative image creation.

## TODO:
1. OpenSource Object Integration Network (OIN) for T2I models and show more examples.
2. OpenSource ChainArchitect
3. OpenSource Dataset
