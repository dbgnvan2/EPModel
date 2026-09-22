# Model-design references in "Modeling Earth-Scale Human-Like Societies with One Billion Agents"

Source paper: Guan, He, Fan et al., Light Society, arXiv:2506.12078v2 (28 Jun 2026). 94 references in total.

Compiled 2026-09-17. Method: references were sorted by title and by the context in which the source paper cites them. The cited papers themselves were not opened. The source paper does not label its references by category; the grouping below is a judgment call. Count of references treated as model-design-specific: 42.

Numbers in square brackets are the source paper's reference numbers.

## 1. LLM-agent simulation frameworks and architectures

- [28] Liu, Y. et al. LMAgent: A large-scale multimodal agents society for multi-user simulation (2024). arXiv:2412.09237
- [29] Yang, Z. et al. OASIS: Open agent social interaction simulations with one million agents (2024). arXiv:2411.11581. Also cited for data-augmentation initialization.
- [32] Piao, J. et al. AgentSociety: Large-scale simulation of LLM-driven generative agents advances understanding of human behaviors and society (2025). arXiv:2502.08691
- [33] Zhang, X. et al. SocioVerse: A world model for social simulation powered by LLM agents and a pool of 10 million real-world users (2025). arXiv:2504.10157
- [35] Chopra, A., Kumar, S., Giray-Kuru, N., Raskar, R. & Quera-Bofarull, A. On the limits of agency in agent-based models. AAMAS 2025, 500–509. https://dl.acm.org/doi/10.5555/3709347.3743565. Cited for rule-based initialization. Note (from outside the source paper): this is the AgentTorch paper; AgentTorch is the 8.4M-agent entry in the source paper's Fig. A1.
- [36] Wang, L., Gao, H., Bo, X., Chen, X. & Wen, J.-R. YuLan-OneSim: Towards the next generation of social simulator with large language models (2025). arXiv:2505.07581
- [37] Gao, D. et al. AgentScope: A flexible yet robust multi-agent platform (2024). arXiv:2402.14034
- [38] Tang, J. et al. GenSim: A general social simulation platform with large language model based agents. NAACL 2025 System Demonstrations, 143–150. https://aclanthology.org/2025.naacl-demo.15/
- [39] Ren, R. et al. BASES: Large-scale web search user simulation with large language model based agents. Findings of EMNLP 2024, 902–917. https://aclanthology.org/2024.findings-emnlp.50/
- [40] Mou, X., Wei, Z. & Huang, X. Unveiling the truth and facilitating change: Towards agent-based large-scale social movement simulation. Findings of ACL 2024, 4789–4809. https://aclanthology.org/2024.findings-acl.285/
- [41] Zhang, X. et al. A large-scale time-aware agents simulation for influencer selection in digital advertising campaigns (2024). arXiv:2411.01143
- [70] Li, N., Gao, C., Li, M., Li, Y. & Liao, Q. EconAgent: Large language model-empowered agents for simulating macroeconomic activities. ACL 2024, 15523–15536. https://aclanthology.org/2024.acl-long.829/
- [71] Park, J. S. et al. Generative agents: Interactive simulacra of human behavior. UIST 2023, 1–22. https://doi.org/10.1145/3586183.3606763. Memory/reflection architecture. (Already in this folder as 2023_Generative_Agents_Park_2304.03442.pdf.)
- [84] Chuang, Y.-S. et al. Simulating opinion dynamics with networks of LLM-based agents. Findings of NAACL 2024, 3326–3346. https://aclanthology.org/2024.findings-naacl.211/
- [87] Gao, C. et al. S3: Social-network simulation system with large language model-empowered agents (2023). arXiv:2307.14984
- [88] Hua, W. et al. War and peace (WarAgent): Large language model-based multi-agent simulation of world wars (2023). arXiv:2311.17227
- [89] Zhang, X. et al. ElectionSim: Massive population election simulation powered by large language model driven agents (2024). arXiv:2410.20746
- [90] Li, J. et al. Agent Hospital: A simulacrum of hospital with evolvable medical agents (2024). arXiv:2405.02957
- [91] Xu, F., Zhang, J., Gao, C., Feng, J. & Li, Y. Urban generative intelligence (UGI): A foundational platform for agents in embodied city environment (2023). arXiv:2312.11813
- [92] Wang, L. et al. User behavior simulation with large language model based agents. ACM Transactions on Information Systems 43, 1–37 (2025). https://doi.org/10.1145/3708985
- [93] Mou, X. et al. From individual to society: A survey on social simulation driven by large language model-based agents (2024). arXiv:2412.03563. A survey, but its subject is this design space.

## 2. Classical (rule-based) agent-based model design

- [22] Axtell, R. L. & Farmer, J. D. Agent-based modeling in economics and finance: Past, present, and future. Journal of Economic Literature 63, 197–287 (2025).
- [23] Axtell, R. L. 120 million agents self-organize into 6 million firms: A model of the U.S. private sector. AAMAS 2016, 806–816.
- [24] Aylett-Bullock, J. et al. JUNE: open-source individual-based epidemiology simulation. Royal Society Open Science 8, 210506 (2021).
- [26] Ghaffarian, S., Roy, D., Filatova, T. & Kerle, N. Agent-based modelling of post-disaster recovery with remote sensing data. International Journal of Disaster Risk Reduction 60, 102285 (2021).
- [27] Bonabeau, E. Agent-based modeling: Methods and techniques for simulating human systems. PNAS 99, 7280–7287 (2002).
- [49] Röchert, D., Cargnino, M. & Neubaum, G. Two sides of the same leader: An agent-based model to analyze the effect of ambivalent opinion leaders in social networks. Journal of Computational Social Science 5, 1159–1205 (2022).
- [72] Schelling, T. C. Models of segregation. The American Economic Review 59, 488–493 (1969).
- [73] Axelrod, R. The dissemination of culture: A model with local convergence and global polarization. The Journal of Conflict Resolution 41, 203–226 (1997).
- [74] Epstein, J. M. Agent-based computational models and generative social science. Complexity 4, 41–60 (1999). The source paper cites this for Sugarscape.
- [75] Palmer, R. G., Arthur, W. B., Holland, J. H. & LeBaron, B. An artificial stock market. Artificial Life and Robotics 3, 27–31 (1999).
- [77] Macal, C. M., Collier, N. T., Ozik, J., Tatara, E. R. & Murphy, J. T. CHISIM: An agent-based simulation model of social interactions in a large urban area. Winter Simulation Conference 2018, 810–820.

## 3. Formal interaction and network mechanisms used in the model

- [46] Xia, H., Wang, H. & Xuan, Z. Opinion dynamics: A multidisciplinary review and perspective on future research. International Journal of Knowledge and Systems Science 2, 72–91 (2011).
- [47] Das, A., Gollapudi, S. & Munagala, K. Modeling opinion dynamics in social networks. WSDM 2014, 403–412.
- [48] Barabási, A.-L. & Albert, R. Emergence of scaling in random networks. Science 286, 509–512 (1999). The network generator the source paper uses.
- [56] Bikhchandani, S., Hirshleifer, D. & Welch, I. A theory of fads, fashion, custom, and cultural change as informational cascades. Journal of Political Economy 100, 992–1026 (1992).
- [57] Granovetter, M. Threshold models of collective behavior. American Journal of Sociology 83, 1420–1443 (1978).
- [59] Castellano, C., Fortunato, S. & Loreto, V. Statistical physics of social dynamics. Reviews of Modern Physics 81, 591–646 (2009).

## 4. Agent grounding and behavioural validation

- [42] Haerpfer, C. et al. World Values Survey: Round Seven – Country-Pooled Datafile Version 6.0 (2022). https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp. Source of the agent profiles.
- [68] Zhao, W. et al. WorldValuesBench: A large-scale benchmark dataset for multi-cultural value awareness of language models. LREC-COLING 2024, 17696–17706. https://aclanthology.org/2024.lrec-main.1539/. The source paper follows its profile-extraction method.
- [43] Xie, C. et al. Can large language model agents simulate human trust behavior? NeurIPS 37, 15674–15729 (2024). Also cited for LLM-based initialization.
- [69] Aher, G. V., Arriaga, R. I. & Kalai, A. T. Using large language models to simulate multiple humans and replicate human subject studies. ICML 2023, PMLR 202, 337–371. https://proceedings.mlr.press/v202/aher23a.html

## Borderline (not in the count of 42)

- Human experimental benchmarks used to validate agent behaviour (empirical studies, not models): [44] Cox 2004, [45] Berg, Dickhaut & McCabe 1995, [94] Cochard, Nguyen Van & Willinger 2004 (trust games); [66] Güth, Schmittberger & Schwarze 1982, [67] Henrich et al. 2001 (ultimatum game); [58] Centola 2010 (spread of behaviour in an online network experiment).
- [9] Huang et al. 2024, [10] Miotto et al. 2022, [11] Chen et al. 2024: LLM personality and role-play. Cited only as background, but they bear on persona design.
- [30] Guo et al. 2024, [31] Vallinder & Hughes 2024: cooperation among LLM agents. [86] Acerbi & Stubbersfield 2023: LLM transmission-chain content biases.
- [65] Rozemberczki, Allen & Sarkar 2021: MUSAE dataset, supplies the empirical Twitch-DE network; the paper itself is about node embeddings.
- [25] Aylett-Bullock et al. 2022: discussion of epidemiological modelling challenges, not a model.

## Excluded

- [1]–[8]: LLM and agent background.
- [12]–[15]: LLM awareness and emotion.
- [16]–[21]: social-science methods.
- [34]: election prediction by LLM reasoning, not a simulation.
- [50], [51]: opinion-leader and influencer background.
- [52]–[55], [60], [61]: psychology used to interpret results.
- [62]–[64]: LightGBM, Transformer, Qwen3 (surrogate tooling).
- [76], [78]–[83]: behaviour trees and reinforcement learning, cited for the limits of those approaches; [82] and [83] are unrelated to social simulation.
- [85]: training socially aligned language models.
