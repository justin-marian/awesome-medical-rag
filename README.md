<div align="center">

# :hospital: Awesome Medical & Multilingual <br> QA/RAG Research Papers

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/justin-marian/awesome-medical-rag) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[![GitHub stars](https://img.shields.io/github/stars/justin-marian/awesome-medical-rag?style=flat&color=yellow&label=Stars)](https://github.com/justin-marian/awesome-medical-rag/stargazers) [![GitHub forks](https://img.shields.io/github/forks/justin-marian/awesome-medical-rag?style=flat&color=orange&label=Forks)](https://github.com/justin-marian/awesome-medical-rag/network) [![GitHub watchers](https://img.shields.io/github/watchers/justin-marian/awesome-medical-rag?style=flat&label=Watchers)](https://github.com/justin-marian/awesome-medical-rag/watchers) [![GitHub last commit](https://img.shields.io/github/last-commit/justin-marian/awesome-medical-rag?color=blue&label=Last%20Update)](https://github.com/justin-marian/awesome-medical-rag/commits/main) [![GitHub issues](https://img.shields.io/github/issues/justin-marian/awesome-medical-rag?color=red)](https://github.com/justin-marian/awesome-medical-rag/issues) [![GitHub pull requests](https://img.shields.io/github/issues-pr/justin-marian/awesome-medical-rag?color=purple)](https://github.com/justin-marian/awesome-medical-rag/pulls) [![GitHub contributors](https://img.shields.io/github/contributors/justin-marian/awesome-medical-rag?color=teal)](https://github.com/justin-marian/awesome-medical-rag/graphs/contributors)

</div>

<br>

> [!TIP]
> **The Definitive Knowledge Hub for Medical AI**
>
> Welcome to a meticulously curated collection of **200+** research papers, datasets, and benchmarks. This repository bridges the gap between **Structured Medical Knowledge** (Knowledge Graphs) and **Generative Reasoning** (LLMs), focusing on:
> * **Retrieval-Augmented Generation (RAG):** Techniques to ground AI in factual clinical data.
> * **Multilingual Equity:** Ensuring medical AI works across diverse languages and cultures.
> * **Complex Reasoning:** Moving from simple Q&A to multi-hop clinical decision support.

> [!NOTE]
> **How to Navigate**
>
> We have organized the research taxonomically to help you find exactly what you need:
> * **By Method:** Looking for *GraphRAG*, *PEFT*, or *Visual QA*?
> * **By Domain:** Interested in *Mental Health*, *Legal*, or *Finance*?
>
> Use the [**Table of Contents**](#-table-of-contents) below to jump straight to your area of interest.

---

## :scroll: Table of Contents

- [:hospital: Awesome Medical \& Multilingual  QA/RAG Research Papers](#hospital-awesome-medical--multilingual--qarag-research-papers)
  - [:scroll: Table of Contents](#scroll-table-of-contents)
  - [:dna: Medical \& Clinical LLMs](#dna-medical--clinical-llms)
  - [:books: Multilingual \& Cross-Lingual QA](#books-multilingual--cross-lingual-qa)
  - [:link: Knowledge Graphs \& Reasoning](#link-knowledge-graphs--reasoning)
  - [:magnet: RAG \& Retrieval Systems](#magnet-rag--retrieval-systems)
  - [:brain: Specialized Domains (Mental Health, Finance, Legal)](#brain-specialized-domains-mental-health-finance-legal)
  - [:trophy: Benchmarks \& Datasets](#trophy-benchmarks--datasets)
  - [:camera: Multimodal \& Visual QA](#camera-multimodal--visual-qa)
  - [:telescope: Future Horizons](#telescope-future-horizons)

---

## :dna: Medical & Clinical LLMs

> [!IMPORTANT]
> **Foundational Models & Clinical Adaptations**
>
> This section aggregates Large Language Models (LLMs) rigorously adapted for the medical domain. It encompasses a spectrum of methodologies from **Continued Pre-training** on biomedical corpora to **Instruction Fine-tuning** and **Parameter-Efficient Fine-Tuning (PEFT)**. These models bridge the gap between general-purpose AI and the high-stakes requirements of clinical reasoning and healthcare NLP.

| Topic | Full Title | Resources | Notes |
| :--- | :--- | :--- | :--- |
| **Medical Adaption** | <small>HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs</small> | [![arXiv](https://img.shields.io/badge/arXiv-2311.09774-b31b1b.svg)](https://arxiv.org/abs/2311.09774) [![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/FreedomIntelligence/HuatuoGPT-II) | One-stage training for medical adaption. |
| **Medical LLM** | <small>Medical mT5: An Open-Source Multilingual Text-to-Text LLM for The Medical Domain</small> | [![arXiv](https://img.shields.io/badge/arXiv-2404.07613-b31b1b.svg)](https://arxiv.org/abs/2404.07613) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-orange)](https://huggingface.co/papers/2404.07613) | Multilingual T5 for medical domain. |
| **Closed-Domain QA** | <small>Large Language Models Encode Clinical Knowledge (Med-PaLM)</small> | [![Nature](https://img.shields.io/badge/Journal-Nature-blue)](https://www.nature.com/articles/s41586-023-06291-2) [![arXiv](https://img.shields.io/badge/arXiv-2212.13138-b31b1b.svg)](https://arxiv.org/abs/2212.13138) | Seminal Med-PaLM paper. |
| **Medical Tuning** | <small>ClinicalGPT: Large Language Models Finetuned with Diverse Medical Data</small> | [![arXiv](https://img.shields.io/badge/arXiv-2306.09968-b31b1b.svg)](https://arxiv.org/abs/2306.09968) | Fine-tuning on diverse data. |
| **Clinical Fine-Tune** | <small>LlamaCare: An Instruction Fine-Tuned Large Language Model for Clinical NLP</small> | [![ACL](https://img.shields.io/badge/ACL-LREC%202024-red)](https://aclanthology.org/2024.lrec-main.930/) | Instruction fine-tuned LLaMA. |
| **Virtual Assistant** | <small>KG-Infused LLM for Virtual Health Assistant: Accelerated Inference and Enhanced Performance</small> | [![IEEE](https://img.shields.io/badge/IEEE-Xplore-blue)](https://ieeexplore.ieee.org/document/10903426) | KG-infused LLM assistant. |
| **Safe MedLLM** | <small>Medical Graph RAG: Towards Safe Medical Large Language Model via Graph RAG</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%202025-red)](https://aclanthology.org/2025.acl-long.1381/) [![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/MedicineToken/Medical-Graph-RAG) | MedGraphRAG framework. |
| **Clinical PEFT** | <small>Parameter-Efficient Fine-Tuning of LLaMA for the Clinical Domain</small> | [![ACL](https://img.shields.io/badge/ACL-ClinicalNLP-red)](https://aclanthology.org/2024.clinicalnlp-1.9/) | Clinical LLaMA-LoRA. |
| **French Model** | <small>DrBERT: A Robust Pre-trained Model in French for Biomedical and Clinical Domains</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%202023-red)](https://aclanthology.org/2023.acl-long.896/) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-orange)](https://huggingface.co/Dr-BERT/DrBERT-7GB) | BERT for French Bio/Clinical. |
| **Spanish Biomed** | <small>Pretrained Biomedical Language Models for Clinical NLP in Spanish</small> | [![ACL](https://img.shields.io/badge/ACL-BioNLP-red)](https://aclanthology.org/2022.bionlp-1.19/) | Beteo / RoBERTa-es. |
| **Vietnamese Health** | <small>ViHealthBERT: Pre-trained Language Models for Vietnamese in Health Text Mining</small> | [![ACL](https://img.shields.io/badge/ACL-LREC-red)](https://aclanthology.org/2022.lrec-1.35/) | ViHealthBERT. |
| **TCM Tuning** | <small>HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge</small> | [![arXiv](https://img.shields.io/badge/arXiv-2304.06975-b31b1b.svg)](https://arxiv.org/abs/2304.06975) [![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese) | HuaTuo (LLaMA tuning). |
| **Complex Reasoning** | <small>HuatuoGPT-o1: Towards Medical Complex Reasoning with LLMs</small> | [![arXiv](https://img.shields.io/badge/arXiv-2412.18925-b31b1b.svg)](https://arxiv.org/abs/2412.18925) | O1-like medical reasoning. |
| **Japanese 70B** | <small>70B-parameter Large Language Models in Japanese Medical Question-Answering</small> | [![ResearchGate](https://img.shields.io/badge/ResearchGate-Link-green)](https://www.researchgate.net/publication/381652469_70B-parameter_large_language_models_in_Japanese_medical_question-answering) | 70B Japanese Medical LLM. |
| **Clinical Mamba** | <small>ClinicalMamba: A Generative Clinical Language Model on Longitudinal Clinical Notes</small> | [![arXiv](https://img.shields.io/badge/arXiv-2403.05795-b31b1b.svg)](https://arxiv.org/abs/2403.05795) | Mamba model for clinical notes. |
| **Trilingual LLM** | <small>ELAINE-medLLM: Lightweight English Japanese Chinese Trilingual Large Language Model</small> | [![ACL](https://img.shields.io/badge/ACL-COLING%202025-red)](https://aclanthology.org/2025.coling-main.313/) | En-Jp-Zh Medical LLM. |
| **Note Gen** | <small>A Continued Pretrained LLM Approach for Automatic Medical Note Generation</small> | [![ACL](https://img.shields.io/badge/ACL-NAACL-red)](https://aclanthology.org/2024.naacl-short.47/) | Auto note generation. |
| **Clinical Needs** | <small>Do We Still Need Clinical LLMs?</small> | [![arXiv](https://img.shields.io/badge/arXiv-2402.04381-b31b1b.svg)](https://arxiv.org/abs/2402.04381) | Questioning need for specialized LLMs. |
| **French Biomed** | <small>AliBERT: A Pre-trained Language Model for French Biomedical Text</small> | [![ResearchGate](https://img.shields.io/badge/ResearchGate-Link-green)](https://www.researchgate.net/publication/372918929_AliBERT_A_Pre-trained_Language_Model_for_French_Biomedical_Text) | AliBERT. |
| **Portuguese** | <small>BioBERTpt: A Portuguese Neural Language Model for Clinical NER</small> | [![ACL](https://img.shields.io/badge/ACL-ClinicalNLP-red)](https://aclanthology.org/2020.clinicalnlp-1.7/) [![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/HAILab-PUCPR/BioBERTpt) | BioBERTpt. |
| **Portuguese Card** | <small>CardioBERTpt: Transformer-based Models for Cardiology Language Representation</small> | [![ACL](https://img.shields.io/badge/ACL-ClinicalNLP-red)](https://aclanthology.org/2022.clinicalnlp-1.15/) | CardioBERTpt. |
| **Spanish Clinical** | <small>Clinical Flair: A Pre-Trained Language Model for Spanish Clinical NLP</small> | [![ACL](https://img.shields.io/badge/ACL-ClinicalNLP-red)](https://aclanthology.org/2022.clinicalnlp-1.9/) | Clinical Flair. |
| **Vietnamese Biomed** | <small>ViPubmedDeBERTa: A Pre-trained Model for Vietnamese Biomedical Text</small> | [![ACL](https://img.shields.io/badge/ACL-PACLIC-red)](https://aclanthology.org/2023.paclic-1.83/) | ViPubmedDeBERTa. |
| **Chinese LLM** | <small>Zhongjing: Enhancing Chinese Medical LLM through Expert Feedback</small> | [![AAAI](https://img.shields.io/badge/AAAI-29907-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/29907) | Zhongjing model. |
| **Textbook Aug** | <small>Augmenting Black-box LLMs with Medical Textbooks for Biomedical QA</small> | [![arXiv](https://img.shields.io/badge/arXiv-2309.02233-b31b1b.svg)](https://arxiv.org/abs/2309.02233) | Textbook augmentation. |
| **Collaborative** | <small>MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning</small> | [![arXiv](https://img.shields.io/badge/arXiv-2311.10537-b31b1b.svg)](https://arxiv.org/abs/2311.10537) | Multi-agent collaboration. |
| **BioBERT** | <small>MSQ-BioBERT: Ambiguity Resolution to Enhance BioBERT Medical QA</small> | [![ACM](https://img.shields.io/badge/ACM-Link-blue)](https://dl.acm.org/doi/abs/10.1145/3543507.3583878) | Ambiguity resolution. |
| **EHR Repr** | <small>MHGRL: An Effective Representation Learning Model for Electronic Health Records</small> | [![ACL](https://img.shields.io/badge/ACL-LREC-red)](https://aclanthology.org/2024.lrec-main.985/) | Representation learning. |
| **CamemBERT** | <small>CamemBERT-bio: Leveraging Continual Pre-training on French Biomedical Data</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%202023-red)](https://aclanthology.org/2023.acl-long.896/) | French domain adaptation. |

---

## :books: Multilingual & Cross-Lingual QA

> [!IMPORTANT]
> **Breaking Language Barriers**
>
> This section is dedicated to democratizing AI access through **Cross-Lingual Transfer** and **Multilingual RAG**. It features benchmarks for under-represented languages (e.g., Amharic, Tigrinya, Kazakh), techniques for cultural alignment, and strategies to bridge the performance gap between high-resource and low-resource linguistic domains.

| Topic | Full Title | Resources | Notes |
| :--- | :--- | :--- | :--- |
| **Multilingual Adapters** | <small>Adapters for Enhanced Modeling of Multilingual Knowledge and Text</small> | [![ACL](https://img.shields.io/badge/ACL-EMNLP-red)](https://aclanthology.org/2022.findings-emnlp.287/) [![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/yifan-h/Multilingual_Space) | Enhanced modeling adapters. |
| **Entity Alignment** | <small>Aligning Cross-Lingual Entities with Multi-Aspect Information</small> | [![ACL](https://img.shields.io/badge/ACL-D19-red)](https://aclanthology.org/D19-1451/) | Multi-aspect alignment. |
| **Arabic Medical** | <small>AraMed: Arabic Medical Question Answering using Pretrained Transformer Language Models</small> | [![ACL](https://img.shields.io/badge/ACL-OSACT-red)](https://aclanthology.org/2024.osact-1.6/) | Arabic Medical QA. |
| **Multi-Domain QA** | <small>M2QA: Multi-domain Multilingual Question Answering</small> | [![arXiv](https://img.shields.io/badge/arXiv-2407.01091-b31b1b.svg)](https://arxiv.org/abs/2407.01091) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-orange)](https://huggingface.co/datasets/UKPLab/m2qa) | M2QA benchmark. |
| **Cross-Lingual** | <small>XLM-K: Improving Cross-Lingual Language Model Pre-training with Multilingual Knowledge</small> | [![AAAI](https://img.shields.io/badge/AAAI-21330-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/21330) | Pre-training with multilingual knowledge. |
| **Prompt Transfer** | <small>Cross-Lingual Transfer for Natural Language Inference via Multilingual Prompt Translator</small> | [![arXiv](https://img.shields.io/badge/arXiv-2403.12407-b31b1b.svg)](https://arxiv.org/abs/2403.12407) | Multilingual prompt translation. |
| **Thai LLM** | <small>Representing the Under-Represented: Cultural and Core Capability Benchmarks for Thai LLMs</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2025.coling-main.278/) | Thai cultural/core benchmarks. |
| **ICL Alignment** | <small>Improving In-context Learning of Multilingual Generative Language Models with Cross-lingual Alignment</small> | [![ACL](https://img.shields.io/badge/ACL-NAACL-red)](https://aclanthology.org/2024.naacl-long.445/) | Cross-lingual alignment for ICL. |
| **Kazakh QA** | <small>KazQAD: Kazakh Open-Domain Question Answering Dataset</small> | [![ACL](https://img.shields.io/badge/ACL-LREC-red)](https://aclanthology.org/2024.lrec-main.843/) | Kazakh open-domain QA. |
| **Multilingual RAG** | <small>Not All Languages are Equal: Insights into Multilingual Retrieval-Augmented Generation</small> | [![arXiv](https://img.shields.io/badge/arXiv-2410.21970-b31b1b.svg)](https://arxiv.org/abs/2410.21970) | Linguistic inequalities in RAG. |
| **Machine Translation** | <small>Enhancing Machine Translation Experiences with Multilingual Knowledge Graphs</small> | [![AAAI](https://img.shields.io/badge/AAAI-30563-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/30563) | MT with Multilingual KG. |
| **Amharic QA** | <small>Low Resource Question Answering: An Amharic Benchmarking Dataset (Amh-QuAD)</small> | [![ACL](https://img.shields.io/badge/ACL-RAIL-red)](https://aclanthology.org/2024.rail-1.14/) | Amh-QuAD / AmQA. |
| **Tigrinya QA** | <small>Question-Answering in a Low-resourced Language: Benchmark Dataset and Models for Tigrinya</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%202023-red)](https://aclanthology.org/2023.acl-long.661/) | Tigrinya QA benchmark. |
| **Portuguese QA** | <small>Pirá: A Bilingual Portuguese-English Dataset for Question-Answering</small> | [![ACM](https://img.shields.io/badge/ACM-Link-blue)](https://dl.acm.org/doi/10.1145/3459637.3482012) | Pirá dataset. |
| **Zero-shot CoT** | <small>AutoCAP: Towards Automatic Cross-lingual Alignment Planning for Zero-shot Chain-of-Thought</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%20Findings-red)](https://aclanthology.org/2024.findings-acl.546/) | AutoCAP. |
| **Cultural QA** | <small>NativQA: Multilingual Culturally-Aligned Natural Query for LLMs</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%20Findings-red)](https://aclanthology.org/2025.findings-acl.770/) | Culturally-aligned questions. |
| **Swedish NLP** | <small>A Benchmark for Swedish Clinical Natural Language Processing</small> | [![ACL](https://img.shields.io/badge/ACL-LREC-red)](https://aclanthology.org/2024.lrec-main.120/) | Swedish Clinical NLP benchmark. |
| **Transferability** | <small>On the Cross-lingual Transferability of Monolingual Representations</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%202020-red)](https://aclanthology.org/2020.acl-main.421/) | Monolingual transferability. |

---

## :link: Knowledge Graphs & Reasoning

> [!WARNING]
> **Bridging Structured & Unstructured Knowledge**
>
> This section dives into the complex integration of **Knowledge Graphs (KGs)** with Large Language Models. It covers advanced reasoning tasks such as **Multi-hop Reasoning**, **Subgraph Extraction**, and **Neuro-Symbolic** approaches. Papers here explore how to ground LLM generation in factual graph structures to improve accuracy and interpretability.

| Topic | Full Title | Resources | Notes |
| :--- | :--- | :--- | :--- |
| **KG Alignment** | <small>MRAEA: An Efficient and Robust Entity Alignment Approach for Cross-lingual KG</small> | [![ACM](https://img.shields.io/badge/ACM-Link-blue)](https://dl.acm.org/doi/10.1145/3336191.3371804) | Cross-lingual KG alignment. |
| **Reasoning Eval** | <small>DIVKNOWQA: Assessing the Reasoning Ability of LLMs via Open-Domain QA over KB and Text</small> | [![arXiv](https://img.shields.io/badge/arXiv-2310.20170-b31b1b.svg)](https://arxiv.org/abs/2310.20170) | Reasoning over KB and Text. |
| **KG + LLM** | <small>ChatDoctor: A Medical Chat Model Fine-Tuned on LLaMA Using Medical Domain Knowledge</small> | [![arXiv](https://img.shields.io/badge/arXiv-2303.14070-b31b1b.svg)](https://arxiv.org/abs/2303.14070) | LLaMA + Medical KG. |
| **KG Prompting** | <small>KG-CoT: Chain-of-Thought Prompting of LLMs over Knowledge Graphs</small> | [![IJCAI](https://img.shields.io/badge/IJCAI-2024-blue)](https://www.ijcai.org/proceedings/2024/734) | CoT over Knowledge Graphs. |
| **KG Sync** | <small>Domain Knowledge Exploration by Synchronizing Knowledge Graph and LLMs</small> | [![arXiv](https://img.shields.io/badge/arXiv-2311.16124-b31b1b.svg)](https://arxiv.org/abs/2311.16124) | Synchronizing KG and LLMs. |
| **Dynamic Reasoning** | <small>Dynamic Hierarchical Reasoning with Language Model and Knowledge Graph (DRLK)</small> | [![arXiv](https://img.shields.io/badge/arXiv-2311.09139-b31b1b.svg)](https://arxiv.org/abs/2311.09139) | Hierarchical reasoning with KG. |
| **Subgraph Reasoning** | <small>ReasoningLM: Enabling Structural Subgraph Reasoning in Pre-trained Language Models</small> | [![ACL](https://img.shields.io/badge/ACL-EMNLP-red)](https://aclanthology.org/2023.emnlp-main.228/) | Structural subgraph reasoning. |
| **Graph Reasoning** | <small>RoG: Reasoning on Graphs (Faithful and Interpretable LLM Reasoning)</small> | [![ICLR](https://img.shields.io/badge/ICLR-OpenReview-blue)](https://openreview.net/forum?id=C9y00Z56W7) [![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/RUC-KB-Reasoning/RoG) | Reasoning paths for QA. |
| **Graph Reasoning** | <small>KG-GPT: A General Framework for Reasoning on Knowledge Graphs Using LLMs</small> | [![ACL](https://img.shields.io/badge/ACL-EMNLP-red)](https://aclanthology.org/2023.findings-emnlp.631/) | KG-GPT framework. |
| **KG Integration** | <small>Bridging the Gap: Integrating Knowledge Graphs into LLMs for Complex QA</small> | [![OpenReview](https://img.shields.io/badge/OpenReview-Link-blue)](https://openreview.net/forum?id=fQjPKAiNbF) | KG + LLM for complex QA. |
| **Multilingual KG** | <small>Joint Completion and Alignment of Multilingual Knowledge Graphs</small> | [![ACL](https://img.shields.io/badge/ACL-EMNLP-red)](https://aclanthology.org/2022.emnlp-main.817/) | JMAC model. |
| **Knowledge Injection** | <small>KIMERA: Injecting Domain Knowledge into Vacant Transformer Heads</small> | [![ACL](https://img.shields.io/badge/ACL-LREC-red)](https://aclanthology.org/2022.lrec-1.38/) | Retraining vacant heads. |
| **Quotes KG** | <small>QuoteKG: A Multilingual Knowledge Graph of Quotes</small> | [![ResearchGate](https://img.shields.io/badge/ResearchGate-Link-green)](https://www.researchgate.net/publication/360958399_QuoteKG_A_Multilingual_Knowledge_Graph_of_Quotes) | KG of quotes (55 langs). |
| **TCM KG** | <small>Traditional Chinese Medicine Knowledge Graph Construction Based on LLMs</small> | [![MDPI](https://img.shields.io/badge/MDPI-Electronics-blue)](https://www.mdpi.com/2079-9292/13/7/1395) | TCM KG construction. |
| **Updating KG** | <small>Up To Date: Automatic Updating Knowledge Graphs Using LLMs</small> | [![Elsevier](https://img.shields.io/badge/Elsevier-Direct-orange)](https://www.sciencedirect.com/science/article/pii/S1877050924030072) | Auto updating KGs. |
| **Semi-supervised** | <small>Semi-supervised Entity Alignment via Joint Knowledge Embedding (KECG)</small> | [![ACL](https://img.shields.io/badge/ACL-D19-red)](https://aclanthology.org/D19-1274/) | KECG model. |
| **Alignment Transf.** | <small>Multi-Modal Knowledge Graph Transformer Framework for Entity Alignment</small> | [![ACL](https://img.shields.io/badge/ACL-EMNLP-red)](https://aclanthology.org/2023.findings-emnlp.70/) | Meaformer. |
| **Contrastive** | <small>Multi-modal Contrastive Representation Learning for Entity Alignment</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2022.coling-1.227/) | MCLEA model. |
| **KBQA Framework** | <small>ChatKBQA: A Generate-then-Retrieve Framework for KBQA with Fine-tuned LLMs</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%20Findings-red)](https://aclanthology.org/2024.findings-acl.122/) | Generate-then-Retrieve. |
| **Few-Shot KBQA** | <small>KB-BINDER: A Unified Semantically Parsing Framework for KBQA</small> | [![arXiv](https://img.shields.io/badge/arXiv-2304.09540-b31b1b.svg)](https://arxiv.org/abs/2304.09540) | Few-shot semantic parsing. |
| **Semantic Parsing** | <small>LLM-based Semantic Parsing for Conversational QA over Knowledge Graphs</small> | [![arXiv](https://img.shields.io/badge/arXiv-2310.10648-b31b1b.svg)](https://arxiv.org/abs/2310.10648) | Semantic parsing for ConvQA. |
| **Complex QA** | <small>KQA Pro: A Dataset with Explicit Compositional Programs for Complex KBQA</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%202022-red)](https://aclanthology.org/2022.acl-long.422/) | KQA Pro dataset. |
| **Rule-Guided** | <small>Rule-KBQA: Rule-Guided Reasoning for Complex KBQA with LLMs</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2025.coling-main.562/) | Rule-guided reasoning. |
| **Medical KG** | <small>Research on Medical Question Answering System Based on Knowledge Graph</small> | [![ResearchGate](https://img.shields.io/badge/ResearchGate-Link-green)](https://www.researchgate.net/publication/348854825_Research_on_Medical_Question_Answering_System_Based_on_Knowledge_Graph) | Neo4j-based Medical KG. |
| **Knowledge Injection** | <small>Infusing Disease Knowledge into BERT for Health QA and Inference</small> | [![ACL](https://img.shields.io/badge/ACL-EMNLP-red)](https://aclanthology.org/2020.emnlp-main.372/) | Infusing knowledge into BERT. |

---

## :magnet: RAG & Retrieval Systems

> [!TIP]
> **Augmenting Generation with External Knowledge**
>
> This section focuses on the critical integration of Retrieval-Augmented Generation (RAG) to mitigate hallucinations and ensure factual consistency. It covers the full lifecycle of RAG development: from **Dynamic Retrieval** and **Re-ranking** strategies to advanced **GraphRAG** implementations. Special emphasis is placed on **Evaluation Frameworks** designed to rigorously measure relevance, faithfulness, and answer credibility.

| Topic | Full Title | Resources | Notes |
| :--- | :--- | :--- | :--- |
| **RAG Evaluation** | <small>Adapting Standard Retrieval Benchmarks to Evaluate Generated Answers</small> | [![arXiv](https://img.shields.io/badge/arXiv-2401.04842-b31b1b.svg)](https://arxiv.org/abs/2401.04842) | Adapting benchmarks for RAG eval. |
| **RAG Relevance** | <small>DR-RAG: Applying Dynamic Document Relevance to Retrieval-Augmented Generation</small> | [![arXiv](https://img.shields.io/badge/arXiv-2406.07348-b31b1b.svg)](https://arxiv.org/abs/2406.07348) | Dynamic document relevance. |
| **RAG Chatbots** | <small>Automated Question-Answer Generation for Evaluating RAG-based Chatbots</small> | [![ACL](https://img.shields.io/badge/ACL-CL4Health-red)](https://aclanthology.org/2024.cl4health-1.25/) | Auto QA generation for evaluation. |
| **RAG Queries** | <small>RichRAG: Crafting Rich Responses for Multi-faceted Queries in RAG</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2025.coling-main.750/) | Handling multi-faceted queries. |
| **RAG Eval** | <small>Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems</small> | [![arXiv](https://img.shields.io/badge/arXiv-2411.19710-b31b1b.svg)](https://arxiv.org/abs/2411.19710) | Taxonomy for RAG eval. |
| **Clinical RAG** | <small>ClinicalRAG: Enhancing Clinical Decision Support through Heterogeneous Knowledge Retrieval</small> | [![ACL](https://img.shields.io/badge/ACL-KnowLLM-red)](https://aclanthology.org/2024.knowllm-1.6/) | Heterogeneous retrieval for CDS. |
| **RAG Practices** | <small>Enhancing Retrieval-Augmented Generation: A Study of Best Practices</small> | [![arXiv](https://img.shields.io/badge/arXiv-2501.07391-b31b1b.svg)](https://arxiv.org/abs/2501.07391) | Study of RAG best practices. |
| **RAG Credibility** | <small>How Credible Is an Answer From Retrieval-Augmented LLMs?</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2025.coling-main.285/) | Credibility in Multi-Hop QA. |
| **Adaptive RAG** | <small>Adaptive-RAG: Learning to Adapt RAG LLMs through Question Complexity</small> | [![ACL](https://img.shields.io/badge/ACL-NAACL-red)](https://aclanthology.org/2024.naacl-long.389/) | Adapting based on complexity. |
| **Knowledge Boundary** | <small>Investigating the Factual Knowledge Boundary of LLMs with Retrieval Augmentation</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2025.coling-main.250/) | Factual boundaries in RAG. |
| **Ranking/Re-ranking** | <small>KG-Rank: Enhancing LLMs for Medical QA with Knowledge Graphs and Ranking Techniques</small> | [![ACL](https://img.shields.io/badge/ACL-BioNLP-red)](https://aclanthology.org/2024.bionlp-1.13/) | Ranking with KGs. |
| **RAG Certainty** | <small>RAG Certainty: Quantifying the Certainty of Context-Based Responses by LLMs</small> | [![IEEE](https://img.shields.io/badge/IEEE-Xplore-blue)](https://ieeexplore.ieee.org/document/10903445) | RAG Certainty metric. |
| **Quantization** | <small>4-bit Quantization in Vector-Embedding for RAG</small> | [![arXiv](https://img.shields.io/badge/arXiv-2501.10534-b31b1b.svg)](https://arxiv.org/abs/2501.10534) | 4-bit embedding quantization. |
| **KG-RAG** | <small>Knowledge Graph-extended Retrieval Augmented Generation (KG-RAG)</small> | [![arXiv](https://img.shields.io/badge/arXiv-2504.08893-b31b1b.svg)](https://arxiv.org/abs/2504.08893) | KG extended RAG. |
| **Unified RAG** | <small>Reasoning by Exploration: A Unified Approach to Retrieval and Generation over Graphs (RoE)</small> | [![arXiv](https://img.shields.io/badge/arXiv-2510.07484-b31b1b.svg)](https://arxiv.org/abs/2510.07484) | Reasoning by Exploration. |
| **Region-First** | <small>ReGraM: Region-First Knowledge Graph Reasoning for Medical QA</small> | [![arXiv](https://img.shields.io/badge/arXiv-2601.09280-b31b1b.svg)](https://arxiv.org/abs/2601.09280) | Region-first reasoning. |
| **Dynamic Rank** | <small>DynRank: Improve Passage Retrieval with Dynamic Zero-Shot Prompting</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2025.coling-main.319/) | DynRank. |
| **Re-Ranking** | <small>ASRank: Zero-Shot Re-Ranking with Answer Scent for Document Retrieval</small> | [![arXiv](https://img.shields.io/badge/arXiv-2501.15245-b31b1b.svg)](https://arxiv.org/abs/2501.15245) | ASRank. |
| **Low-Resource Retrieval** | <small>Unsupervised Domain Adaptation of Dense Retrieval via Zero-Shot Sim Transfer</small> | [![arXiv](https://img.shields.io/badge/arXiv-2112.07577-b31b1b.svg)](https://arxiv.org/abs/2112.07577) | Unsupervised domain adaptation. |
| **Passage Expansion** | <small>Knowledge Graph-Guided Retrieval Augmented Generation (KG$^2$RAG)</small> | [![arXiv](https://img.shields.io/badge/arXiv-2502.06864-b31b1b.svg)](https://arxiv.org/abs/2502.06864) | KG for chunk expansion. |
| **Copilot Reasoning** | <small>MedRAG: Enhancing RAG with KG-Elicited Reasoning for Healthcare Copilot</small> | [![arXiv](https://img.shields.io/badge/arXiv-2502.04413-b31b1b.svg)](https://arxiv.org/abs/2502.04413) | MedRAG copilot. |
| **ConvRAG** | <small>Boosting Conversational Question Answering with Fine-Grained Retrieval-Augmentation</small> | [![arXiv](https://img.shields.io/badge/arXiv-2403.18243-b31b1b.svg)](https://arxiv.org/abs/2403.18243) | Fine-grained retrieval + self-check. |
| **Nutrigenetics** | <small>A Retrieval-Augmented Generation Application For Question-Answering in Nutrigenetics</small> | [![ResearchGate](https://img.shields.io/badge/ResearchGate-Link-green)](https://www.researchgate.net/publication/383998708_Enhancing_Dietary_Supplement_Question_Answer_via_Retrieval-Augmented_Generation_RAG_with_LLM) | Nutrigenetic GraphRAG. |
| **Reddit RAG** | <small>Two-Layer RAG Framework for Low-Resource Medical QA Using Reddit Data</small> | [![Journal](https://img.shields.io/badge/Journal-JMIR-blue)](https://www.jmir.org/2025/1/e66220) | Reddit-based Medical RAG. |
| **Query Reformulation** | <small>Semantic Grounding of LLMs Using Knowledge Graphs for Query Reformulation</small> | [![IEEE](https://img.shields.io/badge/IEEE-Xplore-blue)](https://ieeexplore.ieee.org/document/10825835) | Semantic grounding. |
| **TCM GraphRAG** | <small>OpenTCM: A GraphRAG-Empowered LLM-based System for TCM</small> | [![arXiv](https://img.shields.io/badge/arXiv-2504.20118-b31b1b.svg)](https://arxiv.org/abs/2504.20118) | GraphRAG for TCM. |
| **Knowledge Selection** | <small>FlexiQA: Leveraging LLM’s Evaluation Capabilities for Flexible Knowledge Selection</small> | [![ACL](https://img.shields.io/badge/ACL-EACL-red)](https://aclanthology.org/2024.findings-eacl.4/) | Flexible knowledge selection. |
| **Historical RAG** | <small>HiRAG: A Historical Information-Driven RAG Framework for Summarization</small> | [![PMLR](https://img.shields.io/badge/PMLR-Link-blue)](https://proceedings.mlr.press/v260/zhou25a.html) | Historical info for summarization. |
| **Uncertainty** | <small>Modeling Uncertainty and Using Post-fusion as Fallback Improves RAG</small> | [![ACL](https://img.shields.io/badge/ACL-KnowLLM-red)](https://aclanthology.org/2024.knowllm-1.7/) | Post-fusion fallback. |
| **Context Use** | <small>Desiderata for the Context Use of Question Answering Systems</small> | [![ACL](https://img.shields.io/badge/ACL-EACL-red)](https://aclanthology.org/2024.eacl-long.47/) | Context use desiderata. |

---

## :brain: Specialized Domains (Mental Health, Finance, Legal)

> [!CAUTION]
> **High-Stakes & Regulatory Compliance**
>
> This section covers applications in sensitive domains where error tolerance is low and privacy is paramount. Papers here explore **Explainability** in financial forecasting, **Anonymity** in legal court decisions, and **Empathy** in mental health support. The focus is on domain-specific adaptation and ethical guardrails.

| Topic | Full Title | Resources | Notes |
| :--- | :--- | :--- | :--- |
| **Mental Health QA** | <small>MentalQA: An Annotated Arabic Corpus for Questions and Answers of Mental Healthcare</small> | [![arXiv](https://img.shields.io/badge/arXiv-2405.12619-b31b1b.svg)](https://arxiv.org/abs/2405.12619) | Arabic Mental Healthcare QA. |
| **Mental Health** | <small>Chinese MentalBERT: Domain-Adaptive Pre-training on Social Media for Chinese Mental Health</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%20Findings-red)](https://aclanthology.org/2024.findings-acl.629/) | Chinese MentalBERT. |
| **Mental Privacy** | <small>Privacy Aware Question-Answering System for Online Mental Health Risk Assessment</small> | [![ACL](https://img.shields.io/badge/ACL-BioNLP-red)](https://aclanthology.org/2023.bionlp-1.18/) | Privacy-aware QA. |
| **Financial QA** | <small>FinReflectKG - MultiHop: Financial QA Benchmark for Reasoning with KG Evidence</small> | [![OpenReview](https://img.shields.io/badge/OpenReview-PDF-blue)](https://openreview.net/pdf/303ebed0963c30d5f9659eb126a4b5da9ae55c66.pdf) | Financial multi-hop reasoning. |
| **Finance KG** | <small>FinQA: A Training-Free Dynamic Knowledge Graph QA System in Finance</small> | [![Springer](https://img.shields.io/badge/Springer-Link-blue)](https://link.springer.com/chapter/10.1007/978-3-031-70371-3_32) | FinQA (Training-Free). |
| **Legal LLM** | <small>InLegalLLaMA: Indian Legal Knowledge Enhanced Large Language Model</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2025.coling-main.738/) | Indian Legal Knowledge LLM. |
| **Legal RAG** | <small>Interpretable Long-Form Legal QA with Retrieval-Augmented LLMs</small> | [![AAAI](https://img.shields.io/badge/AAAI-30232-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/30232) | Interpretability in Legal QA. |
| **Legal GraphRAG** | <small>Myanmar Law Cases and Proceedings Retrieval with GraphRAG</small> | [![IEEE](https://img.shields.io/badge/IEEE-Xplore-blue)](https://ieeexplore.ieee.org/document/10825155) | GraphRAG for Law. |
| **Legislative RAG** | <small>LexDrafter: Terminology Drafting for Legislative Documents using RAG</small> | [![ACL](https://img.shields.io/badge/ACL-LREC-red)](https://aclanthology.org/2024.lrec-main.913/) | Legislative drafting. |
| **Legal Retriever** | <small>UniLR: Unleashing the Power of LLMs on Multiple Legal Tasks with a Unified Legal Retriever</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%202025-red)](https://aclanthology.org/2025.acl-long.584/) | Unified Legal Retriever. |
| **Re-Identification** | <small>Anonymity at Risk? Assessing Re-Identification Capabilities of LLMs in Court Decisions</small> | [![ACL](https://img.shields.io/badge/ACL-NAACL-red)](https://aclanthology.org/2024.findings-naacl.157/) | Re-ID in court decisions. |
| **Crypto QA** | <small>CryptOpiQA: A new Opinion and Question Answering dataset on Cryptocurrency</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2025.coling-main.736/) | Crypto sentiment/QA. |
| **Geopolitical** | <small>Scaling LLM-Based Knowledge Graph Generation: A Case Study of Italian Geopolitical News</small> | [![IEEE](https://img.shields.io/badge/IEEE-Xplore-blue)](https://ieeexplore.ieee.org/document/10825345) | Italian Geopolitical KG. |
| **e-Governance** | <small>An Open-Domain QA System for e-Governance</small> | [![ACL](https://img.shields.io/badge/ACL-CLIB-red)](https://aclanthology.org/2022.clib-1.12/) | Open-domain QA. |
| **Education** | <small>Enhancing Textbook Question Answering with KG-Augmented LLMs</small> | [![PMLR](https://img.shields.io/badge/PMLR-Link-blue)](https://proceedings.mlr.press/v260/he25a.html) | KG-augmented Textbook QA. |
| **Education** | <small>Introducing CQuAE: A New French Contextualised QA Corpus for Education</small> | [![ACL](https://img.shields.io/badge/ACL-LREC-red)](https://aclanthology.org/2024.lrec-main.808/) | CQuAE educational corpus. |
| **Grading** | <small>SteLLA: A Structured Grading System Using LLMs with RAG</small> | [![arXiv](https://img.shields.io/badge/arXiv-2501.09092-b31b1b.svg)](https://arxiv.org/abs/2501.09092) | SteLLA grading system. |
| **Linguistics** | <small>ELQA: A Corpus of Metalinguistic Questions and Answers about English</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%202023-red)](https://aclanthology.org/2023.acl-long.113/) | Metalinguistic QA corpus. |

---

## :trophy: Benchmarks & Datasets

> [!NOTE]
> **Evaluating the State-of-the-Art**
>
> This section is a comprehensive repository of the gold-standard datasets required to train and evaluate medical QA systems. It spans **Multi-Choice Licensing Exams** (USMLE, Chinese Medical Exam), **Open-Ended Clinical QA**, and specialized **Shared Tasks** (e.g., BioASQ). These resources are essential for benchmarking performance across different modalities, languages, and reasoning complexities.

| Topic | Full Title | Resources | Notes |
| :--- | :--- | :--- | :--- |
| **Medical Dataset** | <small>Huatuo-26M: A Large-Scale Chinese Medical Question Answering Dataset</small> | [![arXiv](https://img.shields.io/badge/arXiv-2305.01526-b31b1b.svg)](https://arxiv.org/abs/2305.01526) | 26M QA pairs dataset. |
| **Multi-Subject QA** | <small>MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain QA</small> | [![PMLR](https://img.shields.io/badge/PMLR-CHIL-blue)](https://proceedings.mlr.press/v174/pal22a.html) | 194k Indian medical questions. |
| **Open Domain QA** | <small>MedQA: What Disease does this Patient Have? (USMLE-based)</small> | [![MDPI](https://img.shields.io/badge/MDPI-Applied%20Sciences-blue)](https://www.mdpi.com/2076-3417/11/14/6421) | MedQA (USMLE-based). |
| **Clinical Benchmark** | <small>M-QALM: A Benchmark to Assess Clinical Reading Comprehension and Knowledge Recall</small> | [![arXiv](https://img.shields.io/badge/arXiv-2406.03699-b31b1b.svg)](https://arxiv.org/abs/2406.03699) | Clinical Reading Comp & Recall. |
| **French Benchmark** | <small>DrBenchmark: A Large Language Understanding Evaluation Benchmark for French Biomedical Domain</small> | [![ACL](https://img.shields.io/badge/ACL-LREC-red)](https://aclanthology.org/2024.lrec-main.478/) | 20 tasks for French biomedical. |
| **Multilingual Med** | <small>Towards Building Multilingual Language Model for Medicine (MMedBench)</small> | [![arXiv](https://img.shields.io/badge/arXiv-2402.13963-b31b1b.svg)](https://arxiv.org/abs/2402.13963) | MMedBench dataset. |
| **Strategy QA** | <small>StrategyQA: Did Aristotle Use a Laptop? A QA Benchmark with Implicit Reasoning Strategies</small> | [![arXiv](https://img.shields.io/badge/arXiv-2101.02235-b31b1b.svg)](https://arxiv.org/abs/2101.02235) | Implicit reasoning benchmark. |
| **Biomed Dataset** | <small>PubMedQA: A Dataset for Biomedical Research Question Answering</small> | [![arXiv](https://img.shields.io/badge/arXiv-1909.06146-b31b1b.svg)](https://arxiv.org/abs/1909.06146) | PubMedQA dataset. |
| **Multilingual FAQ** | <small>MFAQ: a Multilingual FAQ Dataset</small> | [![ACL](https://img.shields.io/badge/ACL-MRQA-red)](https://aclanthology.org/2021.mrqa-1.1/) | 6M FAQ pairs. |
| **Literary Fiction** | <small>LFED: A Literary Fiction Evaluation Dataset for Large Language Models</small> | [![ACL](https://img.shields.io/badge/ACL-LREC-red)](https://aclanthology.org/2024.lrec-main.915/) | Literary fiction eval. |
| **Dynamic QA** | <small>Let LLMs Take on the Latest Challenges! A Chinese Dynamic QA Benchmark</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2025.coling-main.695/) | Dynamic QA benchmark. |
| **Arabic Dataset** | <small>ArabicaQA: A Comprehensive Dataset for Arabic Question Answering</small> | [![arXiv](https://img.shields.io/badge/arXiv-2403.17848-b31b1b.svg)](https://arxiv.org/abs/2403.17848) | Comprehensive Arabic QA. |
| **Chinese Benchmark** | <small>CMB: A Comprehensive Medical Benchmark in Chinese</small> | [![ACL](https://img.shields.io/badge/ACL-NAACL-red)](https://aclanthology.org/2024.naacl-long.343/) | Chinese medical benchmark. |
| **French Medical** | <small>FrenchMedMCQA: A French Multiple-Choice QA Dataset for Medical Domain</small> | [![ACL](https://img.shields.io/badge/ACL-LOUHI-red)](https://aclanthology.org/2022.louhi-1.5/) | French Multi-Choice Medical QA. |
| **Realistic QA** | <small>RealMedQA: A Pilot Biomedical QA Dataset Containing Realistic Clinical Questions</small> | [![PMC](https://img.shields.io/badge/PMC-Link-blue)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12099375/) | Real clinical questions (NICE). |
| **Pharmacist Exam** | <small>ExplainCPE: A Free-text Explanation Benchmark of Chinese Pharmacist Examination</small> | [![ACL](https://img.shields.io/badge/ACL-EMNLP-red)](https://aclanthology.org/2023.findings-emnlp.129/) | Pharmacist exam explanations. |
| **Arabic Health** | <small>AraHealthQA 2025: The First Shared Task on Arabic Health QA</small> | [![ACL](https://img.shields.io/badge/ACL-ArabicNLP-red)](https://aclanthology.org/2025.arabicnlp-sharedtasks.18/) | Shared task (Mental/General). |
| **Japanese Eval** | <small>JMedBench: A Benchmark for Evaluating Japanese Biomedical LLMs</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2025.coling-main.395/) | JMedBench. |
| **EHR QA** | <small>DrugEHRQA: A QA Dataset on Structured and Unstructured EHR</small> | [![ACL](https://img.shields.io/badge/ACL-LREC-red)](https://aclanthology.org/2022.lrec-1.117/) | DrugEHRQA. |
| **Radiology QA** | <small>RadQA: A QA Dataset to Improve Comprehension of Radiology Reports</small> | [![ACL](https://img.shields.io/badge/ACL-LREC-red)](https://aclanthology.org/2022.lrec-1.672/) | RadQA. |
| **Clinical QA** | <small>RJUA-QA: A Comprehensive QA Dataset for Clinical Reasoning in Urology</small> | [![arXiv](https://img.shields.io/badge/arXiv-2312.09785-b31b1b.svg)](https://arxiv.org/abs/2312.09785) | Urology clinical reasoning. |
| **Short-Answer** | <small>ACTA: Short-Answer Grading in High-Stakes Medical Exams</small> | [![ACL](https://img.shields.io/badge/ACL-BEA-red)](https://aclanthology.org/2023.bea-1.36/) | Grading medical exams. |
| **Arabic Wikipedia** | <small>ArTrivia: Harvesting Arabic Wikipedia to Build A New Arabic QA Dataset</small> | [![ACL](https://img.shields.io/badge/ACL-ArabicNLP-red)](https://aclanthology.org/2023.arabicnlp-1.17/) | Arabic Wikipedia QA dataset. |
| **Scientific QA** | <small>SciQA: Extensive Analysis of the SciQA Benchmark with LLMs</small> | [![Conference](https://img.shields.io/badge/Conf-ESWC-blue)](https://2024.eswc-conferences.org/wp-content/uploads/2024/04/146640194.pdf) | SciQA benchmark analysis. |

---

## :camera: Multimodal & Visual QA

> [!IMPORTANT]
> **Beyond Text: The Convergence of Vision and Language**
>
> This section explores the frontier of **Multimodal AI**, where Large Language Models (LLMs) connect with visual data to "see" and interpret medical contexts. It covers **Medical Visual Question Answering (Med-VQA)**, **Radiology Report Generation**, and **Multimodal Knowledge Graphs**. These resources are pivotal for systems that must reason over heterogeneous data sources, such as aligning clinical notes with pixel-level evidence from X-rays, CT scans, and pathology slides.

| Topic | Full Title | Resources | Notes |
| :--- | :--- | :--- | :--- |
| **Multimodal QA** | <small>CLIPSyntel: CLIP and LLM Synergy for Multimodal Question Summarization in Healthcare</small> | [![arXiv](https://img.shields.io/badge/arXiv-2312.11541-b31b1b.svg)](https://arxiv.org/abs/2312.11541) | CLIP + LLM for summarization. |
| **Multimodal KG** | <small>VisualSem: A High-Quality Knowledge Graph for Vision and Language</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%202021-red)](https://aclanthology.org/2021.acl-long.499/) | VisualSem KG. |
| **Visual QA** | <small>Seeing Beyond: Enhancing Visual Question Answering with Multi-Modal Retrieval</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2025.coling-industry.35/) | Multi-modal retrieval for VQA. |
| **EHR + CXR** | <small>EHRXQA: A Multi-Modal QA Dataset for Electronic Health Records with Chest X-ray Images</small> | [![NeurIPS](https://img.shields.io/badge/NeurIPS-OpenReview-blue)](https://openreview.net/forum?id=Pk2x7FPuZ4) | Multi-modal EHR QA. |
| **VQA Integration** | <small>Modality-Aware Integration with LLMs for Knowledge-Based Visual Question Answering</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%202024-red)](https://aclanthology.org/2024.acl-long.132/) | MAIL framework for VQA. |
| **Multimodal Sentiment** | <small>ConKI: Contrastive Knowledge Injection for Multimodal Sentiment Analysis</small> | [![ACL](https://img.shields.io/badge/ACL-ACL%20Findings-red)](https://aclanthology.org/2023.findings-acl.860/) | Contrastive knowledge injection. |
| **Visual Reasoning** | <small>Med-R1: Reinforcement Learning for Generalizable Medical Reasoning in VLMs</small> | [![arXiv](https://img.shields.io/badge/arXiv-2503.13939-b31b1b.svg)](https://arxiv.org/abs/2503.13939) | RL for VLMs in medicine. |
| **Vietnamese VQA** | <small>Enhancing Vietnamese VQA through Curriculum Learning on Raw and Augmented Text</small> | [![arXiv](https://img.shields.io/badge/arXiv-2503.03285-b31b1b.svg)](https://arxiv.org/abs/2503.03285) | Vietnamese VQA. |
| **Drug Interaction** | <small>MKG-FENN: A Multimodal KG Fused End-to-End Neural Network for DDI Prediction</small> | [![AAAI](https://img.shields.io/badge/AAAI-28887-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/28887) | Drug-drug interaction. |
| **Negative Sampling** | <small>Modality-Aware Negative Sampling for Multi-modal Knowledge Graph Embedding</small> | [![IEEE](https://img.shields.io/badge/IEEE-Xplore-blue)](https://ieeexplore.ieee.org/document/10191314) | MANS negative sampling. |
| **Aspect-aware KG** | <small>AspectMMKG: A Multi-modal Knowledge Graph with Aspect-aware Entities</small> | [![ACM](https://img.shields.io/badge/ACM-Link-blue)](https://dl.acm.org/doi/10.1145/3583780.3615023) | AspectMMKG. |
| **Fact Verification** | <small>Multi-source Knowledge Enhanced Graph Attention Networks for Multimodal Fact Verification</small> | [![ResearchGate](https://img.shields.io/badge/ResearchGate-Link-green)](https://www.researchgate.net/publication/382271719_Multi-source_Knowledge_Enhanced_Graph_Attention_Networks_for_Multimodal_Fact_Verification) | Multi-source graph attention. |
| **Entity Tagging** | <small>Multimodal Entity Tagging with Multimodal Knowledge Base</small> | [![OpenReview](https://img.shields.io/badge/OpenReview-Link-blue)](https://openreview.net/forum?id=878288e498) | Multimodal Entity Tagging (MET). |
| **Noise-powered** | <small>Noise-powered Multi-modal Knowledge Graph Representation Framework</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2025.coling-main.11/) | SNAG method. |
| **Spatial Expl** | <small>Spatially Grounded Explanations in Vision-Language Models for Document VQA</small> | [![arXiv](https://img.shields.io/badge/arXiv-2507.12490-b31b1b.svg)](https://arxiv.org/abs/2507.12490) | Spatially grounded explanations. |
| **Structure Guided** | <small>SGMEA: Structure-Guided Multimodal Entity Alignment</small> | [![ACL](https://img.shields.io/badge/ACL-COLING-red)](https://aclanthology.org/2025.coling-main.525/) | SGMEA. |
| **Spoken MRC** | <small>VlogQA: Task, Dataset, and Baseline Models for Vietnamese Spoken-Based MRC</small> | [![ACL](https://img.shields.io/badge/ACL-EACL-red)](https://aclanthology.org/2024.eacl-long.79/) | VlogQA. |
| **Multimodal Doc** | <small>T2KG: Transforming Multimodal Document to Knowledge Graph</small> | [![ACL](https://img.shields.io/badge/ACL-RANLP-red)](https://aclanthology.org/2023.ranlp-1.43/) | T2KG. |
| **Visual Rel** | <small>Visual Relationship Detection With Visual-Linguistic Knowledge</small> | [![IEEE](https://img.shields.io/badge/IEEE-Xplore-blue)](https://ieeexplore.ieee.org/document/9366367) | Visual relationship detection. |
| **Zero-Shot Rel** | <small>Zero-Shot Relational Learning for Multimodal Knowledge Graphs</small> | [![arXiv](https://img.shields.io/badge/arXiv-2404.06220-b31b1b.svg)](https://arxiv.org/abs/2404.06220) | Zero-shot relational learning. |


---

## :telescope: Future Horizons

This repository chronicles a pivotal shift in Medical AI from static **Information Retrieval** to dynamic **Clinical Reasoning**. As the field matures, three defining paradigms are emerging:

* **The Agentic Shift:** We are moving beyond passive chatbots to proactive **Copilots**. Systems like *MedAgents* and *Dr. Copilot* demonstrate that the future lies in LLMs that can plan, self-correct, and execute multi-step diagnostic workflows.
* **Multimodal Synergy:** True clinical understanding requires seeing as well as reading. The next generation of models achieves **Multimodal Fluency**, seamlessly synthesizing pixel-level evidence (X-rays, Pathology) with textual knowledge (Guidelines, EHRs).
* **Democratization & Safety:** As capabilities scale, so must responsibility. The focus is pivoting toward **Privacy-Preserving RAG** and **Linguistic Equity**, ensuring that life-saving AI is robust, compliant, and accessible across all languages from English to Amharic.
