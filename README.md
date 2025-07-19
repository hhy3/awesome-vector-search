# Awesome Vector Search

This repo collects papers, documents, and codes about vector search for anyone who wants to research it. We are continuously improving the project. Welcome to PR the works (papers, repositories) that the repo misses.

## Papers

### 2025

- [[VLDB](https://arxiv.org/abs/2507.10070)] Breaking the Storage-Compute Bottleneck in Billion-Scale ANNS: A GPU-Driven Asynchronous I/O Framework [**`Hardware`**] [[code](https://github.com/Tinkerver/ann_search)] ![GitHub Repo stars](https://img.shields.io/github/stars/Tinkerver/ann_search)

- [[arXiv](https://arxiv.org/abs/2507.06653)] Towards Efficient and Scalable Distributed Vector Search with RDMA [**`Hardware`**]

- [[arXiv](https://arxiv.org/abs/2507.04256)] OneDB: A Distributed Multi-Metric Data Similarity Search System [**`Multimodel`**]

- [[ISCA](https://arxiv.org/abs/2506.16444)] REIS: A High-Performance and Energy-Efficient Retrieval System with In-Storage Processing [**`Hardware`**]

- [[KDD](https://arxiv.org/abs/2506.15986)] Empowering Graph-based Approximate Nearest Neighbor Search with Adaptive Awareness Capabilities [**`Graph`**]

- [[SIGMOD](https://dl.acm.org/doi/10.1145/3725401)] Efficient Dynamic Indexing for Range Filtered Approximate Nearest Neighbor Search [**`Filter`**] [[code](https://github.com/CUHK-DBGroup/RangePQ)] ![GitHub Repo stars](https://img.shields.io/github/stars/CUHK-DBGroup/RangePQ)

- [[SIGMOD](https://dl.acm.org/doi/10.1145/3725399)] DIGRA: A Dynamic Graph Indexing for Approximate Nearest Neighbor Search with Range Filter [**`Filter`**] 

- [[SIGMOD](https://dl.acm.org/doi/10.1145/3725325)] MIRAGE-ANNS: Mixed Approach Graph-based Indexing for Approximate Nearest Neighbor Search [**`Graph`**] [[code](https://github.com/dsg-uwaterloo/mirage)] ![GitHub Repo stars](https://img.shields.io/github/stars/dsg-uwaterloo/mirage)

- [[arXiv](https://arxiv.org/abs/2506.13144)] EnhanceGraph: A Continuously Enhanced Graph-based Index for High-dimensional Approximate Nearest Neighbor Search [**`Graph`**] [[code](https://github.com/alipay/vsag)] ![GitHub Repo stars](https://img.shields.io/github/stars/alipay/vsag)

- [[VecDB](https://openreview.net/forum?id=6AEsfCLRm3)] DistributedANN: Efficient Scaling of a Single DiskANN Graph Across Thousands of Computers [**`Graph`**] [**`Distributed`**]

- [[arXiv](https://arxiv.org/abs/2506.08276)] LEANN: A Low-Storage Vector Index [**`Hardware`**]

- [[ODSI](https://arxiv.org/abs/2506.03437)] Quake: Adaptive Indexing for Vector Search [**`Streaming`**] [[code](https://github.com/marius-team/quake)] ![GitHub Repo stars](https://img.shields.io/github/stars/marius-team/quake)

- [[SIGMOD](https://arxiv.org/abs/2506.00812)] VecFlow: A High-Performance Vector Data Management System for Filtered-Search on GPUs [**`Filter`**] [**`Hardware`**] [[code](https://github.com/Supercomputing-System-AI-Lab/VecFlow)] ![GitHub Repo stars](https://img.shields.io/github/stars/Supercomputing-System-AI-Lab/VecFlow)

- [[arXiv](https://arxiv.org/abs/2505.17810)] VIBE: Vector Index Benchmark for Embeddings [**`Benchmark`**] [[code](https://github.com/vector-index-bench/vibe)] ![GitHub Repo stars](https://img.shields.io/github/stars/vector-index-bench/vibe)

- [[arXiv](https://arxiv.org/abs/2505.16096)] Cosmos: A CXL-Based Full In-Memory System for Approximate Nearest Neighbor Search [**`Hardware`**]

- [[arXiv](https://arxiv.org/abs/2505.12524)] HAKES: Scalable Vector Database for Embedding Search Service [**`Streaming`**] [[code](https://github.com/nusdbsystem/HAKES-Search)] ![GitHub Repo stars](https://img.shields.io/github/stars/nusdbsystem/HAKES-Search)

- [[arXiv](https://arxiv.org/abs/2505.07621)] Bang for the Buck: Vector Search on Cloud CPUs [**`Experiment`**]

- [[arXiv](https://arxiv.org/abs/2505.06501)] Survey of Filtered Approximate Nearest Neighbor Search over the Vector-Scalar Hybrid Data [**`Survey`**] [**`Filter`**] [[code](https://github.com/lyj-fdu/FANNS)] ![GitHub Repo stars](https://img.shields.io/github/stars/lyj-fdu/FANNS)

- [[arXiv](https://arxiv.org/abs/2505.02922)] RetroInfer: A Vector-Storage Approach for Scalable Long-Context LLM Inference [**`KVCache`**] [[code](https://github.com/microsoft/RetrievalAttention)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/RetrievalAttention)

- [[arXiv](https://arxiv.org/abs/2504.20461)] Efficient Graph-Based Approximate Nearest Neighbor Search Achieving: Low Latency Without Throughput Loss [**`Graph`**] [**`Serving`**]

- [[arXiv](https://arxiv.org/abs/2504.20018)] MINT: Multi-Vector Search Index Tuning [**`Tuning`**]

- [[arXiv](https://arxiv.org/abs/2504.19874)] TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate [**`Compression`**]

- [[SIGIR](https://arxiv.org/abs/2504.14861)] Stitching Inner Product and Euclidean Metrics for Topology-aware Maximum Inner Product Search [**`MIPS`**] [[code](https://github.com/ZJU-DAILY/MAG)] ![GitHub Repo stars](https://img.shields.io/github/stars/ZJU-DAILY/MAG)

- [[SIGMOD](https://arxiv.org/abs/2504.10326)] AlayaDB: The Data Foundation for Efficient and Effective Long-context LLM Inference [**`KVCache`**]

- [[SIGMOD](https://arxiv.org/abs/2504.05573)] MicroNN: An On-device Disk-resident Updatable Vector Database [**`Hardware`**]

- [[arXiv](https://arxiv.org/abs/2504.04018)] ESG: Elastic Graphs for Range-Filtering Approximate 𝑘-Nearest Neighbor Search [**`Filter`**]

- [[arXiv](https://arxiv.org/abs/2503.17911)] VSAG: An Optimized Search Framework for Graph-based Approximate Nearest Neighbor Search [**`Framework`**] [[code](https://github.com/antgroup/vsag)] ![GitHub Repo stars](https://img.shields.io/github/stars/antgroup/vsag)

- [[VLDB](https://arxiv.org/abs/2503.06882)] Maximum Inner Product is Query-Scaled Nearest Neighbor [**`MIPS`**] [[code](https://github.com/ZJU-DAILY/PSP)] ![GitHub Repo stars](https://img.shields.io/github/stars/ZJU-DAILY/PSP)

- [[SIGMOD](https://arxiv.org/abs/2503.04422)] PDX: A Data Layout for Vector Similarity Search [**`Hardware`**] [[code](https://github.com/cwida/PDX)] ![GitHub Repo stars](https://img.shields.io/github/stars/cwida/PDX)

- [[ICML](https://openreview.net/forum?id=JnXbUKtLzz&noteId=Xgl1OzI5nG)] Sort Before You Prune: Improved Worst-Case Guarantees of the DiskANN Family of Graphs [**`Graph`**]

- [[WWW'25](https://arxiv.org/abs/2502.20695)] Scalable Overload-Aware Graph-Based Index Construction for 10-Billion-Scale Vector Similarity Search [**`Graph`**]

- [[SIGMOD](https://arxiv.org/abs/2502.18113)] Accelerating Graph Indexing for ANNS on Modern CPUs [**`Graph`**] [[code](https://github.com/ZJU-DAILY/HNSW-Flash)] ![GitHub Repo stars](https://img.shields.io/github/stars/ZJU-DAILY/HNSW-Flash)

- [[arXiv](https://arxiv.org/abs/2502.13826)] In-Place Updates of a Graph Index for Streaming Approximate Nearest Neighbor Search [**`Streaming`**]

- [[ICML](https://openreview.net/forum?id=dmN2fQ3woH&noteId=dQULdIbto1)] Graph-Based Algorithms for Diverse Similarity Search [**`Filter`**] [**`Graph`**] [[code](https://github.com/microsoft/DiskANN/tree/diversity)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DiskANN)

- [[arXiv](https://arxiv.org/abs/2502.08246)] Inference-time sparse attention with asymmetric indexing [**`KVCache`**]

- [[SIGMOD](https://arxiv.org/abs/2502.07343)] DEG: Efficient Hybrid Vector Search Using the Dynamic Edge Navigation Graph [**`Filter`**] [[code](https://github.com/Heisenberg-Yin/DEG)] ![GitHub Repo stars](https://img.shields.io/github/stars/Heisenberg-Yin/DEG)

- [[arXiv](https://arxiv.org/abs/2502.06766)] Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs [**`KVCache`**] [[code](https://github.com/ryansynk/topk-decoding)] ![GitHub Repo stars](https://img.shields.io/github/stars/ryansynk/topk-decoding)

- [[arXiv](https://arxiv.org/abs/2502.06163)] Scalable k-Means Clustering for Large k via Seeded Approximate Nearest-Neighbor Search [**`Clustering`**] [[code](https://github.com/jacketsj/mopbucket)] ![GitHub Repo stars](https://img.shields.io/github/stars/jacketsj/mopbucket)

- [[SIGMOD](https://arxiv.org/abs/2502.05575)] Graph-Based Vector Search: An Experimental Evaluation of the State-of-the-Art [**`Graph`**]

- [[arXiv](https://arxiv.org/abs/2501.10479)] Lossless Compression of Vector IDs for Approximate Nearest Neighbor Search [**`Compression`**] [[code](https://github.com/facebookresearch/vector_db_id_compression)] ![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/vector_db_id_compression)

- [[ICLR](https://arxiv.org/abs/2501.03078)] Qinco2: Vector Compression and Search with Improved Implicit Neural Codebooks [**`Compression`**] [[code](https://github.com/facebookresearch/Qinco)] ![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/Qinco)

### 2024

- [[VLDB](https://arxiv.org/abs/2412.02448)] UNIFY: Unified Index for Range Filtered Approximate Nearest Neighbors Search [**`Filter`**] [[code](https://github.com/sjtu-dbgroup/UNIFY)] ![GitHub Repo stars](https://img.shields.io/github/stars/sjtu-dbgroup/UNIFY)

- [[SIGMOD](https://dl.acm.org/doi/10.1145/3698822)] Navigating Labels and Vectors: A Unified Approach to Filtered Approximate Nearest Neighbor Search [**`Filter`**] [[code](https://github.com/YZ-Cai/Unified-Navigating-Graph)] ![GitHub Repo stars](https://img.shields.io/github/stars/YZ-Cai/Unified-Navigating-Graph)

- [[VLDB](https://arxiv.org/abs/2411.17229)] Efficient Data-aware Distance Comparison Operations for High-Dimensional Approximate Nearest Neighbor Search [**`DCO`**] [[code](https://github.com/Ur-Eine/DADE)] ![GitHub Repo stars](https://img.shields.io/github/stars/Ur-Eine/DADE)

- [[SIGMOD](https://arxiv.org/abs/2411.14754)] Subspace Collision: An Efficient and Accurate Framework for High-dimensional Approximate Nearest Neighbor Search [**`DCO`**]

- [[SIGMOD](https://arxiv.org/abs/2411.12229)] SymphonyQG: Towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search [**`Graph`**] [[code](https://github.com/gouyt13/SymphonyQG)] ![GitHub Repo stars](https://img.shields.io/github/stars/gouyt13/SymphonyQG)

- [[arXiv](https://arxiv.org/abs/2410.20381)] Efficient and Effective Retrieval of Dense-Sparse Hybrid Vectors using Graph-based Approximate Nearest Neighbor Search [**`Hybrid`**] [**`Graph`**]

- [[NeurIPS](https://arxiv.org/abs/2410.18926)] LoRANN: Low-Rank Matrix Factorization for Approximate Nearest Neighbor Search [**`Compression`**] [[code](https://github.com/ejaasaari/lorann)] ![GitHub Repo stars](https://img.shields.io/github/stars/ejaasaari/lorann)

- [[FAST](https://arxiv.org/abs/2409.16576)] FusionANNS: An Efficient CPU/GPU Cooperative Processing Architecture for Billion-scale Approximate Nearest Neighbor Search

- [[arXiv](https://arxiv.org/abs/2409.10516)] RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval [**`KVCache`**]

- [[SIGMOD](https://arxiv.org/abs/2409.09913)] Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search [**`Compression`**] [[code](https://github.com/VectorDB-NTU/Extended-RaBitQ)] ![GitHub Repo stars](https://img.shields.io/github/stars/VectorDB-NTU/Extended-RaBitQ)

- [[SIGMOD](https://arxiv.org/abs/2409.02571)] iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search [**`Filter`**] [[code](https://github.com/YuexuanXu7/iRangeGraph)] ![GitHub Repo stars](https://img.shields.io/github/stars/YuexuanXu7/iRangeGraph)

- [[VLDB](https://www.arxiv.org/abs/2408.13899)] Steiner -Hardness: A Query Hardness Measure for Graph-Based ANN Indexes [**`Graph`**]

- [[VLDB](https://arxiv.org/abs/2408.08933)] RoarGraph: A Projected Bipartite Graph for Efficient Cross-Modal Approximate Nearest Neighbor Search [**`Graph`**] [[code](https://github.com/matchyc/RoarGraph)] ![GitHub Repo stars](https://img.shields.io/github/stars/matchyc/RoarGraph)

- [[VLDB](https://cs.purdue.edu/homes/csjgwang/pubs/VLDB24_SingleStoreVec.pdf)] SingleStore-V: An Integrated Vector Database System in SingleStore [**`Database`**]

- [[CIKM](https://arxiv.org/abs/2408.02937)] A Real-Time Adaptive Multi-Stream GPU System for Online Approximate Nearest Neighborhood Search [**`Hardware`**] [**`Streaming`**]

- [[ATC](https://www.usenix.org/conference/atc24/presentation/tian)] Scalable Billion-point Approximate Nearest Neighbor Search Using SmartSSDs [**`Hardware`**]

- [[arXiv](https://arxiv.org/abs/2406.19651)] CANDY: A Benchmark for Continuous Approximate Nearest Neighbor Search with Dynamic Data Ingestion [**`Benchmark`**] [**`Streaming`**] [[code](https://github.com/intellistream/candy)] ![GitHub Repo stars](https://img.shields.io/github/stars/intellistream/candy)

- [[SIGMOD](https://dl.acm.org/doi/10.1145/3654990)] Vexless: A Serverless Vector Data Management System Using Cloud Functions [**`Serverless`**]

- [[SIDMOG](https://arxiv.org/abs/2405.12497)] RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search [**`Compression`**] [[code](https://github.com/gaoj0017/RaBitQ)] ![GitHub Repo stars](https://img.shields.io/github/stars/gaoj0017/RaBitQ)

- [[VLDB](https://arxiv.org/abs/2406.10938)] DET-LSH: A Locality-Sensitive Hashing Scheme with Dynamic Encoding Tree for Approximate Nearest Neighbor Search [**`LSH`**] [[code](https://github.com/WeiJiuQi/DET-LSH)] ![GitHub Repo stars](https://img.shields.io/github/stars/WeiJiuQi/DET-LSH)


- [[ICDE](https://arxiv.org/abs/2404.16322)] Effective and General Distance Computation for Approximate Nearest Neighbor Search [**`DCO`**] [[code](https://github.com/mingyu-hkustgz/Res-Infer)] ![GitHub Repo stars](https://img.shields.io/github/stars/mingyu-hkustgz/Res-Infer)

- [[arXiv](https://arxiv.org/abs/2404.06004)] AiSAQ: All-in-Storage ANNS with Product Quantization for DRAM-free Information Retrieval [**`Hardware`**] [[code](https://github.com/kioxia-jp/aisaq-diskann)] ![GitHub Repo stars](https://img.shields.io/github/stars/kioxia-jp/aisaq-diskann)

- [[SIGMOD](https://arxiv.org/abs/2404.00966)] GTS: GPU-based Tree Index for Fast Similarity Search [**`Hardware`**] [[code](https://github.com/ZJU-DAILY/GTS)] ![GitHub Repo stars](https://img.shields.io/github/stars/ZJU-DAILY/GTS)

- [[arXiv](https://arxiv.org/abs/2403.13491)] Dimensionality-Reduction Techniques for Approximate Nearest Neighbor Search: A Survey and Evaluation [**`Survey`**]

- [[SIGMOD](https://arxiv.org/abs/2403.04871)] ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data [**`Filter`**]

- [[ICML](https://arxiv.org/pdf/2402.11354)] Probabilistic Routing for Graph-Based Approximate Nearest Neighbor Search [**`Graph`**] [[code](https://github.com/ICML2024-code/PEOs)] ![GitHub Repo stars](https://img.shields.io/github/stars/ICML2024-code/PEOs)

- [[ICML](https://arxiv.org/abs/2402.00943)] Approximate Nearest Neighbor Search with Window Filters [**`Filter`**] [[code](https://github.com/JoshEngels/RangeFilteredANN)] ![GitHub Repo stars](https://img.shields.io/github/stars/JoshEngels/RangeFilteredANN)

- [[ICML](https://arxiv.org/abs/2401.14732)] Residual Quantization with Implicit Neural Codebooks [**`Compression`**] [[code](https://github.com/facebookresearch/Qinco)] ![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/Qinco)

- [[arXiv](https://arxiv.org/abs/2401.11324)] BANG: Billion-Scale Approximate Nearest Neighbor Search using a Single GPU [**`Hardware`**] [[code](https://github.com/karthik86248/BANG-Billion-Scale-ANN)] ![GitHub Repo stars](https://img.shields.io/github/stars/karthik86248/BANG-Billion-Scale-ANN)

- [[arXiv](https://arxiv.org/abs/2401.07119)] Curator: Efficient Indexing for Multi-Tenant Vector Databases [**`Multitenancy`**]

- [[SIGMOD](https://arxiv.org/abs/2401.02116)] Starling: An I/O-Efficient Disk-Resident Graph Index Framework for High-Dimensional Vector Similarity Search on Data Segment [**`Hardware`**] [**`Graph`**] [[code](https://github.com/zilliztech/starling)] ![GitHub Repo stars](https://img.shields.io/github/stars/zilliztech/starling)

### 2023

- [[NeurIPS](https://papers.nips.cc/paper_files/paper/2023/hash/32e41d6b0a51a63a9a90697da19d235d-Abstract-Conference.html)] An Efficient and Robust Framework for Approximate Nearest Neighbor Search with Attribute Constraint [**`Filter`**]

- [[ICDE](https://arxiv.org/abs/2312.06397)] MUST: An Effective and Scalable Framework for Multimodal Search of Target Modality [**`Multimodal`**] [[code](https://github.com/ZJU-DAILY/MUST)] ![GitHub Repo stars](https://img.shields.io/github/stars/ZJU-DAILY/MUST)

- [[MM](https://arxiv.org/abs/2310.20419)] Relative NN-Descent: A Fast Index Construction for Graph-Based Approximate Nearest Neighbor Search [**`Graph`**] [[code](https://github.com/mti-lab/rnn-descent)] ![GitHub Repo stars](https://img.shields.io/github/stars/mti-lab/rnn-descent)

- [[VLDB](https://arxiv.org/abs/2310.09949)] Chameleon: a Heterogeneous and Disaggregated Accelerator System for Retrieval-Augmented Language Models [**`Hardware`**] [[code](https://github.com/fpgasystems/Chameleon-RAG-Acceleration)] ![GitHub Repo stars](https://img.shields.io/github/stars/fpgasystems/Chameleon-RAG-Acceleration)

- [[WWW](https://dl.acm.org/doi/10.1145/3543507.3583552)] Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters [**`Filter`**]

- [[ATC](https://www.usenix.org/conference/atc23/presentation/jang)] CXL-ANNS: Software-Hardware Collaborative Memory Disaggregation and Computation for Billion-Scale Approximate Nearest Neighbor Search [**`Hardware`**] [**`Distributed`**]

- [[SOSP](https://arxiv.org/abs/2410.14452)] SPFresh: Incremental In-Place Update for Billion-Scale Vector Search [**`Streaming`**]

- [[OSDI](https://www.usenix.org/conference/osdi23/presentation/zhang-qianxi)] VBASE: Unifying Online Vector Similarity Search and Relational Queries via Relaxed Monotonicity [**`Filter`**] [[code](https://github.com/microsoft/MSVBASE)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/MSVBASE)