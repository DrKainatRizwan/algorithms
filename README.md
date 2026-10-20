# Algorithms

A collection of from-scratch implementations of algorithms in optimisation, adversarial ML, reinforcement learning, quantum computing and cryptography, each with tests, a runnable demo and benchmarks.

**Author:** Dr. Kainat Rizwan

## Projects

<!-- PROJECTS:START -->
| Date | Project | Description |
|------|---------|-------------|
| 2026-10-20 | [Q-Learning Traffic Signal Controller on a Simulated Intersection Network](q-learning-traffic-signal-control/) | Queue-based grid traffic simulator with Q-learning, Double Q and Expected SARSA signal agents compared against fixed-time, longest-queue and max-pressure control. |
| 2026-10-19 | [Fast Fourier Transform Library with Polynomial Multiplication and Big-Integer Arithmetic](fft-polynomial-bigint-multiplication/) | Radix-2 FFT, Bluestein, three-prime NTT, polynomial arithmetic with Newton division, wildcard matching and FFT-based big-integer multiplication with benchmarks. |
| 2026-10-19 | [Malware Byte-to-Image Feature Extraction and k-NN Family Classifier](malware-byte-image-knn/) | Byte-to-image conversion, GLCM/LBP/entropy features and a k-NN classifier with KD-tree on synthetic malware families, with evasion robustness tests. |
| 2026-10-18 | [B+ Tree Storage Engine with Range Scans and Bulk Loading](b-plus-tree-storage-engine/) | B+ tree with borrow/merge deletion, range scans, bottom-up bulk loading, page serialisation, an LRU buffer-pool simulator and a WAL-backed key-value store. |
| 2026-10-18 | [Multi-Armed Bandits Lab: Epsilon-Greedy, UCB1, Thompson Sampling and EXP3](multi-armed-bandits-lab/) | Epsilon-greedy, softmax, UCB1, UCB-Tuned, Thompson sampling, EXP3 and sliding-window UCB with stochastic, switching and adversarial environments and regret bounds. |
| 2026-10-17 | [Hungarian Algorithm and Auction Algorithm for Optimal Assignment](hungarian-auction-assignment/) | Hungarian and epsilon-scaling auction solvers with dual certificates, bottleneck assignment, gated tracking association and benchmarks. |
| 2026-10-17 | [Shamir Secret Sharing and Secure Multi-Party Computation Sum Protocol](shamir-secret-sharing-mpc-sum/) | Shamir sharing with Feldman verification, Berlekamp-Welch error correction, share refresh, dropout-tolerant secure sums and a BGW arithmetic engine. |
| 2026-10-16 | [Interval Trees and Segment Trees for Computational Geometry Queries](interval-segment-tree-geometry/) | Centred interval tree, augmented interval treap and segment trees powering sweeps for rectangle union area, intersection reporting, window queries and skylines. |
| 2026-10-16 | [Fast Gradient Sign and PGD Adversarial Attacks with Adversarial Training in NumPy](fgsm-pgd-adversarial-training/) | FGSM, PGD (L-inf/L2, targeted), MI-FGSM and adversarial training on a NumPy MLP with exact input gradients and robustness benchmarks. |
| 2026-10-15 | [A*, Jump Point Search and D* Lite Pathfinding Benchmark on Large Grids](astar-jps-dstar-lite-grids/) | A* (plain and weighted), Jump Point Search and incremental D* Lite on 8-connected grids with map generators, navigation in unknown terrain and benchmarks. |
| 2026-10-15 | [Zero-Trust Access Policy Engine with Continuous Trust Scoring and Drift Detection](zero-trust-policy-engine/) | Attribute-based policy engine with log-odds trust decay, Page-Hinkley/CUSUM drift detection, PSI and KS tests, and a hash-chained audit log. |
| 2026-10-14 | [Elliptic Curve Cryptography from Scratch: ECDH and ECDSA over Finite Fields](elliptic-curve-ecdh-ecdsa/) | Elliptic curve arithmetic with Jacobian coordinates and wNAF, ECDH, ECIES, RFC 6979 ECDSA with key recovery, and discrete-log and nonce-reuse attacks. |
| 2026-10-14 | [Policy Gradient (REINFORCE and A2C) in NumPy for CartPole-Style Control](policy-gradient-reinforce-a2c/) | REINFORCE with baselines and GAE-based A2C with hand-written backprop, a CartPole simulator, vectorised environments and benchmarks. |
| 2026-10-13 | [Fenwick Trees, Sparse Tables and Sqrt Decomposition: Range Query Benchmark Suite](fenwick-sparse-table-sqrt/) | Fenwick trees (point, range, 2-D), sparse and disjoint sparse tables, sqrt decomposition and Mo's algorithm with benchmarks. |
| 2026-10-13 | [Variational Quantum Eigensolver Simulator for Small Molecular Hamiltonians](variational-quantum-eigensolver-molecular/) | Variational quantum eigensolver with a reduced H2 Hamiltonian, spin chains, parameter-shift gradients, five optimisers, shot noise and ADAPT-VQE. |
| 2026-10-12 | [Raft Consensus Algorithm Simulator with Leader Election and Log Replication](raft-consensus-leader-election/) | Deterministic Raft simulator with leader election, log replication, crash and partition injection, and runtime safety-invariant checking. |
| 2026-10-12 | [Anomaly-Based Intrusion Detection with Isolation Forest Implemented from Scratch](isolation-forest-intrusion-detection/) | Isolation Forest and Extended Isolation Forest with a flow-level intrusion detector, synthetic attack traffic and per-flow explanations. |
| 2026-10-11 | [Simulated Annealing vs Genetic Algorithm vs Tabu Search for the Travelling Salesman Problem](tsp-annealing-genetic-tabu/) | Simulated annealing, genetic algorithm (OX/ERX) and tabu search with 2-opt, Or-opt and Held-Karp baselines, benchmarked on random TSP instances. |
| 2026-10-11 | [Graph Neural Network Message Passing from Scratch for Node Classification](graph-neural-network-message-passing/) | GCN, GraphSAGE and multi-head GAT layers with hand-written backprop, gradient checks, and over-smoothing analysis on block-model graphs. |
| 2026-10-10 | [Bloom Filter, Count-Min Sketch and HyperLogLog Streaming Analytics Library](bloom-filter-count-min-hyperloglog/) | Bloom, counting and scalable Bloom filters, Count-Min sketch with heavy hitters and HyperLogLog, with error benchmarks on Zipf streams. |
| 2026-10-10 | [Monte Carlo Tree Search Engine with UCT for Connect-Four](monte-carlo-tree-search-connect-four/) | UCT Monte Carlo Tree Search with a solver and tactical rollouts on a bitboard Connect-Four, benchmarked against alpha-beta. |
| 2026-10-09 | [Homomorphic Encryption Playground: Paillier Cryptosystem and Private Aggregation](paillier-homomorphic-aggregation/) | Paillier cryptosystem with CRT decryption, fixed-point encoding and private sum, gradient averaging, voting and masking protocols. |
| 2026-10-09 | [Max-Flow Min-Cut Suite: Edmonds-Karp, Dinic and Push-Relabel with Benchmarks](max-flow-min-cut-suite/) | Edmonds-Karp, Dinic and push-relabel max-flow solvers with min-cut certificates, matching reductions and benchmarks. |
| 2026-10-08 | [Differential Privacy Toolkit: Laplace, Gaussian and Exponential Mechanisms with Privacy Accounting](differential-privacy-toolkit-accounting/) | Laplace, Gaussian, exponential and sparse-vector mechanisms with basic, advanced and Renyi-DP accounting and a DP-SGD trainer. |
| 2026-10-08 | [Deep Q-Learning from Scratch in NumPy with Experience Replay on Gridworld Mazes](deep-q-learning-gridworld-mazes/) | NumPy DQN with Double-DQN targets, uniform and prioritised replay and a BFS-verified maze environment. |
| 2026-10-07 | [Suffix Automaton and Suffix Array Toolkit for Large-Scale String Analysis](suffix-automaton-array-toolkit/) | Suffix array, LCP, suffix automaton, BWT and substring analyses with pattern search and benchmarks. |
| 2026-10-07 | [Quantum Circuit State-Vector Simulator with Grover's Search and QFT](quantum-circuit-statevector-grover-qft/) | NumPy state-vector simulator with controlled gates, measurement, QFT, Grover search and phase estimation. |
| 2026-10-06 | [Federated Averaging Simulator with Byzantine-Robust Aggregation](federated-averaging-byzantine-robust/) | FedAvg simulation with Krum, Multi-Krum, trimmed mean, median and geometric median defences against four Byzantine attacks. |
| 2026-10-06 | [Persistent Segment Tree Library with Time-Travel Range Queries](persistent-segment-tree-time-travel/) | Versioned segment trees with lazy range add, range k-th queries and timestamped history queries. |
<!-- PROJECTS:END -->
