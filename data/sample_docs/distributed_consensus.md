# Distributed Consensus and Raft Protocol

Distributed consensus algorithms ensure that multiple nodes in a distributed system agree on a shared state machine log even in the presence of network partitions and node crashes.

The Raft consensus protocol achieves consensus through an elected leader. The lifecycle consists of three distinct states: Follower, Candidate, and Leader.

Followers remain passive and respond to RPCs from candidates and leaders. If a follower receives no communication over an election timeout period, it transitions to the Candidate state and initiates a leader election by incrementing the current term and requesting votes from peer nodes.

A candidate wins an election if it receives votes from a majority of nodes in the full cluster. Once elected, the Leader manages log replication by appending log entries and broadcasting AppendEntries RPCs to all follower nodes.

In the event of a network split, only the partition containing a strict majority (quorum) of nodes can successfully commit new entries to the replicated log. Minority partitions cannot commit entries, preventing split-brain anomalies and data divergence.
