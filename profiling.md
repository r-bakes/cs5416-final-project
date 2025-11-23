# profiling

## Experiment 1: Initial Profile

In this experiment, we explore the as-given single node `pipeline.py` performance in order to begin developing a strategy for the passing grade implementation and then future microservice decomposition. In pursuit of this goal, we use:

1. memory-profiler to identify memory usage patterns and a simple custom decorator to track per step times.

More concretely, we do `uv pip install memory-profiler` and then thread the `@profile` decorator at each of the STEP functions in `pipeline.py`. We write a custom `@profile_with_time` function to print a simple time out for each invocation. While execution time is dependent on underlying test hardware, relative execution time metrics per function call are still informative for bottleneck analysis.

We then run `python3 pipeline.py >> exp1_memory_profile.log` to generate a memory profile log with our simple timing data using the `client.py` script.

### Data

From the data pulled into the raw log file we calculate the max and average max across the 6 `client.py` calls per step-function and overall. We make some immediate observations:

1. `_faiss_search_batch` is the most time consuming step, and consumes the most memory.
2. ~4.5 GB is the max memory used across the 6 requests at that step.
3. ~52% of the total processing time was for that step on average as well.

With this in mind, we can estimate an optimal batch size for the initial replication of the monolith across the 3 nodes for the naive project implementation to be validated with actual benchmarking.

- Base memory: ~800 MiB (Python runtime, previous step outputs)
- FAISS index load: ~323 MiB (line 207, shared across batch)
- Per-query search: ~2873 MiB (line 209, conservatively, scales linearly with batch size)
- Total for batch=1: 800 + 323 + 2873 = 3996 MiB

With 16 GB of RAM as our max, and a desire to keep memory consumption at a max of 85%\* our max batch size is:

Calculating optimal batch size:
Available = 16 GB × 0.85 = 13.6 GB
Batch=3: 0.8 + 0.323 + \~(2.873 × 3) = 9.74 GB
Batch=4: 0.8 + 0.323 + \~(2.873 × 4) = 12.3 GB
Batch=5: 0.8 + 0.323 + \~(2.873 × 5) = 15.0 GB -> (exceeds limit)

\*Regarding the 85% limit set. In a production system, we would likely have autoscaling of compute nodes, triggered on some kind of resource consumption metric such as this memory utilization being hit. Technically, the naive pipeline script only has a single worker so we could push this to 4 to maximize resource utilization and we would be guaranteed safety assuming scaling assumptions are conservative enough, however, in a production system we would not have that guarantee of a single worker. Additionally, it provides a buffer if a particular batched of requests all consume an larger memory quantity.

## Experiment 2: Initial split of monolith across 3 nodes

In this experiment we achieve the lowest tier requirements:

1. Modify the pipeline to run across 3 nodes.
2. Opportunistic batching (low hanging fruit!)
3. Evolution of our benchmarking code in experiment 1 to capture latency and throughput.

We will use this data for the third experiment's configurations and to guide initial microservice decomposition strategy.

###

### Functional Requirements

1. Simple distribution across 3 nodes.
2. Fixed batch size.
3. Round-robin scheduling.

### Adhoc Benchmark

## Experiment 3: Initial Split of monolith into microservices
