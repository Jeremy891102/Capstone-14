# Independent final audit — PASS

Read-only review reparsed all 4,672 raw successes, checked exact requests/body hashes, response model/finish/schema, absence of media, ground truth and scores. Verified 2,336 paired targets across 73 videos, all 640 retained bytes identical, and 4,032 new successes.

Independently reproduced video counts 1,402→1,572, no-video counts 620→817, all recording-bootstrap intervals (10,000 draws, seed 20261112), and interaction −1.1558 pp, 95% [−4.1952, +1.7979].

Usage and frozen-rate costs match: input 6,016,986; answer 84,303; reasoning 176,974; completion including reasoning 261,277. Successful-call estimate $14.646742 total / $12.624522 new; failed billing unknown.

All 18 archived failures (17 timeouts, one network-unreachable error) match exact inputs of eventual successes; no failed call was retried more than once. 4,690 total request attempts including retained history. Manual recovery reviews are preserved; the automatic supervisor ceiling remained unchanged.

Published summary/CSVs/raw logs match audited outputs. All saved responses reconstruct byte-exactly; decoded credential-pattern scan clean. Offline reconstruction and complete reanalysis reproduce summary.json exactly. No paid API calls made during audit.

Interpretation is limited correctly: reference context helps without video; there is no evidence its gain is larger with video. This is not equivalence or proof that text fully explains batching gains. Video still adds approximately 32–33 percentage points.
