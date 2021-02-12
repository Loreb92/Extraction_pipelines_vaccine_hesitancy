This folder contains the interaction network (interaction_network.csv) and three other files as an example of the input files for the extraction pipelines (only the names of each colomn are shown).

The interaction network is provided by the adjacency list. It is a CSV file with five columns:
- source : (str) the id of the source node
- target : (str) the id of the target node
- weight : (int) the weight of the link
- leaning_source : (int) the vaccination schedule followed by the source user (+1 for recommended schedule, -1 for alternative schedule, missing value if the vaccination behavior is unknown)
- leaning_target : (int) the vaccination schedule followed by the source user (+1 for recommended schedule, -1 for alternative schedule, missing value if the vaccination behavior is unknown)
