import pandas as pd
import numpy as np

graph = pd.read_csv("data/dbp_graph.ttl", sep=" ")
graph.columns = ["sub", "pred", "obj", "dot"]
graph = graph.drop(["pred", "dot"], axis = 1)

entities: pd.Series = graph["sub"]
entities = entities.append(graph["obj"])

entities = entities.str.replace("<", "").str.replace(">", "")
entities = entities[entities.str.startswith("http://de.dbpedia.org/resource/")]
entities = entities.unique()

np.save("data/entities.npy",entities)