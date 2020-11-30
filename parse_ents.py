import pandas as pd
import numpy as np
import logging

logging.basicConfig(filename="rdf2vec.log", level = logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logging.info("Create DataFrame...")
graph = pd.read_csv("data/dbp_graph_sample_head.ttl", sep=" ")
graph.columns = ["sub", "pred", "obj", "dot"]
graph = graph.drop(["pred", "dot"], axis = 1)

logging.info("Filter out entities...")
entities: pd.Series = graph["sub"]
entities = entities.append(graph["obj"])

logging.info("Replace pointy brackets...")
entities = entities.str.replace("<", "").str.replace(">", "")
logging.info("Filter non-dpb resources...")
entities = entities[entities.str.startswith("http://de.dbpedia.org/resource/")]
logging.info("Make unique...")
entities = entities.unique()
logging.info("Write to disk...")
np.save("data/entities.npy",entities)
logging.info("Finished job.")