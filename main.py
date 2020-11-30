from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import RandomWalker, Walker
import numpy as np
import logging

logging.basicConfig(filename="rdf2vec.log", level = logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

# Define the label predicates, all triples with these predicates
# will be excluded from the graph
logging.info("Read in knowledge graph.")
label_predicates = []
kg = KG(location="data/dbp_graph.ttl", label_predicates=label_predicates)

logging.info("Create walkers and transformers.")
walkers = [RandomWalker(4, 5, UniformSampler())]
transformer = RDF2VecTransformer(Word2Vec(sg=1), walkers=walkers)

logging.info("Read in entities.")
# Entities should be a list of URIs that can be found in the Knowledge Graph
entities = list(np.load("data/entities.npy", allow_pickle=True))
logging.info("Calculate embeddings.")
embeddings = transformer.fit_transform(kg, entities)
logging.info("Write embeddings to disk.")
np.save("data/embeddings.npy", embeddings)
logging.info("Finished job.")