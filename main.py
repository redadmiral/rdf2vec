from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec import RDF2VecTransformer

# Define the label predicates, all triples with these predicates
# will be excluded from the graph
label_predicates = []
kg = KG(location="samples/mutag/mutag.owl", label_predicates=label_predicates)

walkers = [RandomWalker(4, 5, UniformSampler())]

transformer = RDF2VecTransformer(walkers=[walkers], sg=1)
# Entities should be a list of URIs that can be found in the Knowledge Graph
embeddings = transformer.fit_transform(kg, entities)
