import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.training import Example

df = pd.read_csv('name_list.csv')

examples = []
for index, row in df.iterrows():
    text = row['Name']
    doc = nlp.make_doc(text)
    entities = [(0, len(text), 'PERSON')] 
    example = Example.from_dict(doc, {"entities": entities})
    examples.append(example)

ner = spacy.blank("en")
ner.add_pipe("ner")
ner.begin_training()

for itn in range(10):
    losses = {}
    examples_batch = spacy.util.minibatch(examples, size=10)
    for batch in examples_batch:
        ner.update(batch, losses=losses)
    print(losses)

ner.to_disk("custom_ner_model")

nlp = spacy.load("custom_ner_model")

def split_concatenated_name(name):
    pass

text = "Apple is looking at buying U.K. startup for $1 billion. SteveJobs founded Apple."
doc = nlp(text)

names = []
for ent in doc.ents:
    if ent.label_ == "PERSON":
        split_name = split_concatenated_name(ent.text)
        names.append(split_name)

for name in names:
    print(name)