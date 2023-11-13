import json 
import sys 

from graph import Graph

def read(fp, text=None):
    def anchor(node):
        anchors = list()
        for string in node[1]:
            string = string.split(":")
            anchors.append({"from": int(string[0]), "to": int(string[1])})
        return anchors
    
    for native in json.load(fp):
        map = dict()

        try:
            graph = Graph(native["sent_id"], flavor=1, framework="evt-ent")
            graph.add_input(native["text"])

            #top = graph.add_node(top=True)

            # add entities as nodes
            for entity in native['entities']:

                key = tuple(entity[1])

                if key in map:
                    _entity = map[key]
                else:
                    _entity = graph.add_node(
                        anchors=anchor(entity),
                        label=entity[-1]
                    )
                    map[key] = _entity
                
                #graph.add_edge(top.id, _entity.id)    

            # add relation edges
            for relation in native['relations']:
                key1 = tuple(relation[0])
                key2 = tuple(relation[1])

                _arg1 = map[key1]
                _arg2 = map[key2]

                graph.add_edge(_arg1.id, _arg2.id, relation[-1])

            yield graph, None
        
        except Exception as error:
            print(
                f"codec.evt_ent.read(): ignoring {native}: {error}",
                file=sys.stderr
            )   

def get_text_span(node, text):
    anchored_text = [text[anchor['from']:anchor['to']] for anchor in node.anchors]
    anchors = [f"{anchor['from']}:{anchor['to']}" for anchor in node.anchors]
    return anchored_text, anchors


def write(graph, input):
    try:
        return write_ent_graph(graph, input)
    except Exception as error:
        print(f"Problem with decoding sentence {graph.id}")
        raise error

def write_ent_graph(graph, input):

    nodes = {node.id: node for node in graph.nodes}

    entities = {}
    relations = []

    # create entities

    #for edge in graph.edges:
    #    if edge.src == 0:
    #        node = nodes[edge.tgt]
    for node in graph.nodes:
        anchored_text, anchors = get_text_span(node, input)
        entities[node.id] = [anchored_text, anchors, node.label]
    
    # create relations
    for edge in graph.edges:
        arg1 = entities[edge.src][1]
        arg2 = entities[edge.tgt][1]

        relations.append([arg1, arg2, edge.lab])
    
    sentence = {
        'sent_id': graph.id,
        'text': input,
        'entities': list(entities.values()),
        'relations': relations
    }

    return sentence



        

