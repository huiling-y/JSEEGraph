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

            top = graph.add_node(top=True)

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
            
            # add triggers as nodes
            for event in native['events']:

                trigger = event['trigger']
                key = tuple(event['trigger'][1])

                #if key in map:
                #    _trigger = map[key]
                #else:
                _trigger = graph.add_node(
                        anchors=anchor(trigger),
                        label='trigger'
                    )
                
                #map[key] = _trigger
                
                graph.add_edge(top.id, _trigger.id, event['event_type'])

                # add trigger-argument edges
                arguments = event['arguments']
                if len(arguments):

                    for argument in arguments:
                        arg_role = argument[-1]
                        key = tuple(argument[1])

                        _argument = map[key]

                        graph.add_edge(_trigger.id, _argument.id, arg_role)
            
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
        return write_evt_ent_graph(graph, input)
    except Exception as error:
        print(f"Problem with decoding sentence {graph.id}")
        raise error


def write_evt_ent_graph(graph, input):

    nodes = {node.id: node for node in graph.nodes}
    assigned = {node.id: False for node in graph.nodes}


    entities = []
    relations = []

    # create events
    events = {}

    for edge in graph.edges:
        if edge.src == 0:
            node = nodes[edge.tgt]
            assigned[node.id] = True
            events[node.id] = {
                'event_type': edge.lab,
                'trigger': [*get_text_span(node, input)],
                'arguments': []
            }
    
    # add event arguments and relations
    for edge in graph.edges:

        if edge.src != 0:
            src_node = nodes[edge.src]
            tgt_node = nodes[edge.tgt]

            # event arguments
            if edge.src in events:

                anchored_text, anchors = get_text_span(tgt_node, input)
                events[edge.src]['arguments'].append([anchored_text, anchors, edge.lab])

                if not assigned[tgt_node.id]:
                    entities.append([anchored_text, anchors, tgt_node.label])
                    assigned[tgt_node.id] = True
            else:

                src_anchored_text, src_anchors = get_text_span(src_node, input)
                tgt_anchored_text, tgt_anchors = get_text_span(tgt_node, input)

                if not assigned[src_node.id]:
                    entities.append([src_anchored_text, src_anchors, src_node.label])
                    assigned[src_node.id] = True
                if not assigned[src_node.id]:
                    entities.append([tgt_anchored_text, tgt_anchors, tgt_node.label])
                    assigned[tgt_node.id] = True

                relations.append([src_anchors, tgt_anchors, edge.lab])
    
    for node_id, _ass in assigned.items():
        if not _ass and node_id != 0 and nodes[node_id].label != 'trigger':
            anchored_text, anchors = get_text_span(nodes[node_id], input)
            entities.append([anchored_text, anchors, nodes[node_id].label])
    
    sentence = {
        'sent_id': graph.id,
        'text': input,
        'entities': entities,
        'relations': relations,
        'events': list(events.values())
    }

    return sentence




    