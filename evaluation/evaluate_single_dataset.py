import json 
from evaluate_pred import (
    convert_event_to_tuple,
    trigger_f1, argument_f1,
    convert_entity_to_tuple, entity_f1)
import argparse

def evaluate(gold_file, pred_file, ent_pred=True, evt_pred=True, lang='en'):

    with open(gold_file, encoding="utf-8") as f:
        gold = json.load(f)

    with open(pred_file, encoding="utf-8") as f:
        preds = json.load(f)


    g = sorted([sent['sent_id'] for sent in gold])#sorted(tgold.keys())
    p = sorted([sent['sent_id'] for sent in preds]) #sorted(tpreds.keys())

    if g != p:
        print("Missing some sentences!")
        return 0.0, 0.0, 0.0
    
    if evt_pred:
    
        tgold_trigger = dict([(s["sent_id"], convert_event_to_tuple(s, lang=lang)[0]) for s in gold])
        tpreds_trigger = dict([(s["sent_id"], convert_event_to_tuple(s, lang=lang)[0]) for s in preds])

        tgold_argument = dict([(s["sent_id"], convert_event_to_tuple(s, lang=lang)[1]) for s in gold])
        tpreds_argument = dict([(s["sent_id"], convert_event_to_tuple(s, lang=lang)[1]) for s in preds])


        trigger_idf = trigger_f1(tgold_trigger, tpreds_trigger, classification=False)
        trigger_cls = trigger_f1(tgold_trigger, tpreds_trigger, classification=True)

        argument_idf = argument_f1(tgold_argument, tpreds_argument, classification=False)
        argument_cls = argument_f1(tgold_argument, tpreds_argument, classification=True)
    
    else:
        trigger_idf = trigger_cls = argument_idf = argument_cls = 0,0,0

    if ent_pred:

        gold_entity_tuples = dict([(s['sent_id'], convert_entity_to_tuple(s, lang=lang)) for s in gold])
        pred_entity_tuples = dict([(s['sent_id'], convert_entity_to_tuple(s, lang=lang)) for s in preds])

        entity_cls = entity_f1(gold_entity_tuples, pred_entity_tuples, classification=True)
    else:
        entity_cls = 0,0,0

    results = {
        'trigger_identification': trigger_idf,
        'trigger_classification': trigger_cls,
        'argument_identification': argument_idf,
        'argument_classification': argument_cls,
        'entity_classification': entity_cls,
        #'relation_classification': relation_cls
    }

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", help="gold json file")
    parser.add_argument("pred_file", help="prediction json file")
    parser.add_argument("--ent_pred", help="entity prediction", default='no', type=str)
    parser.add_argument("--lang", help="entity prediction", default='en', type=str)



    args = parser.parse_args()



    if args.ent_pred == 'no':
        results = evaluate(args.gold_file, args.pred_file, ent_pred=False, lang=args.lang)
    else:
        results = evaluate(args.gold_file, args.pred_file, ent_pred=True, lang=args.lang)

    print(json.dumps(results, indent=2))
    print()
    print(list(results.values()))


if __name__ == "__main__":
    main()