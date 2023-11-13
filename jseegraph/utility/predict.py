import os
import json
import torch
import sys

from subprocess import run
from data.batch import Batch

sys.path.append("../evaluation")
from evaluate_single_dataset import evaluate

graph_modes_mapping = {
    "ace_p_evt": "ace", "ace_p_evt_ent": "evt-ent",
    "ace_pp_evt": "ace", "ace_pp_evt_ent": "evt-ent",
    "ere_p_evt": "ace", "ere_p_evt_ent": "evt-ent",
    "ere_pp_evt": "ace", "ere_pp_evt_ent": "evt-ent",
}


def sentence_condition(s, f, l):
    return ("framework" not in s or f == s["framework"]) and ("framework" in s or f in s["targets"])

def predict(model, data, input_paths, raw_input_paths, args, logger, output_directory, device, mode="validation", epoch=None):
    model.eval()

    input_files = {(f, l): input_paths[(f, l)] for f, l in args.frameworks}
    if raw_input_paths:
        raw_files = {(f, l): raw_input_paths[(f, l)] for f, l in args.frameworks}

    sentences = {(f, l): {} for f, l in args.frameworks}
    for framework, language in args.frameworks:
        with open(input_files[(framework, language)], encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line)

                #if not sentence_condition(line, framework, language):
                #    continue

                line['nodes'], line['edges'], line['tops'] = [], [], []
                line['framework'], line['language'] = framework, language
                sentences[(framework, language)][line['id']] = line

    for i, batch in enumerate(data):
        with torch.no_grad():
            all_predictions = model(Batch.to(batch, device), inference=True)

        for (framework, language), predictions in all_predictions.items():
            for prediction in predictions:
                for key, value in prediction.items():
                    sentences[(framework, language)][prediction['id']][key] = value
    
    for framework, language in args.frameworks:

        if epoch is not None:
            output_path = f"{output_directory}/prediction_{mode}_{epoch}_{framework}_{language}.json"
        else:
            output_path = f"{output_directory}/prediction_{framework}_{language}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            for sentence in sentences[(framework, language)].values():
                json.dump(sentence, f, ensure_ascii=False)
                f.write("\n")
                f.flush()

        graphMode = graph_modes_mapping[framework]

        run(["./convert.sh", graphMode, output_path])

        if raw_input_paths:
            raw_file = raw_files[(framework, language)]
            _evt_pred = 'evt' in graphMode or 'ace' in graphMode
            _ent_pred = 'ent' in graphMode

            results = evaluate(raw_file, f"{output_path}_converted", ent_pred=_ent_pred, evt_pred=_evt_pred, lang=language)
            print(mode, framework, language, results, flush=True)

            if logger is not None:
                logger.log_evaluation(results, mode, epoch, framework, language)

            if mode == 'test':
                experimental_results_f = f"{args.output_directory}/experimental_results.txt"
                _run_results = {
                    f"{args.name}:{mode}:{language}": results
                }

                with open(experimental_results_f, 'a') as f:
                    f.write(json.dumps(_run_results))
                    f.write('\n')
