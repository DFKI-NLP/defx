import csv
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Any
from zipfile import ZipFile

from allennlp.predictors import Predictor
from allennlp.data.instance import Instance
from tqdm import tqdm


class PredictionsWriter:
    def __init__(self,
                 input_data: str,
                 output_dir: str,
                 model_archive: str,
                 predictor: str,
                 subtasks: List[int],
                 batch_size: int = 16,
                 cuda_device: int = -1):
        self._predictor = Predictor.from_path(model_archive,
                                              predictor,
                                              cuda_device=cuda_device)
        self._input_data = input_data
        self._output_dir = output_dir
        self._subtasks = subtasks
        self._batch_size = batch_size

    def missing_predictions(self):
        for subtask in self._subtasks:
            if not any(self._output_dir.glob(f'task_{subtask}*')):
                return True
        return False

    def run(self):
        logging.info('Reading instances...')
        instances = self._predictor._dataset_reader.read(self._input_data)

        logging.info('Running prediction...')
        results = self._batched_predict_instances(instances)
        predicted_instances = list(zip(instances, results))

        logging.info('Writing predictions...')
        Path(self._output_dir).mkdir(exist_ok=True)
        if 1 in self._subtasks:
            input_files = self._input_data.glob('*.deft')
            self._write_task1_predictions(input_files, predicted_instances)
        if 2 in self._subtasks:
            self._write_task2_predictions(predicted_instances)
        if 3 in self._subtasks:
            self._write_task3_predictions(self._output_dir, predicted_instances)

        logging.info('Creating archive...')
        with ZipFile(Path(self._output_dir, 'submission.zip'), 'w') as zf:
            for pred_file in Path(self._output_dir).iterdir():
                if pred_file.suffix == '.deft':
                    zf.write(pred_file, pred_file.name)

        logging.info('Done.')

    def _batched_predict_instances(self,
                                   examples: List[Instance]
                                   ) -> List[Dict[str, Any]]:
        results = []  # type: List[Dict[str, Any]]
        for i in tqdm(range(0, len(examples), self._batch_size)):
            batch_examples = examples[i: i + self._batch_size]
            batch_results = self._predictor.predict_batch_instance(batch_examples)
            results.extend(batch_results)
        return results

    def _write_task1_predictions(self, input_files, predicted_instances):
        sentences = []
        source_files = []
        source_rows = []
        for input_file in input_files:
            sents, files, rows = self._read_task1_input_file(input_file)
            sentences.extend(sents)
            source_files.extend(files)
            source_rows.extend(rows)

        assert len(sentences) == len(predicted_instances)
        instances, results = zip(*predicted_instances)
        predictions = zip(source_files, source_rows, sentences, instances, results)
        predictions_by_origin = itertools.groupby(predictions,
                                                  lambda x: x[0])
        for input_file, prediction_group in predictions_by_origin:
            output_file_name = f'task_1_{input_file}'
            output_file = Path(self._output_dir, output_file_name)
            self._write_subtask1_output_file(output_file, prediction_group)

    @staticmethod
    def _write_subtask1_output_file(output_file, predictions):
        with output_file.open('w') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='"',
                                quoting=csv.QUOTE_ALL)
            for prediction in predictions:
                input_file, input_row, text, instance, result = prediction
                expected_origin = f'{input_file}##{input_row}'
                label = 1 if result.get('label') == 'HasDef' else 0
                assert instance.get('origin').metadata == expected_origin
                writer.writerow([text, label])

    @staticmethod
    def _read_task1_input_file(input_file):
        sentences = []
        source_files = []
        source_rows = []
        with input_file.open() as f:
            for idx, row in enumerate(
                csv.reader(f, delimiter='\t', quotechar='"')
            ):
                source_files.append(input_file.name)
                source_rows.append(idx)
                sentences.append(row[0])
        return sentences, source_files, source_rows

    def _write_task2_predictions(self, predicted_instances):
        predictions_by_file = itertools.groupby(predicted_instances,
                                                key=self.get_file_name_from_iterator)
        for input_file, prediction_group in predictions_by_file:
            output_file_name = f'task_2_{input_file}'
            output_file = Path(self._output_dir, output_file_name)
            self._write_subtask2_output_file(output_file, list(prediction_group))

    def _write_subtask2_output_file(self, output_file, predictions):
        with output_file.open('w') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='"',
                                quoting=csv.QUOTE_ALL)
            for prediction in predictions:
                instance, results = prediction
                source_file = self._get_instance_source_file(instance)
                for word, tag in zip(results['words'], results['tags']):
                    writer.writerow([word, source_file, 666, 666, tag])
                writer.writerow([])

    def _write_task3_predictions(self, output_dir, predicted_instances):
        predictions_by_file = itertools.groupby(predicted_instances,
                                                key=self.get_file_name_from_iterator)
        for input_file, prediction_group in predictions_by_file:
            output_file_name = f'task_3_{input_file}'
            output_file = Path(output_dir, output_file_name)
            self._write_subtask3_output_file(output_file, list(prediction_group))

    def _write_subtask3_output_file(self, output_file, predictions):
        with output_file.open('w') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='"',
                                quoting=csv.QUOTE_ALL)
            for prediction in predictions:
                instance, results = prediction
                source_file = self._get_instance_source_file(instance)
                words = results['words']
                if 'ner_ids' in results:
                    ner_ids = results['ner_ids']
                else:
                    ner_ids = ['-1'] * len(words)
                if 'heads' in results:
                    predicted_heads = results['heads']
                else:
                    predicted_heads = results['head_offsets']

                fields = zip(words,
                             results['tags'],
                             ner_ids,
                             predicted_heads,
                             results['relations'])
                for word, tag, ner_id, head, relation in fields:
                    writer.writerow(
                        [word, source_file, 666, 666, tag, ner_id, head, relation]
                    )
                writer.writerow([])

    @staticmethod
    def get_file_name_from_iterator(i):
        return PredictionsWriter._get_instance_source_file(i[0])

    @staticmethod
    def _get_instance_source_file(instance):
        return instance.get('metadata')['example_id'].split('##')[0]
