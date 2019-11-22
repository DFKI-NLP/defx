
class DisplacyFormatter:
    def format(self, prediction):
        words = [
            {'text': w, 'tag': t}
            for w, t in zip(prediction['words'],
                            prediction['tags'])
        ]
        arcs = []
        relations = zip(range(len(prediction['words'])),
                        prediction['head_offsets'],
                        prediction['relations'])
        for tail, head, relation in relations:
            if relation == '0':
                continue
            if head > tail:
                arcs.append({
                    "start": tail, "end": head, "label": relation, "dir": "left"
                })
            else:
                arcs.append({
                    "start": head, "end": tail, "label": relation, "dir": "right"
                })
        output = {
            'words': words,
            'arcs': arcs
        }
        return output
