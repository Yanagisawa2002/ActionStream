"""Token-trie slot decoding: structure and enum vocabulary, never truth labels."""

import json


def evidence_slots():
    slots = [["{"]]
    for view in range(2):
        for entity in ("t", "b"):
            slots.append([f'"{entity}{view}":['])
            for coordinate in range(4):
                delimiter = "," if coordinate < 3 else "],"
                slots.append([str(n) + delimiter for n in range(1001)])
        for key, values in (
            ("i", ("match", "other", "unknown")),
            ("j", ("match", "other", "unknown")),
            ("v", ("visible", "partial", "absent", "unknown")),
            ("w", ("visible", "partial", "absent", "unknown")),
            ("r", ("inside", "outside", "unknown")),
        ):
            slots.append([f'"{key}{view}":'])
            delimiter = "}" if view == 1 and key == "r" else ","
            slots.append([json.dumps(value) + delimiter for value in values])
    return slots


def status_slots():
    return [['{"status":'], ['"complete"}', '"incomplete"}', '"unknown"}']]


class SlotDecoder:
    def __init__(self, tokenizer, slots, prefix_length):
        self.prefix_length = prefix_length
        self.eos = tokenizer.eos_token_id
        self.tries = []
        cache = {}
        for options in slots:
            key = tuple(options)
            if key not in cache:
                trie = {}
                for option in options:
                    node = trie
                    for token in tokenizer.encode(option, add_special_tokens=False):
                        node = node.setdefault(token, {})
                    node[None] = True
                cache[key] = trie
            self.tries.append(cache[key])

    def __call__(self, batch_id, input_ids):
        generated = input_ids[self.prefix_length :].tolist()
        position = 0
        for trie in self.tries:
            node = trie
            while None not in node:
                if position == len(generated):
                    return list(node)
                token = generated[position]
                if token not in node:
                    raise ValueError("Generated prefix escaped the frozen slot schema")
                node = node[token]
                position += 1
        return [self.eos]
