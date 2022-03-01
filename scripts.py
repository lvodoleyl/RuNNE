import numpy as np
from natasha import Doc, NewsEmbedding, NewsSyntaxParser, Segmenter, MorphVocab, NewsMorphTagger
from pathlib import Path


class NatashaInfo:

    def __init__(self):
        self.emb = NewsEmbedding()
        self.segmenter = Segmenter()
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.morph_vocab = MorphVocab()
        self.morph_tagger = NewsMorphTagger(self.emb)

    def convert_format_to_natasha(self, text):
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.parse_syntax(self.syntax_parser)
        return doc

    def lemmatization(self, doc):
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)

    def get_vector_tokens(self, doc):
        vectors = []
        for token in doc.tokens:
            vectors.append(self.emb.get(token.lemma, np.zeros(300)))
        return np.array(vectors)


def create_syntax_matrix(doc, symmetry=False):
    data = {
      'sentences': [],
      'dep': []
    }

    for sent in doc.sents:
        words = []
        syntax = [[] for _ in range(len(sent.tokens))]
        for token in sent.tokens:
            words.append(token.text)
            to_ = int(token.id.split('_')[1]) - 1
            from_ = int(token.head_id.split('_')[1]) - 1
            if from_ >= 0:
                syntax[from_].append(to_)
            if symmetry:
                syntax[to_].append(from_)
        data['sentences'].append(words)
        data['dep'].append({'nodes': syntax})
    return data


def get_data_ner(doc, ann, doc_name):
    data = {
        'ner': [[] for _ in range(len(doc.sents))],
        'relations': [[] for _ in range(len(doc.sents))]
    }
    for line in ann:
        try:
            other_part, from_other, to_other = None, None, None
            num, meta, text = line.split('	')
            if ';' in meta:
                meta, other_part = meta.split(';')
            class_, from_, to_ = meta.split(' ')
            from_, to_ = int(from_), int(to_)
            if other_part:
                from_other, to_other = other_part.split(' ')
                from_other, to_other = int(from_other), int(to_other)
        except Exception as e:
            if line:
                print('*' * 20, f'\n Прощена сл.сущность: {line}')
            continue
        for _from_, _to_ in [[from_, to_], [from_other, to_other]]:
            if _from_ is None or _to_ is None:
                continue
            ner_info = []
            for i, token in enumerate(doc.tokens):
                if _from_ == token.start:
                    sents, info_id = token.id.split('_')
                    ner_info.append(i)
                if _to_ in [token.stop, token.stop + 1, token.stop - 1] and len(ner_info) > 0:
                    sents, info_id = token.id.split('_')
                    ner_info.append(i)
                if len(ner_info) > 1:
                    break
            ner_info.append(class_ + '+' if from_other else class_)
            if len(ner_info) == 3:
                data['ner'][int(sents) - 1].append(ner_info)
            else:
                print(
                    f"{'*' * 20}\nЧто-то не так с поиском NER в тексте, {ner_info}, \nNER: {_from_}-{_to_}, \nline: {line}, \nsent: {doc.sents[int(sents) - 1].text}")
                print(
                    f'Tokens: {[(token.text, token.id, token.start, token.stop) for token in doc.sents[int(sents) - 1].tokens]}')
                print(doc_name)

    for num, sent in enumerate(data['ner']):
        relations_for_sent1, relations_for_sent2 = [], []
        for num_ner_1 in range(len(sent) - 1):
            for num_ner_2 in range(num_ner_1 + 1, len(sent)):
                if sent[num_ner_1][2] == sent[num_ner_2][2] and '+' in sent[num_ner_1][2]:
                    sent[num_ner_1][2] = sent[num_ner_1][2].replace('+', '')
                    sent[num_ner_2][2] = sent[num_ner_2][2].replace('+', '')
                    relations_for_sent1 = sent[num_ner_1][:2] + sent[num_ner_2][:2] + ['Overlap']
                    relations_for_sent2 = sent[num_ner_2][:2] + sent[num_ner_1][:2] + ['Overlap']
                if sent[num_ner_1][0] <= sent[num_ner_2][0] <= sent[num_ner_2][1] <= sent[num_ner_1][1] or (
                        sent[num_ner_2][0] <= sent[num_ner_1][0] <= sent[num_ner_1][1] <= sent[num_ner_2][1]
                ):
                    relations_for_sent1 = sent[num_ner_1][:2] + sent[num_ner_2][:2] + ['Combined']
                    relations_for_sent2 = sent[num_ner_2][:2] + sent[num_ner_1][:2] + ['Combined']
        if relations_for_sent1 and relations_for_sent2:
            data['relations'][num].extend([relations_for_sent1, relations_for_sent2])

    return data


def create_dataset(path: Path, natasha_: NatashaInfo, ignore_empty_sent=False) -> list:
    dataset = []
    couple = dict()
    files = [x for x in path.glob('*') if x.is_file()]
    for file_ in files:
        if file_.name[:-3] not in couple:
          couple[file_.name[:-3]] = dict()
        couple[file_.name[:-3]][file_.suffix] = file_
    for name_key in couple.keys():
        if '.ann' in couple[name_key] and '.txt' in couple[name_key]:
            ann = couple[name_key]['.ann'].read_text().split('\n')
            text = couple[name_key]['.txt'].read_text()
            doc = natasha_.convert_format_to_natasha(text.replace('-', ' '))
            natasha_.lemmatization(doc)
            result = {**create_syntax_matrix(doc, True), **get_data_ner(doc, ann, name_key),
                      'doc_key': str(path)+name_key+'.txt'}
            if ignore_empty_sent:
                ner = []
                relations = []
                sentences = []
                dep = []
                last_end = 0
                for n, r, s, d in zip(result["ner"], result["relations"], result["sentences"], result["dep"]):
                    if len(n) > 0:
                        pos_shift = last_end - n[0][0]
                        for n_item in n:
                            n_item[0] = n_item[0] + pos_shift
                            n_item[1] = n_item[1] + pos_shift
                        for r_item in r:
                            r_item[0] = r_item[0] + pos_shift
                            r_item[1] = r_item[1] + pos_shift
                            r_item[2] = r_item[2] + pos_shift
                            r_item[3] = r_item[3] + pos_shift
                        last_end = last_end + len(s)

                        ner.append(n)
                        relations.append(r)
                        sentences.append(s)
                        dep.append(d)
                result["ner"] = ner
                result["relations"] = relations
                result["sentences"] = sentences
                result["dep"] = dep
            dataset.append(result)

    return dataset


def create_inference_dataset(json_file, natasha_: NatashaInfo):
    from json import loads
    dataset = []
    with open(json_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                text = loads(line)
                doc = natasha_.convert_format_to_natasha(text["sentences"].replace('-', ' '))
                natasha_.lemmatization(doc)
                syntax_matrix = create_syntax_matrix(doc, True)
                tokens_pos = [[(token.start, token.end)] for token in sent for sent in doc.sents]
                data_item = {"doc_key": text["id"], "tokens_pos": tokens_pos,
                             "sentences": syntax_matrix["sentences"],
                             "dep": syntax_matrix['dep']}
                dataset.append(data_item)
    return dataset


def convert_result_to_runne_jsonl(dataset_file, result_file):
    from json import loads
    result_list = []
    with open(dataset_file, 'r', encoding='utf-8') as dataset:
        with open(result_file, 'r', encoding='utf-8') as result:
            for item_d, item_r in zip(dataset, result):
                item_d = loads(item_d)
                item_r = loads(item_r)
                ners = []
                tokens_pos = item_d["tokens_pos"]
                for i, sent in enumerate(item_r):
                    for obj in sent["ner"]:
                        # позиция начала, позиция последнего символа, класс
                        ners.append([tokens_pos[i][obj[0]][0], tokens_pos[i][obj[1]][1]-1, obj[2]])
                res_item = {"id": item_d["doc_key"], "ners": ners }
                result_list.append(res_item)
    return result_list


def join_result(result_for_f1, result_for_few):
    result_for_f1_copy = result_for_f1.copy()
    result_for_few_copy = result_for_few.copy()
    result_after_decision_rules = []
    for f1, few in zip(result_for_f1_copy, result_for_few_copy):
        local_ner = []
        for ner_local_f1 in f1['ners']:
            if ner_local_f1[2] not in ['DISEASE', 'WORK_OF_ART', 'PENALTY']:
                local_ner.append(ner_local_f1)
        for ner_local_few in few['ners']:
            if ner_local_few[2] in ['DISEASE', 'WORK_OF_ART', 'PENALTY']:
                flag = True
                for local in local_ner:
                    if local == ner_local_few:
                        flag = False 
                    if local[0] == ner_local_few[0] and local[1] == ner_local_few[1]:
                        flag = False
                        local[2] = ner_local_few[2]
                if flag:
                    local_ner.append(ner_local_few)
        result_after_decision_rules.append({'id': f1['id'], 'ners': local_ner})
    return result_after_decision_rules

if __name__ == "__main__":
    from pathlib import Path
    dataset_path = "../../few_shot"
    dataset = create_dataset(Path(dataset_path), NatashaInfo(), ignore_empty_sent=True)
    print(dataset)